use std::time::Instant;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::Solve;
use ocl::{Device, Platform, core::DeviceInfo};
use registration_opencl::{
    convert_dtoh::convert_dtoh,
    gpu_cov::OclCovContext,
    gpu_gicp::OclGicpContext,
    gpu_search::OclSearchContext,
    gpu_transform::OclTransformContext,
    gpu_voxel::OclVoxelContext,
    ocl_context::OclRuntime,
    operate_pcd_file::{
        PointXYZ, PointXYZRGB, PointXYZT, load_pcd_xyz, load_pcd_xyzrgb, load_pcd_xyzt,
        save_pcd_xyzrgb,
    },
};

const VOXEL_SIZE: f32 = 0.5;
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;
const GICP_MAX_ITERATIONS: usize = 120;

fn main() -> Result<()> {
    check_device_info()?;

    println!("=== Initializing GPU context ===");
    let ocl_runtime = OclRuntime::new().expect("Failed to create OclRuntime");
    println!("Using Platform: {}", ocl_runtime.platform.name()?);
    println!("Using Device:   {}", ocl_runtime.device.name()?);

    let mut gpu_voxel =
        OclVoxelContext::new(ocl_runtime.clone()).expect("Failed to create OclVoxelContext");
    let mut gpu_covs =
        OclCovContext::new(ocl_runtime.clone()).expect("Failed to create OclCovContext");
    let mut gpu_transform = OclTransformContext::new(ocl_runtime.clone())
        .expect("Failed to create OclTransformContext");
    let mut gpu_search =
        OclSearchContext::new(ocl_runtime.clone()).expect("Failed to create OclSearchContext");
    let mut gpu_gicp =
        OclGicpContext::new(ocl_runtime.clone()).expect("Failed to create OclGicpContext");

    let source_pcd_path = "data/input/aist/vggt-sansouken-room-scale-7_5_voxel_025_xyz_only.pcd";
    let source_pcd = load_pcd_xyz(source_pcd_path).expect("Failed to load initial PCD file");
    let source_pts = pcd_to_array2(&source_pcd);

    let target_pcd_path = "data/input/aist/aist-voxelized-025.pcd";
    let target_pcd = load_pcd_xyz(target_pcd_path).expect("Failed to load initial PCD file");
    let target_pts = pcd_to_array2(&target_pcd);

    println!("\n=== Loaded PCD files ===");
    println!("Source points: {}", source_pts.nrows());
    println!("Loaded source PCD from: {}", source_pcd_path);
    println!("Target points: {}", target_pts.nrows());
    println!("Loaded target PCD from: {}", target_pcd_path);

    // Center the source points to target points
    let (_, overlaped_source_pts) = registration_pcd_center(&source_pts, &target_pts);

    // Voxel downsample
    let (d_v_source_pts, v_source_pts_num) = gpu_voxel
        .voxel_downsample(
            &overlaped_source_pts,
            overlaped_source_pts.nrows(),
            VOXEL_SIZE,
        )
        .context("Failed to compute voxel")?;
    println!(
        "Voxel downsampled source points: {} -> {}",
        source_pts.nrows(),
        v_source_pts_num
    );

    let (d_v_target_pts, v_target_pts_num) = gpu_voxel
        .voxel_downsample(&target_pts, target_pts.nrows(), VOXEL_SIZE)
        .context("Failed to compute voxel")?;
    println!(
        "Voxel downsampled target points: {} -> {}",
        target_pts.nrows(),
        v_target_pts_num
    );

    // Compute covariances
    let d_source_covs = gpu_covs
        .compute_covariances(&d_v_source_pts, v_source_pts_num)
        .context("Failed to compute covariances")?;
    let d_target_covs = gpu_covs
        .compute_covariances(&d_v_target_pts, v_target_pts_num)
        .context("Failed to compute covariances")?;


    let mut transform_matrix = Array2::<f32>::eye(4);


    // Iterations of GICP
    for i in 0..GICP_MAX_ITERATIONS {
        println!("\n=== GICP Iteration {} ===", i + 1);
    

        // Transform each points
        let (d_transformed_source_pts, d_transformed_source_covs) = gpu_transform
            .apply_transform(
                &d_v_source_pts,
                &d_source_covs,
                v_source_pts_num,
                &transform_matrix,
            )
            .context("Failed to apply transform")?;

        // Compute to find nearest neighbor pts
        let (d_indices, d_dists_sq, indices, dists_sq) = gpu_search
            .compute_find_nearest_neighbor(
                &d_transformed_source_pts,
                v_source_pts_num,
                &d_v_target_pts,
                v_target_pts_num,
            )
            .context("Failed to compute nearest neighbor")?;

        // Debug
        let max_dist2: f32 = VOXEL_SIZE * VOXEL_SIZE * 2.0;
        let valid_pairs = indices
            .iter()
            .zip(dists_sq.iter())
            .filter(|(idx, dist)| **idx >= 0 && **dist <= max_dist2)
            .count();
        println!(
            "Iteration {}: Found {} nearest neighbor correspondences",
            i + 1, valid_pairs
        );

        // Compute GICP
        let (h_matrix, b_vector) = gpu_gicp
            .compute_gicp(
                &d_transformed_source_pts,
                &d_transformed_source_covs,
                v_source_pts_num,
                &d_v_target_pts,
                &d_target_covs,
                v_target_pts_num,
                &d_indices,
                &d_dists_sq,
                max_dist2,
            )
            .context("Failed to calculate GICP")?;

        let delta_t = solve_linear_system_6x6(h_matrix, b_vector)?;
        transform_matrix = mat4_mul( &delta_t, &transform_matrix);
        // println!("Computed delta transform:\n{:?}", delta_t);
        println!("Updated transform matrix:\n{:?}", transform_matrix);

        // Check convergence (RMSE)
        let mut sum = 0.0f32;
        let mut cnt = 0usize;

        for j in 0..v_source_pts_num {
            let idx = indices[j];
            if idx < 0 {
                continue;
            }
            if dists_sq[j] > max_dist2 {
                continue;
            }
            sum += dists_sq[j];
            cnt += 1;
        }
        let rmse = (sum / cnt as f32).sqrt();
        println!("Iteration {}: RMSE = {}", i + 1, rmse);

        // break;
    }

    // <!--- DEBUG --->
    let transformed_original_source_pts = convert_dtoh(&d_v_source_pts, v_source_pts_num)
        .context("Failed to convert device to host")?;

    // let transformed_original_source_pcd =
    //     convert_array2_to_pcd(&transformed_original_source_pts, 255, 0, 0); // Red color
    let transformed_original_source_pcd =
        convert_array2_to_pcd(&overlaped_source_pts, 255, 0, 0); // Red color
    // <!--- DEBUG --->

    let (d_final_transformed_source_pts, d_final_transformed_source_covs) = gpu_transform
        .apply_transform(&d_v_source_pts, &d_source_covs, v_source_pts_num, &transform_matrix)
        .context("Failed to apply transform")?;

    // <!--- DEBUG --->
    let final_transformed_source_pts =
        convert_dtoh(&d_final_transformed_source_pts, v_source_pts_num)
            .context("Failed to convert device to host")?;

    let final_transformed_source_pcd =
        convert_array2_to_pcd(&final_transformed_source_pts, 0, 255, 0); // Green color
    let target_pcd = convert_array2_to_pcd(&target_pts, 0, 0, 255); // Blue color
    
    // let debug_source_pcd = convert_array2_to_pcd(&debug_source_pts, 255, 0, 255); // Blue color
    // let mut combined_pcd = debug_source_pcd.clone();
    let mut combined_pcd = transformed_original_source_pcd.clone();
    // combined_pcd.extend(transformed_original_source_pcd);
    combined_pcd.extend(final_transformed_source_pcd);
    combined_pcd.extend(target_pcd);

    let output_path = format!("data/output/test/test-gicp-v-{}-iter-{}.pcd", VOXEL_SIZE, GICP_MAX_ITERATIONS);
    save_pcd_xyzrgb(
        &combined_pcd, &output_path)?;

    // <!--- DEBUG --->

    Ok(())
}

fn inverse_transform(m: &Array2<f32>) -> Array2<f32> {
    let mut inv = Array2::<f32>::eye(4);
    
    // 回転成分の転置 (R^T)
    for r in 0..3 {
        for c in 0..3 {
            inv[[r, c]] = m[[c, r]];
        }
    }

    // 平行移動成分の逆変換 (t' = -R^T * t)
    for r in 0..3 {
        let mut sum = 0.0;
        for c in 0..3 {
            sum += inv[[r, c]] * m[[c, 3]];
        }
        inv[[r, 3]] = -sum;
    }
    
    inv
}

fn calculate_centroid(pts: &Array2<f32>) -> Array1<f32> {
    // let n = pts.nrows() as f32;
    // let mut centroid = Array1::<f32>::zeros(3);

    // for i in 0..n as usize {
    //     centroid[0] += pts[[i, 0]];
    //     centroid[1] += pts[[i, 1]];
    //     centroid[2] += pts[[i, 2]];
    // }
    // centroid /= n;

    // centroid

    pts.mean_axis(ndarray::Axis(0)).unwrap()
}

fn convert_array2_to_pcd(points: &Array2<f32>, r: u8, g: u8, b: u8) -> Vec<PointXYZRGB> {
    let n = points.nrows();
    let mut pcd_points = Vec::with_capacity(n);

    // RGB を uint32 としてパック
    let rgb_packed = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    let rgb_as_f32 = f32::from_bits(rgb_packed);

    for i in 0..n {
        let pt = PointXYZRGB {
            x: points[[i, 0]],
            y: points[[i, 1]],
            z: points[[i, 2]],
            rgb: rgb_as_f32,
        };
        pcd_points.push(pt);
    }

    pcd_points
}

fn mat4_mul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // a(4x4) * b(4x4)
    let mut out = Array2::<f32>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                s += a[[i, k]] * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn solve_linear_system_6x6(a: Array2<f64>, b: Array1<f64>) -> Result<Array2<f32>> {
    let x = a
        .solve(&b)
        .or_else(|_| Err(anyhow::anyhow!("Linear solve failed")))?;

    // x = [alpha, beta, gamma, tx, ty, tz]
    let delta_matrix = convert_se3_to_matrix4(x);
    Ok(delta_matrix)
}

// [alpha, beta, gamma, tx, ty, tz] -> 4x4 matrix
fn convert_se3_to_matrix4(x: Array1<f64>) -> Array2<f32> {
    let alpha = x[0];
    let beta = x[1];
    let gamma = x[2];
    let tx = x[3];
    let ty = x[4];
    let tz = x[5];

    let theta = (alpha * alpha + beta * beta + gamma * gamma).sqrt();
    let r: Array2<f64>;

    if theta < 1e-9 {
        r = ndarray::array![
            [1.0, -gamma, beta],
            [gamma, 1.0, -alpha],
            [-beta, alpha, 1.0]
        ];
    } else {
        let k_x = alpha / theta;
        let k_y = beta / theta;
        let k_z = gamma / theta;
        let c = theta.cos();
        let s = theta.sin();
        let v = 1.0 - c;

        r = ndarray::array![
            [
                k_x * k_x * v + c,
                k_x * k_y * v - k_z * s,
                k_x * k_z * v + k_y * s
            ],
            [
                k_x * k_y * v + k_z * s,
                k_y * k_y * v + c,
                k_y * k_z * v - k_x * s
            ],
            [
                k_x * k_z * v - k_y * s,
                k_y * k_z * v + k_x * s,
                k_z * k_z * v + c
            ]
        ];
    }

    ndarray::array![
        [
            r[[0, 0]] as f32,
            r[[0, 1]] as f32,
            r[[0, 2]] as f32,
            tx as f32
        ],
        [
            r[[1, 0]] as f32,
            r[[1, 1]] as f32,
            r[[1, 2]] as f32,
            ty as f32
        ],
        [
            r[[2, 0]] as f32,
            r[[2, 1]] as f32,
            r[[2, 2]] as f32,
            tz as f32
        ],
        [0.0, 0.0, 0.0, 1.0]
    ]
}

// fn pcd_to_array2(pcd_points: &[PointXYZRGB]) -> Array2<f32> {
    // fn pcd_to_array2(pcd_points: &[PointXYZT]) -> Array2<f32> {
fn pcd_to_array2(pcd_points: &[PointXYZ]) -> Array2<f32> {
    let n = pcd_points.len();
    let mut arr = Array2::<f32>::zeros((n, 3));

    for (i, pt) in pcd_points.iter().enumerate() {
        arr[[i, 0]] = pt.x;
        arr[[i, 1]] = pt.y;
        arr[[i, 2]] = pt.z;
    }

    arr
}

fn check_device_info() -> Result<()> {
    for plat in Platform::list() {
        println!("=== Platform: {} ===", plat.name()?);

        let devices = Device::list_all(plat)?;
        for dev in devices {
            let name = dev.name()?;
            let dtype = dev.info(DeviceInfo::Type)?.to_string();
            let version = dev.version()?;

            // 拡張一覧（長い文字列）
            let exts = dev.info(DeviceInfo::Extensions)?.to_string();

            let has_float_atomics = exts.split_whitespace().any(|e| e == "cl_ext_float_atomics");

            println!("Device: {}", name);
            println!("  Type: {}", dtype);
            println!("  Version: {}", version);
            println!("  cl_ext_float_atomics: {}", has_float_atomics);

            // ついでに “似た名前” を含む拡張も拾いたい場合
            let related: Vec<&str> = exts
                .split_whitespace()
                .filter(|e| e.contains("float") && e.contains("atomic"))
                .collect();
            if !related.is_empty() {
                println!("  related: {:?}", related);
            }
            println!();
        }
    }
    Ok(())
}

fn registration_pcd_center(
    source_points: &Array2<f32>,
    target_points: &Array2<f32>
) -> (Array2<f32>, Array2<f32>) {
    // Calculate centroids
    let source_centroid = source_points.mean_axis(Axis(0)).unwrap();
    let target_centroid = target_points.mean_axis(Axis(0)).unwrap();

    // Calculate translation
    let translation = &target_centroid - &source_centroid;

    // Create transformation matrix
    let mut transform = Array2::<f32>::eye(4);
    transform[[0, 3]] = translation[0];
    transform[[1, 3]] = translation[1];
    transform[[2, 3]] = translation[2];
    
    let n_points = source_points.nrows();
    let mut source_copy = Array2::<f32>::ones((n_points, 4));
    source_copy.slice_mut(s![.., 0..3]).assign(source_points);

    let transformed = source_copy.dot(&transform.t());
    let transformed_source = transformed.slice(s![.., 0..3]).to_owned();
    (transform, transformed_source)
}