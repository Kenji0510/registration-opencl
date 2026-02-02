use std::time::Instant;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
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

const VOXEL_SIZE: f32 = 0.25;
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;
const GICP_MAX_ITERATIONS: usize = 5;

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

    let source_pcd_path = "data/input/merged_until_650-20251205-02-H927.pcd";
    let source_pcd = load_pcd_xyzrgb(source_pcd_path).expect("Failed to load initial PCD file");
    let source_pts = pcd_to_array2(&source_pcd);

    let target_pcd_path = "data/input/merged_until_650-20251205-02-H927.pcd";
    let target_pcd = load_pcd_xyzrgb(target_pcd_path).expect("Failed to load initial PCD file");
    let target_pts = pcd_to_array2(&target_pcd);

    println!("\n=== Loaded PCD files ===");
    println!("Source points: {}", source_pts.nrows());
    println!("Loaded source PCD from: {}", source_pcd_path);
    println!("Target points: {}", target_pts.nrows());
    println!("Loaded target PCD from: {}", target_pcd_path);

    // <!--- DEBUG --->
    // Transform each points
    // 90度回転（Z軸周り）+ X方向に2m移動
    let mut transform = Array2::<f32>::eye(4);
    let angle = std::f32::consts::FRAC_PI_2; // 90度
    transform[[0, 0]] = angle.cos(); // cos(90°) = 0
    transform[[0, 1]] = -angle.sin(); // -sin(90°) = -1
    transform[[1, 0]] = angle.sin(); // sin(90°) = 1
    transform[[1, 1]] = angle.cos(); // cos(90°) = 0
    transform[[0, 3]] = 2.0; // X方向に2m移動

    let mut source_pts_transformed = Array2::<f32>::zeros(source_pts.dim());
    for i in 0..source_pts.nrows() {
        let x = source_pts[[i, 0]];
        let y = source_pts[[i, 1]];
        let z = source_pts[[i, 2]];
        
        source_pts_transformed[[i, 0]] = transform[[0, 0]] * x + transform[[0, 1]] * y + transform[[0, 2]] * z + transform[[0, 3]];
        source_pts_transformed[[i, 1]] = transform[[1, 0]] * x + transform[[1, 1]] * y + transform[[1, 2]] * z + transform[[1, 3]];
        source_pts_transformed[[i, 2]] = transform[[2, 0]] * x + transform[[2, 1]] * y + transform[[2, 2]] * z + transform[[2, 3]];
    }
    let source_pts = source_pts_transformed;
    let debug_source_pts = source_pts.clone();
    // <!--- DEBUG --->

    // let source_centroid = calculate_centroid(&source_pts);
    let source_centroid = calculate_centroid(&source_pts);
    let target_centroid = calculate_centroid(&target_pts);

    println!(
        "Source centroid: ({:.3}, {:.3}, {:.3})",
        source_centroid[0], source_centroid[1], source_centroid[2]
    );
    println!(
        "Target centroid: ({:.3}, {:.3}, {:.3})",
        target_centroid[0], target_centroid[1], target_centroid[2]
    );

    // Move source centroid to target centroid
    let translation = &target_centroid - &source_centroid;
    println!(
        "Initial translation to align centroids: ({:.3}, {:.3}, {:.3})",
        translation[0], translation[1], translation[2]
    );

    let overlaped_source_pts = &source_pts + &translation;

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

    // // Transform each points
    // // 90度回転（Z軸周り）+ X方向に2m移動
    // let mut transform = Array2::<f32>::eye(4);
    // let angle = std::f32::consts::FRAC_PI_2; // 90度
    // transform[[0, 0]] = angle.cos(); // cos(90°) = 0
    // transform[[0, 1]] = -angle.sin(); // -sin(90°) = -1
    // transform[[1, 0]] = angle.sin(); // sin(90°) = 1
    // transform[[1, 1]] = angle.cos(); // cos(90°) = 0
    // transform[[0, 3]] = 2.0; // X方向に2m移動

    // // <!--- DEBUG --->
    // let (d_transformed_source_pts, d_transformed_source_covs) = gpu_transform
    //     .apply_transform(
    //         &d_v_source_pts,
    //         &d_source_covs,
    //         v_source_pts_num,
    //         &transform,
    //     )
    //     .context("Failed to apply transform")?;
    // // <!--- DEBUG --->

    // Compute to find nearest neighbor pts
    let (d_indices, d_dists_sq, indices, dists_sq) = gpu_search
        .compute_find_nearest_neighbor(
            &d_v_source_pts,
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
        1, valid_pairs
    );

    // Compute GICP
    let (h_matrix, b_vector) = gpu_gicp
        .compute_gicp(
            &d_v_source_pts,
            &d_source_covs,
            v_source_pts_num,
            &d_v_target_pts,
            &d_target_covs,
            v_target_pts_num,
            &d_indices,
            &d_dists_sq,
            VOXEL_SIZE * VOXEL_SIZE,
        )
        .context("Failed to calculate GICP")?;

    let delta_t = solve_linear_system_6x6(h_matrix, b_vector)?;
    println!("Computed delta transform:\n{:?}", delta_t);

    // <!--- DEBUG --->
    let transformed_original_source_pts = convert_dtoh(&d_v_source_pts, v_source_pts_num)
        .context("Failed to convert device to host")?;

    // let transformed_original_source_pcd =
    //     convert_array2_to_pcd(&transformed_original_source_pts, 255, 0, 0); // Red color
    let transformed_original_source_pcd =
        convert_array2_to_pcd(&overlaped_source_pts, 255, 0, 0); // Red color
    // <!--- DEBUG --->

    let (d_final_transformed_source_pts, d_final_transformed_source_covs) = gpu_transform
        .apply_transform(&d_v_source_pts, &d_source_covs, v_source_pts_num, &delta_t)
        .context("Failed to apply transform")?;

    // <!--- DEBUG --->
    let final_transformed_source_pts =
        convert_dtoh(&d_final_transformed_source_pts, v_source_pts_num)
            .context("Failed to convert device to host")?;

    let final_transformed_source_pcd =
        convert_array2_to_pcd(&final_transformed_source_pts, 0, 255, 0); // Green color
    let target_pcd = convert_array2_to_pcd(&target_pts, 0, 0, 255); // Blue color
    
    let debug_source_pcd = convert_array2_to_pcd(&debug_source_pts, 255, 0, 255); // Blue color
    let mut combined_pcd = debug_source_pcd.clone();
    combined_pcd.extend(transformed_original_source_pcd);
    combined_pcd.extend(final_transformed_source_pcd);
    combined_pcd.extend(target_pcd);

    let output_path = format!("data/output/test/test-gicp.pcd");
    save_pcd_xyzrgb(
        &combined_pcd, &output_path)?;

    // <!--- DEBUG --->

    // let mut gpu_voxel =
    //     OclVoxelContext::new(ocl_runtime.clone()).expect("Failed to create OclVoxelContext");
    // let mut gpu_covs =
    //     OclCovContext::new(ocl_runtime.clone()).expect("Failed to create OclCovContext");
    // let mut gpu_search =
    //     OclSearchContext::new(ocl_runtime.clone()).expect("Failed to create OclSearchContext");
    // let mut gpu_transform = OclTransformContext::new(ocl_runtime.clone())
    //     .expect("Failed to create OclTransformContext");
    // let mut gpu_gicp = OclGicpContext::new(ocl_runtime.clone())
    //     .expect("Failed to create OclGicpContext");
    // println!("=== Completed GPU context ===");

    // println!("\n=== Warning up ({} iterations) ===", WARMUP_ITERATIONS);
    // for i in 0..WARMUP_ITERATIONS {
    //     let (d_v_points, valid) = gpu_voxel
    //         .voxel_downsample(&init_points, init_points.nrows(), VOXEL_SIZE)
    //         .expect("Voxel downsample failed");
    //     println!("Warmup {}: {} output points", i + 1, valid);

    //     let _ = gpu_covs
    //         .compute_covariances(&d_v_points, valid)
    //         .expect("Compute covariances failed");
    // }

    // println!(
    //     "\n=== Benchmarking ({} iterations) ===",
    //     BENCHMARK_ITERATIONS
    // );
    // let mut times = Vec::with_capacity(BENCHMARK_ITERATIONS);
    // let mut valid_count = 0;

    // for i in 0..BENCHMARK_ITERATIONS {
    //     let init_start = Instant::now();
    //     // Voxelization
    //     let (d_v_points, v_source_count) = gpu_voxel
    //         .voxel_downsample(&init_points, init_points.nrows(), VOXEL_SIZE)
    //         .expect("Voxel downsample failed");
    //     let elapsed_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    //     // println!("Voxelization: {} output points", v_source_count);
    //     println!("Iteration {}: Voxelization: {:.3} ms", i + 1, elapsed_ms);

    //     // Covariance computation
    //     let start = Instant::now();
    //     let d_covs = gpu_covs
    //         .compute_covariances(&d_v_points, v_source_count)
    //         .expect("Compute covariances failed");
    //     let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //     println!(
    //         "Iteration {}: Covariance computation: {:.3} ms",
    //         i + 1,
    //         elapsed_ms
    //     );

    //     // Transform points
    //     let start = Instant::now();
    //     // let identity_transform = Array2::<f32>::eye(4);
    //     // 90度回転（Z軸周り）+ X方向に2m移動
    //     let mut transform = Array2::<f32>::eye(4);
    //     let angle = std::f32::consts::FRAC_PI_2; // 90度
    //     transform[[0, 0]] = angle.cos();  // cos(90°) = 0
    //     transform[[0, 1]] = -angle.sin(); // -sin(90°) = -1
    //     transform[[1, 0]] = angle.sin();  // sin(90°) = 1
    //     transform[[1, 1]] = angle.cos();  // cos(90°) = 0
    //     transform[[0, 3]] = 2.0;          // X方向に2m移動

    //     let (d_transformed_points, d_transformed_covs) = gpu_transform
    //         .apply_transform(&d_v_points, &d_covs, v_source_count, &transform)
    //         .expect("Apply transform failed");
    //     let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //     println!(
    //         "Iteration {}: Transform points: {:.3} ms",
    //         i + 1,
    //         elapsed_ms
    //     );

    //     // Nearest neighbor search
    //     let start = Instant::now();
    //     let (d_indices, d_dists_sq, indices, dists_sq) = gpu_search
    //         .compute_find_nearest_neighbor(&d_transformed_points, v_source_count, VOXEL_SIZE, &gpu_voxel)
    //         .expect("Nearest neighbor search failed");
    //     let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //     println!(
    //         "Iteration {}: Nearest neighbor search: {:.3} ms",
    //         i + 1,
    //         elapsed_ms
    //     );

    //     // GICP
    //     let start = Instant::now();
    //     let (h_matrix, b_vector) = gpu_gicp.compute_gicp(
    //         &d_transformed_points,
    //         &d_transformed_covs,
    //         &d_v_points,
    //         &d_covs,
    //         &d_indices,
    //         &d_dists_sq,
    //         VOXEL_SIZE * VOXEL_SIZE
    //     ).context("Failed to calculate GICP")?;
    //     let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //     println!("Iteration {}: GICP computation: {:.3} ms", i + 1, elapsed_ms);

    //     let elapsed_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    //     times.push(elapsed_ms);
    //     valid_count = v_source_count;
    //     println!("Iteration {}: {:.3} ms", i + 1, elapsed_ms);

    //     // <!--- DEBUG --->
    //     let max_dist2: f32 = 2.0;

    //     let delta_t = solve_linear_system_6x6(h_matrix, b_vector)?;

    //     // Check convergence (RMSE)
    //     let mut sum = 0.0f32;
    //     let mut cnt = 0usize;
    //     for j in 0..v_source_count {
    //         let idx = indices[j];
    //         if idx < 0 { continue; }
    //         if dists_sq[j] > max_dist2 { continue; }
    //         sum += dists_sq[j];
    //         cnt += 1;
    //     }
    //     let rmse = (sum / cnt as f32).sqrt();
    //     println!("Iteration {}: RMSE = {}", i, rmse);

    //     // The matrix is calculated the adjusted transform for the next iteration
    //     let adjusted_transform = mat4_mul(&delta_t, &transform);
    //     // println!("Iteration {}: Adjusted Transform:\n{:?}", i, adjusted_transform);

    //     // Rotated source pts by initial transform
    //     let rotated_pts = convert_dtoh(
    //         &d_transformed_points,
    //         v_source_count
    //     ).context("Failed to convert device to host")?;

    //     let (d_adjusted_pts, _ ) = gpu_transform
    //         .apply_transform(&d_transformed_points, &d_transformed_covs, v_source_count, &delta_t)
    //         .expect("Apply adjusted transform failed");

    //     let adjusted_pts = convert_dtoh(
    //         &d_adjusted_pts,
    //         v_source_count
    //     ).context("Failed to convert device to host")?;

    //     // Adjusted source pts by gicp
    //     let adjusted_pcd = convert_array2_to_pcd(&adjusted_pts, 255, 0, 0);  // Red color

    //     let rotated_pcd = convert_array2_to_pcd(&rotated_pts, 0, 0, 255);  // Blue color

    //     let mut combined_pcd = init_pcd.clone();
    //     combined_pcd.extend(adjusted_pcd);
    //     combined_pcd.extend(rotated_pcd);

    //     let output_path = format!("data/output/iteration_{:02}.pcd", i + 1);
    //     save_pcd_xyzrgb(
    //         &combined_pcd, &output_path)?;

    //     // <!--- DEBUG --->

    // }

    // // 統計情報
    // let sum: f64 = times.iter().sum();
    // let mean = sum / times.len() as f64;

    // let mut sorted = times.clone();
    // sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // let median = sorted[sorted.len() / 2];
    // let min = sorted[0];
    // let max = sorted[sorted.len() - 1];

    // let variance: f64 = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    // let stddev = variance.sqrt();

    // println!("\n=== Benchmark Results ===");
    // println!("Input points:  {}", init_points.nrows());
    // println!("Output points: {}", valid_count);
    // println!("Voxel size:    {}", VOXEL_SIZE);
    // println!("\nTiming statistics (ms):");
    // println!("  Mean:   {:.3}", mean);
    // println!("  Median: {:.3}", median);
    // println!("  Min:    {:.3}", min);
    // println!("  Max:    {:.3}", max);
    // println!("  Stddev: {:.3}", stddev);

    Ok(())
}

fn calculate_centroid(pts: &Array2<f32>) -> Array1<f32> {
    let n = pts.nrows() as f32;
    let mut centroid = Array1::<f32>::zeros(3);

    for i in 0..n as usize {
        centroid[0] += pts[[i, 0]];
        centroid[1] += pts[[i, 1]];
        centroid[2] += pts[[i, 2]];
    }
    centroid /= n;

    centroid
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

fn pcd_to_array2(pcd_points: &[PointXYZRGB]) -> Array2<f32> {
    // fn pcd_to_array2(pcd_points: &[PointXYZT]) -> Array2<f32> {
    // fn pcd_to_array2(pcd_points: &[PointXYZ]) -> Array2<f32> {
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
