use std::{
    os::unix::process,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::Solve;
use ocl::{Device, Platform, core::DeviceInfo};
use registration_opencl::{
    convert_dtoh::convert_dtoh,
    gpu_cov::OclCovContext,
    gpu_gicp::OclGicpContext,
    gpu_icp::{self, OclIcpContext},
    gpu_normals::OclNormalsContext,
    gpu_search::OclSearchContext,
    gpu_search_neighbors::OclSearchNeighborsContext,
    gpu_transform::OclTransformContext,
    gpu_voxel::OclVoxelContext,
    icp_iterations::{self, ICPProcessArgs02, icp_registration},
    ocl_context::{OclContexts, OclRuntime},
    operate_pcd_file::{
        PointXYZ, PointXYZRGB, PointXYZT, load_pcd_xyz, load_pcd_xyzrgb, load_pcd_xyzt,
        save_pcd_xyz, save_pcd_xyzrgb,
    },
    rotation_matrix::{
        create_fb_flip_matrix, create_lr_flip_matrix, create_rot_90_x_matrix,
        create_rot_90_y_matrix, create_rot_90_z_matrix, create_rot_180_x_matrix,
        create_rot_180_y_matrix, create_rot_180_z_matrix, create_rot_minus_90_x_matrix,
        create_rot_minus_90_y_matrix, create_rot_minus_90_z_matrix, create_ud_flip_matrix,
    },
};

const VOXEL_SIZE: f32 = 0.25;
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;
const GICP_MAX_ITERATIONS: usize = 60;
const ICP_MAX_ITERATIONS: usize = 60;
const TOLERANCE: f64 = 0.015;

#[derive(Clone)]
struct ProcessTimes {
    registration_pcd_center_time: Duration,
    voxel_time: Duration,
    cov_time: Duration,
    normal_time: Duration,
    transform_time: Duration,
    search_neighbor_time: Duration,
    search_neighbors_time: Duration,
    gicp_time: Duration,
    icp_time: Duration,
}

#[derive(Debug, Clone)]
struct ICPMatrixs {
    init_icp_matrix: Array2<f32>,
    init_icp_matrix_rmse: f32,
    address_back_to_front: Array2<f32>,
    address_back_to_front_rmse: f32,
    address_upside_down: Array2<f32>,
    address_upside_down_rmse: f32,
    address_back_to_front_and_upside_down: Array2<f32>,
    address_back_to_front_and_upside_down_rmse: f32,
}

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
    let mut gpu_normals =
        OclNormalsContext::new(ocl_runtime.clone()).expect("Failed to create OclNormalsContext");
    let mut gpu_transform = OclTransformContext::new(ocl_runtime.clone())
        .expect("Failed to create OclTransformContext");
    let mut gpu_search =
        OclSearchContext::new(ocl_runtime.clone()).expect("Failed to create OclSearchContext");
    let mut gpu_search_neighbors = OclSearchNeighborsContext::new(ocl_runtime.clone())
        .expect("Failed to create OclSearchNeighborsContext");
    let mut gpu_gicp =
        OclGicpContext::new(ocl_runtime.clone()).expect("Failed to create OclGicpContext");
    let mut gpu_icp =
        OclIcpContext::new(ocl_runtime.clone()).expect("Failed to create OclIcpContext");

    let mut ocl_contexts = OclContexts {
        gpu_voxel: &mut gpu_voxel,
        gpu_covs: &mut gpu_covs,
        gpu_normals: &mut gpu_normals,
        gpu_transform: &mut gpu_transform,
        gpu_search: &mut gpu_search,
        gpu_search_neighbors: &mut gpu_search_neighbors,
        gpu_gicp: &mut gpu_gicp,
        gpu_icp: &mut gpu_icp,
    };

    let mut icp_matrixs = ICPMatrixs {
        init_icp_matrix: Array2::<f32>::eye(4),
        init_icp_matrix_rmse: 0.0,
        address_back_to_front: Array2::<f32>::eye(4),
        address_back_to_front_rmse: 0.0,
        address_upside_down: Array2::<f32>::eye(4),
        address_upside_down_rmse: 0.0,
        address_back_to_front_and_upside_down: Array2::<f32>::eye(4),
        address_back_to_front_and_upside_down_rmse: 0.0,
    };

    let mut process_times = ProcessTimes {
        registration_pcd_center_time: Duration::new(0, 0),
        voxel_time: Duration::new(0, 0),
        cov_time: Duration::new(0, 0),
        normal_time: Duration::new(0, 0),
        transform_time: Duration::new(0, 0),
        search_neighbor_time: Duration::new(0, 0),
        search_neighbors_time: Duration::new(0, 0),
        gicp_time: Duration::new(0, 0),
        icp_time: Duration::new(0, 0),
    };

    let source_pcd_path = "data/input/H927/vggt-data_output_voxel_025_xyz_only.pcd";
    let source_pcd = load_pcd_xyz(source_pcd_path).expect("Failed to load initial PCD file");
    let source_pts = pcd_to_array2(&source_pcd);

    let target_pcd_path = "data/input/H927/lab-room_voxel_025_xyz_only.pcd";
    let target_pcd = load_pcd_xyz(target_pcd_path).expect("Failed to load initial PCD file");
    let target_pts = pcd_to_array2(&target_pcd);

    println!("\n=== Loaded PCD files ===");
    println!("Source points: {}", source_pts.nrows());
    println!("Loaded source PCD from: {}", source_pcd_path);
    println!("Target points: {}", target_pts.nrows());
    println!("Loaded target PCD from: {}", target_pcd_path);

    let start = std::time::Instant::now();
    let start_center = std::time::Instant::now();

    // Center the source points to target points
    let (_, overlaped_source_pts) = registration_pcd_center(&source_pts, &target_pts);
    process_times.registration_pcd_center_time = start_center.elapsed();

    let start_voxel = std::time::Instant::now();
    // Voxel downsample
    let (d_v_source_pts, v_source_pts_num) = ocl_contexts
        .gpu_voxel
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

    let (d_v_target_pts, v_target_pts_num) = ocl_contexts
        .gpu_voxel
        .voxel_downsample(&target_pts, target_pts.nrows(), VOXEL_SIZE)
        .context("Failed to compute voxel")?;
    println!(
        "Voxel downsampled target points: {} -> {}",
        target_pts.nrows(),
        v_target_pts_num
    );
    process_times.voxel_time = start_voxel.elapsed();

    let start_covs = std::time::Instant::now();
    // Compute covariances
    let d_source_covs = ocl_contexts
        .gpu_covs
        .compute_covariances(&d_v_source_pts, v_source_pts_num)
        .context("Failed to compute covariances")?;
    let d_target_covs = ocl_contexts
        .gpu_covs
        .compute_covariances(&d_v_target_pts, v_target_pts_num)
        .context("Failed to compute covariances")?;
    process_times.cov_time = start_covs.elapsed();

    // let transform_matrix = Array2::<f32>::eye(4);

    // let mut icp_process_args = ICPProcessArgs {
    //     d_v_source_pts: d_v_source_pts.clone(),
    //     d_source_covs: d_source_covs.clone(),
    //     v_source_pts_num,
    //     d_v_target_pts: d_v_target_pts.clone(),
    //     d_target_covs: d_target_covs.clone(),
    //     v_target_pts_num,
    //     transform_matrix: transform_matrix.clone(),
    // };

    // // Initial iterations of ICP
    // let (icp_transform_matrix, icp_rmse) = icp_iteration(
    //     &mut ocl_contexts,
    //     &mut icp_process_args.clone(),
    //     &mut process_times,
    // )?;

    // icp_matrixs.init_icp_matrix = icp_transform_matrix.clone();
    // icp_matrixs.init_icp_matrix_rmse = icp_rmse;

    let flip_rot_transforms = vec![
        ("Initial", Array2::<f32>::eye(4)),
        ("LR_Flip_Reverse-Y", create_lr_flip_matrix()),
        ("UD_Flip_Reverse-Z", create_ud_flip_matrix()),
        ("Front-Back_Flip_Reverse-X", create_fb_flip_matrix()),
        ("Rot_180_Z", create_rot_180_z_matrix()),
        ("Rot_180_Y", create_rot_180_y_matrix()),
        ("Rot_180_X", create_rot_180_x_matrix()),
        (
            "LR+UD_Flip",
            create_lr_flip_matrix().dot(&create_ud_flip_matrix()),
        ),
        (
            "LR+FB_Flip",
            create_lr_flip_matrix().dot(&create_fb_flip_matrix()),
        ),
        (
            "UD+FB_Flip",
            create_ud_flip_matrix().dot(&create_fb_flip_matrix()),
        ),
        ("Rot90_X", create_rot_90_x_matrix()),
        ("Rot-90_X", create_rot_minus_90_x_matrix()),
        ("Rot90_Y", create_rot_90_y_matrix()),
        ("Rot-90_Y", create_rot_minus_90_y_matrix()),
        ("Rot90_Z", create_rot_90_z_matrix()),
        ("Rot-90_Z", create_rot_minus_90_z_matrix()),
    ];

    let mut candidates: Vec<(
        &str,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        f32,
    )> = Vec::new();

    for (label, flip_rot_matrix) in flip_rot_transforms {
        // === Downsample the source and target points ===
        let (d_v_source_pts, v_source_pts_num) = ocl_contexts
            .gpu_voxel
            .voxel_downsample(&source_pts, source_pts.nrows(), VOXEL_SIZE)
            .context("Failed to compute voxel")?;

        let (d_v_target_pts, v_target_pts_num) = ocl_contexts
            .gpu_voxel
            .voxel_downsample(&target_pts, target_pts.nrows(), VOXEL_SIZE)
            .context("Failed to compute voxel")?;

        let v_source_pts = convert_dtoh(&d_v_source_pts, v_source_pts_num)
            .context("Failed to copy device memory to host memory")?;

        let v_target_pts = convert_dtoh(&d_v_target_pts, v_target_pts_num)
            .context("Failed to copy device memory to host memory")?;

        // === Centralize the source points to the target points ===
        // Apply flip/rotation to the source points
        // Extract 3x3 rotation part from 4x4 matrix
        let rotation_3x3 = flip_rot_matrix.slice(s![0..3, 0..3]);
        let fliped_rotated_v_source_pts = v_source_pts.dot(&rotation_3x3.t());

        let (centralized_matrix, centralized_v_source_pts) =
            registration_pcd_center(&fliped_rotated_v_source_pts, &v_target_pts);

        let mut icp_process_args = ICPProcessArgs02 {
            d_v_source_pts,
            d_v_source_covs: d_source_covs.clone(),
            v_source_pts_num,
            d_v_target_pts,
            v_target_pts_num,
            flip_rot_matrix: flip_rot_matrix.clone(),
            transform_matrix: centralized_matrix.clone(),
        };

        // === compute ICP ===
        let (icp_transform_matrix, icp_rmse) = icp_registration(
            &mut ocl_contexts,
            &mut icp_process_args,
            VOXEL_SIZE,
            ICP_MAX_ITERATIONS,
            TOLERANCE,
        )?;

        candidates.push((
            label,
            v_source_pts,
            flip_rot_matrix,
            centralized_matrix,
            icp_transform_matrix,
            icp_rmse,
        ));
    }

    // === Select the best result based on RMSE ===
    let mut best_rmse = f32::MAX;
    let mut best_label = String::from("None");
    let mut best_final_pts = Array2::<f32>::zeros((0, 3));
    let mut best_tf = Array2::<f32>::eye(4);

    for (
        label,
        v_source_pts,
        flip_rot_matrix,
        centralized_matrix,
        icp_transform_matrix,
        icp_rmse,
    ) in candidates.clone()
    {
        if icp_rmse < best_rmse {
            best_rmse = icp_rmse;
            best_label = label.to_string();
            best_final_pts = v_source_pts;
            // icp_transform_matrix already contains centralized_matrix
            best_tf = icp_transform_matrix.clone();
        }
    }

    println!("----------------------------------------");
    println!("Best: {} (RMSE: {:.6})", best_label, best_rmse);
    println!("Best Transform:\n{:?}", best_tf);

    let best_transformed_pts =
        best_final_pts.dot(&best_tf.slice(s![0..3, 0..3]).t()) + &best_tf.slice(s![0..3, 3]);

    let source_pts_pcd = convert_array2_to_pcd(&source_pts, 255, 0, 0); // Red color
    let best_pts_pcd = convert_array2_to_pcd(&best_transformed_pts, 0, 255, 0); // Green color
    let target_pts_pcd = convert_array2_to_pcd(&target_pts, 0, 0, 255); // Blue color

    let mut combined_pcd = source_pts_pcd.clone();
    combined_pcd.extend(best_pts_pcd);
    combined_pcd.extend(target_pts_pcd);

    let save_file_path = format!(
        "data/output/transformed_data/best_transformed_icp_voxel_{}_{}.pcd",
        VOXEL_SIZE, best_label
    );
    save_pcd_xyzrgb(&combined_pcd, &save_file_path)?;

    // === Debug: Save all candidates' results ===
    println!("\n=== Debug: Saving all candidates' transformed PCDs ===");
    for (
        label,
        v_source_pts,
        flip_rot_matrix,
        centralized_matrix,
        icp_transform_matrix,
        icp_rmse,
    ) in candidates.clone()
    {
        let transformed_pts = v_source_pts
            .dot(&flip_rot_matrix.slice(s![0..3, 0..3]).t())
            // .dot(&centralized_matrix.slice(s![0..3, 0..3]).t())
            .dot(&icp_transform_matrix.slice(s![0..3, 0..3]).t());

        let transformed_pts = v_source_pts.dot(&icp_transform_matrix.slice(s![0..3, 0..3]).t())
            + &icp_transform_matrix.slice(s![0..3, 3]);

        let source_pcd = convert_array2_to_pcd(&source_pts, 255, 0, 0); // Red color
        let transformed_pcd = convert_array2_to_pcd(&transformed_pts, 0, 255, 0); // Green color
        let target_pcd = convert_array2_to_pcd(&target_pts, 0, 0, 255); // Blue color

        let mut combined_pcd = source_pcd.clone();
        combined_pcd.extend(transformed_pcd);
        combined_pcd.extend(target_pcd);

        let save_file_path = format!(
            "data/output/transformed_data/Debug/transformed_icp_voxel_{}_{}.pcd",
            VOXEL_SIZE, label
        );
        save_pcd_xyzrgb(&combined_pcd, &save_file_path)?;
        println!("Saved transformed PCD for {}: {}", label, save_file_path);
    }

    // let mut icp_process_args_02 = ICPProcessArgs02 {
    //     d_v_centralized_pts
    // }

    // Back to front
    // let address_back_to_front = ndarray::array![
    //     [-1.0f32, 0.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, -1.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, 1.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, 0.0f32, 1.0f32]
    // ];
    // icp_process_args.transform_matrix = address_back_to_front;

    // let (icp_transform_matrix, icp_rmse) = icp_iteration(
    //     &mut ocl_contexts,
    //     &mut icp_process_args.clone(),
    //     &mut process_times,
    // )?;

    // icp_matrixs.address_back_to_front = icp_transform_matrix.clone();
    // icp_matrixs.address_back_to_front_rmse = icp_rmse;

    // // Upside down
    // let address_upside_down = ndarray::array![
    //     [1.0f32, 0.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, -1.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, -1.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, 0.0f32, 1.0f32],
    // ];
    // icp_process_args.transform_matrix = address_upside_down;

    // let (icp_transform_matrix, icp_rmse) = icp_iteration(
    //     &mut ocl_contexts,
    //     &mut icp_process_args.clone(),
    //     &mut process_times,
    // )?;
    // icp_matrixs.address_upside_down = icp_transform_matrix.clone();
    // icp_matrixs.address_upside_down_rmse = icp_rmse;

    // // Back to front & upside down
    // let address_back_to_front_and_upside_down = ndarray::array![
    //     [-1.0f32, 0.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, 1.0f32, 0.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, -1.0f32, 0.0f32],
    //     [0.0f32, 0.0f32, 0.0f32, 1.0f32],
    // ];

    // icp_process_args.transform_matrix = address_back_to_front_and_upside_down;
    // let (icp_transform_matrix, icp_rmse) = icp_iteration(
    //     &mut ocl_contexts,
    //     &mut icp_process_args.clone(),
    //     &mut process_times,
    // )?;
    // icp_matrixs.address_back_to_front_and_upside_down = icp_transform_matrix.clone();
    // icp_matrixs.address_back_to_front_and_upside_down_rmse = icp_rmse;

    // let elapsed = start.elapsed();
    // println!("\n=== Process times ===");
    // println!(
    //     "Registration PCD Center time: {:.2?}",
    //     process_times.registration_pcd_center_time
    // );
    // println!("Voxel time: {:.2?}", process_times.voxel_time);
    // println!("Covariance time: {:.2?}", process_times.cov_time);
    // println!("Normal time: {:.2?}", process_times.normal_time);
    // println!("Transform time: {:.2?}", process_times.transform_time);
    // println!(
    //     "Transform time (Per 1 iteration): {:.2?}",
    //     process_times.transform_time / GICP_MAX_ITERATIONS as u32
    // );
    // println!(
    //     "Search Neighbor time: {:.2?}",
    //     process_times.search_neighbor_time
    // );
    // println!(
    //     "Search Neighbor time (Per 1 iteration): {:.2?}",
    //     process_times.search_neighbor_time / GICP_MAX_ITERATIONS as u32
    // );
    // println!(
    //     "Search Neighbors time: {:.2?}",
    //     process_times.search_neighbors_time
    // );
    // println!(
    //     "Search Neighbors time (Per 1 iteration): {:.2?}",
    //     process_times.search_neighbors_time / GICP_MAX_ITERATIONS as u32
    // );
    // println!("GICP time: {:.2?}", process_times.gicp_time);
    // println!(
    //     "GICP time (Per 1 iteration): {:.2?}",
    //     process_times.gicp_time / GICP_MAX_ITERATIONS as u32
    // );
    // println!("ICP time: {:.2?}", process_times.icp_time);
    // println!(
    //     "ICP time (Per 1 iteration): {:.2?}",
    //     process_times.icp_time / GICP_MAX_ITERATIONS as u32
    // );
    // println!("Total time: {:.2?}", elapsed);

    // println!("\n=== ICP transformed results ===");
    // save_processed_pcds(
    //     &mut ocl_contexts,
    //     &overlaped_source_pts,
    //     &target_pts,
    //     &icp_process_args,
    //     &icp_matrixs.init_icp_matrix,
    //     "init_icp",
    // )?;
    // println!("Initial ICP RMSE: {}", icp_matrixs.init_icp_matrix_rmse);

    // save_processed_pcds(
    //     &mut ocl_contexts,
    //     &overlaped_source_pts,
    //     &target_pts,
    //     &icp_process_args,
    //     &icp_matrixs.address_back_to_front,
    //     "address_back_to_front",
    // )?;
    // println!(
    //     "Address Back to Front RMSE: {}",
    //     icp_matrixs.address_back_to_front_rmse
    // );

    // save_processed_pcds(
    //     &mut ocl_contexts,
    //     &overlaped_source_pts,
    //     &target_pts,
    //     &icp_process_args,
    //     &icp_matrixs.address_upside_down,
    //     "address_upside_down",
    // )?;
    // println!(
    //     "Address Upside Down RMSE: {}",
    //     icp_matrixs.address_upside_down_rmse
    // );

    // save_processed_pcds(
    //     &mut ocl_contexts,
    //     &overlaped_source_pts,
    //     &target_pts,
    //     &icp_process_args,
    //     &icp_matrixs.address_back_to_front_and_upside_down,
    //     "address_back_to_front_and_upside_down",
    // )?;
    // println!(
    //     "Address Back to Front and Upside Down RMSE: {}",
    //     icp_matrixs.address_back_to_front_and_upside_down_rmse
    // );

    Ok(())
}

fn save_processed_pcds(
    ocl_contexts: &mut OclContexts,
    overlapped_source_pts: &Array2<f32>,
    target_pts: &Array2<f32>,
    icp_process_args: &ICPProcessArgs,
    transform_matrix: &Array2<f32>,
    rotate_type: &str,
) -> Result<()> {
    let transformed_original_source_pcd = convert_array2_to_pcd(&overlapped_source_pts, 255, 0, 0); // Red color
    // <!--- DEBUG --->

    let (d_final_transformed_source_pts, d_final_transformed_source_covs) = ocl_contexts
        .gpu_transform
        .apply_transform(
            &icp_process_args.d_v_source_pts,
            &icp_process_args.d_source_covs,
            icp_process_args.v_source_pts_num,
            transform_matrix,
        )
        .context("Failed to apply transform")?;

    // <!--- DEBUG --->
    let final_transformed_source_pts = convert_dtoh(
        &d_final_transformed_source_pts,
        icp_process_args.v_source_pts_num,
    )
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

    let output_path = format!(
        "data/output/transformed_data/transformed_icp-iter-{}_voxel-{}_{}.pcd",
        GICP_MAX_ITERATIONS, VOXEL_SIZE, rotate_type
    );
    save_pcd_xyzrgb(&combined_pcd, &output_path)?;
    println!("Saved combined PCD to: {}", output_path);

    Ok(())
}

#[derive(Clone)]
struct ICPProcessArgs {
    d_v_source_pts: ocl::Buffer<f32>,
    d_source_covs: ocl::Buffer<f32>,
    v_source_pts_num: usize,
    d_v_target_pts: ocl::Buffer<f32>,
    d_target_covs: ocl::Buffer<f32>,
    v_target_pts_num: usize,
    transform_matrix: Array2<f32>,
}

fn icp_iteration(
    ocl_contexts: &mut OclContexts,
    icp_args: &mut ICPProcessArgs,
    process_times: &mut ProcessTimes,
) -> Result<(Array2<f32>, f32)> {
    let mut transform_matrix = icp_args.transform_matrix.clone();
    let mut rmse: f32 = 0.0;

    for i in 0..GICP_MAX_ITERATIONS {
        println!("\n=== GICP Iteration {} ===", i + 1);

        let start = std::time::Instant::now();
        // Transform each points
        let (d_transformed_source_pts, d_transformed_source_covs) = ocl_contexts
            .gpu_transform
            .apply_transform(
                &icp_args.d_v_source_pts,
                &icp_args.d_source_covs,
                icp_args.v_source_pts_num,
                &transform_matrix,
            )
            .context("Failed to apply transform")?;
        process_times.transform_time += start.elapsed();

        let start = std::time::Instant::now();
        // Compute to find nearest neighbor pts
        let (d_indices, d_dists_sq, indices, dists_sq) = ocl_contexts
            .gpu_search
            .compute_find_nearest_neighbor(
                &d_transformed_source_pts,
                icp_args.v_source_pts_num,
                &icp_args.d_v_target_pts,
                icp_args.v_target_pts_num,
            )
            .context("Failed to compute nearest neighbor")?;
        process_times.search_neighbor_time += start.elapsed();

        let start = std::time::Instant::now();
        // Search neigghbors to compute normals
        let (d_target_indices, d_target_dists_sq, _, _) = ocl_contexts
            .gpu_search_neighbors
            .compute_find_neighbors(&icp_args.d_v_target_pts, icp_args.v_target_pts_num)
            .context("Failed to compute self k-nearest neighbors")?;
        process_times.search_neighbors_time += start.elapsed();

        let start = std::time::Instant::now();
        // Compute normals
        let d_target_normals = ocl_contexts
            .gpu_normals
            .compute_normals(
                &icp_args.d_v_target_pts,
                icp_args.v_target_pts_num,
                &d_target_indices,
                0.0,
                0.0,
                0.0,
            )
            .context("Failed to compute normals")?;
        process_times.normal_time += start.elapsed();

        // Debug
        let max_dist2: f32 = VOXEL_SIZE * VOXEL_SIZE;
        // let max_dist2: f32 = VOXEL_SIZE * 1.5;
        let valid_pairs = indices
            .iter()
            .zip(dists_sq.iter())
            .filter(|(idx, dist)| **idx >= 0 && **dist <= max_dist2)
            .count();
        // println!(
        //     "Iteration {}: Found {} nearest neighbor correspondences",
        //     i + 1,
        //     valid_pairs
        // );

        // let start = std::time::Instant::now();
        // Compute GICP
        // let (h_matrix, b_vector) = ocl_contexts.gpu_gicp
        //     .compute_gicp(
        //         &d_transformed_source_pts,
        //         &d_transformed_source_covs,
        //         icp_argsv_source_pts_num,
        //         &d_v_target_pts,
        //         &d_target_covs,
        //         v_target_pts_num,
        //         &d_indices,
        //         &d_dists_sq,
        //         max_dist2,
        //     )
        //     .context("Failed to calculate GICP")?;
        // process_times.gicp_time += start.elapsed();

        let start = std::time::Instant::now();
        let (h_matrix, b_vector) = ocl_contexts
            .gpu_icp
            .compute_icp(
                &d_transformed_source_pts,
                icp_args.v_source_pts_num,
                &icp_args.d_v_target_pts,
                &d_target_normals,
                icp_args.v_target_pts_num,
                &d_indices,
                &d_dists_sq,
                max_dist2,
            )
            .context("Failed to calculate ICP")?;
        process_times.icp_time += start.elapsed();

        let delta_t = solve_linear_system_6x6(h_matrix, b_vector)?;
        transform_matrix = mat4_mul(&delta_t, &transform_matrix);
        // println!("Computed delta transform:\n{:?}", delta_t);
        // println!("Updated transform matrix:\n{:?}", transform_matrix);

        // Check convergence (RMSE)
        let mut sum = 0.0f32;
        let mut cnt = 0usize;

        for j in 0..icp_args.v_source_pts_num {
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
        rmse = (sum / cnt as f32).sqrt();

        if i == (GICP_MAX_ITERATIONS - 1) {
            println!("Updated transform matrix:\n{:?}", transform_matrix);
            println!("Iteration {}: RMSE = {}", i + 1, rmse);
        }

        // break;
    }

    Ok((transform_matrix, rmse))
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
    target_points: &Array2<f32>,
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
