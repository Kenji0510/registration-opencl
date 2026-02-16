use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

use crate::ocl_context::OclContexts;

#[derive(Clone)]
pub struct ICPProcessArgs02 {
    pub d_v_source_pts: ocl::Buffer<f32>,
    pub d_v_source_covs: ocl::Buffer<f32>,
    pub v_source_pts_num: usize,
    // d_v_centralized_pts: ocl::Buffer<f32>,
    // v_centralized_pts_num: usize,
    pub d_v_target_pts: ocl::Buffer<f32>,
    pub v_target_pts_num: usize,
    // pub d_v_target_normals: ocl::Buffer<f32>,
    pub flip_rot_matrix: Array2<f32>,
    pub transform_matrix: Array2<f32>,
}

pub fn icp_registration(
    ocl_context: &mut OclContexts,
    icp_process_args: &mut ICPProcessArgs02,
    voxel_size: f32,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Array2<f32>, f32)> {
    // Initial transform matrix is centerlized matrix.
    let mut combined_transform = mat4_mul(
        &icp_process_args.transform_matrix,
        &icp_process_args.flip_rot_matrix,
    );
    let mut rmse: f32 = 0.0;

    // === Transform source points using flip_rot_matrix === Should fix
    let (mut d_src_pts_cur, mut d_src_covs_cur) = ocl_context.gpu_transform.apply_transform(
        &icp_process_args.d_v_source_pts,
        &icp_process_args.d_v_source_covs,
        icp_process_args.v_source_pts_num,
        &combined_transform,
    )?;

    for i in 0..max_iterations {
        // === Transform source points using current estimate === Should fix
        // let (d_v_transformed_source_pts, d_v_transformed_source_covs) = ocl_context
        //     .gpu_transform
        //     .apply_transform(
        //         &d_v_transformed_source_pts,
        //         &d_v_transformed_source_covs,
        //         icp_process_args.v_source_pts_num,
        //         &combined_transform,
        //     )
        //     .context("Failed to apply transform to source points")?;

        // === Compute normals for source points ===
        let (d_v_source_indices, d_v_source_dists_sq, v_source_indices, v_source_dists_sq) =
            ocl_context
                .gpu_search
                .compute_find_nearest_neighbor(
                    &d_src_pts_cur,
                    icp_process_args.v_source_pts_num,
                    &icp_process_args.d_v_target_pts,
                    icp_process_args.v_target_pts_num,
                )
                .context("Failed to compute nearest neighbors for source points")?;

        // === Compute normals for target points ===
        let (d_v_target_indices, d_v_target_dists_sq, v_target_indices, v_target_dists_sq) =
            ocl_context
                .gpu_search_neighbors
                .compute_find_neighbors(
                    &icp_process_args.d_v_target_pts,
                    icp_process_args.v_target_pts_num,
                )
                .context("Failed to compute self k-nearest neighbors")?;

        let d_v_target_normals = ocl_context
            .gpu_normals
            .compute_normals(
                &icp_process_args.d_v_target_pts,
                icp_process_args.v_target_pts_num,
                &d_v_target_indices,
                0.0,
                0.0,
                0.0,
            )
            .context("Failed to compute normals for target points")?;

        let max_dist2: f32 = voxel_size * 1.5;
        // let max_dist2: f32 = voxel_size * voxel_size;
        let valid_pairs = v_source_indices
            .iter()
            .zip(v_source_dists_sq.iter())
            .filter(|(idx, dist)| **idx >= 0 && **dist <= max_dist2)
            .count();

        let (h_matrix, b_vector) = ocl_context
            .gpu_icp
            .compute_icp(
                &d_src_pts_cur,
                icp_process_args.v_source_pts_num,
                &icp_process_args.d_v_target_pts,
                &d_v_target_normals,
                icp_process_args.v_target_pts_num,
                &d_v_source_indices,
                &d_v_source_dists_sq,
                max_dist2,
            )
            .context("Failed to compute ICP")?;

        let delta_t = solve_linear_system_6x6(h_matrix, b_vector)?;
        combined_transform = mat4_mul(&delta_t, &combined_transform);

        (d_src_pts_cur, d_src_covs_cur) = ocl_context.gpu_transform.apply_transform(
            &d_src_pts_cur,
            &d_src_covs_cur,
            icp_process_args.v_source_pts_num,
            &delta_t,
        )?;

        // Check convergence (RMSE)
        let mut sum = 0.0f32;
        let mut cnt = 0usize;

        for j in 0..icp_process_args.v_source_pts_num {
            let idx = v_source_indices[j];
            if idx < 0 {
                continue;
            }
            if v_source_dists_sq[j] > max_dist2 {
                continue;
            }
            sum += v_source_dists_sq[j];
            cnt += 1;
        }
        rmse = (sum / cnt as f32).sqrt();

        if i == (max_iterations - 1) {
            println!("Updated transform matrix:\n{:?}", combined_transform);
            println!("Iteration {}: RMSE = {}", i + 1, rmse);
        }
    }

    // let total = mat4_mul(&combined_transform, &icp_process_args.flip_rot_matrix);
    // Ok((total, rmse))

    Ok((combined_transform, rmse))
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
