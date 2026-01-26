// OpenCL 1.2

__kernel void transform_points_and_covs(
    __global const float* restrict pts,
    __global const float* restrict covs,
    int num_points,
    float r00, float r01, float r02, float t0,
    float r10, float r11, float r12, float t1,
    float r20, float r21, float r22, float t2,
    __global float* restrict out_pts,
    __global float* restrict out_covs
) {
    int idx = get_global_id(0);

    if (idx >= num_points) return;

    float px = pts[idx * 3 + 0];
    float py = pts[idx * 3 + 1];
    float pz = pts[idx * 3 + 2];

    // Rotate each point
    out_pts[idx * 3 + 0] = r00 * px + r01 * py + r02 * pz + t0;
    out_pts[idx * 3 + 1] = r10 * px + r11 * py + r12 * pz + t1;
    out_pts[idx * 3 + 2] = r20 * px + r21 * py + r22 * pz + t2;

    // Rotate each covariance
    float c00 = covs[idx * 9 + 0];
    float c01 = covs[idx * 9 + 1];
    float c02 = covs[idx * 9 + 2];
    float c10 = covs[idx * 9 + 3];
    float c11 = covs[idx * 9 + 4];
    float c12 = covs[idx * 9 + 5];
    float c20 = covs[idx * 9 + 6];
    float c21 = covs[idx * 9 + 7];
    float c22 = covs[idx * 9 + 8];

    float tmp00 = r00 * c00 + r01 * c10 + r02 * c20;
    float tmp01 = r00 * c01 + r01 * c11 + r02 * c21;
    float tmp02 = r00 * c02 + r01 * c12 + r02 * c22;

    float tmp10 = r10 * c00 + r11 * c10 + r12 * c20;
    float tmp11 = r10 * c01 + r11 * c11 + r12 * c21;
    float tmp12 = r10 * c02 + r11 * c12 + r12 * c22;

    float tmp20 = r20 * c00 + r21 * c10 + r22 * c20;
    float tmp21 = r20 * c01 + r21 * c11 + r22 * c21;
    float tmp22 = r20 * c02 + r21 * c12 + r22 * c22;

    float tc00 = tmp00 * r00 + tmp01 * r01 + tmp02 * r02;
    float tc01 = tmp00 * r10 + tmp01 * r11 + tmp02 * r12;
    float tc02 = tmp00 * r20 + tmp01 * r21 + tmp02 * r22;

    float tc10 = tmp10 * r00 + tmp11 * r01 + tmp12 * r02;
    float tc11 = tmp10 * r10 + tmp11 * r11 + tmp12 * r12;
    float tc12 = tmp10 * r20 + tmp11 * r21 + tmp12 * r22;

    float tc20 = tmp20 * r00 + tmp21 * r01 + tmp22 * r02;
    float tc21 = tmp20 * r10 + tmp21 * r11 + tmp22 * r12;
    float tc22 = tmp20 * r20 + tmp21 * r21 + tmp22 * r22;

    out_covs[idx * 9 + 0] = tc00;
    out_covs[idx * 9 + 1] = tc01;
    out_covs[idx * 9 + 2] = tc02;
    out_covs[idx * 9 + 3] = tc10;
    out_covs[idx * 9 + 4] = tc11;
    out_covs[idx * 9 + 5] = tc12;
    out_covs[idx * 9 + 6] = tc20;
    out_covs[idx * 9 + 7] = tc21;
    out_covs[idx * 9 + 8] = tc22;
}