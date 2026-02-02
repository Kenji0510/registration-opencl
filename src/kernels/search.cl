// OpenCL 1.2

__kernel void find_nearest_neighbor(
    __global const float* restrict source_pts,
    int num_source,
    __global const float* restrict target_pts,
    int num_target,
    __global int* restrict out_indices,
    __global float* restrict out_dists_sq
) {
    int idx = get_global_id(0);

    if (idx >= num_source) return;

    float px = source_pts[idx * 3 + 0];
    float py = source_pts[idx * 3 + 1];
    float pz = source_pts[idx * 3 + 2];

    float best_dist_sq = 1.0e30f;
    int best_target_idx = -1;

    for (int j = 0; j < num_target; j++) {
        float tx = target_pts[j * 3 + 0];
        float ty = target_pts[j * 3 + 1];
        float tz = target_pts[j * 3 + 2];

        float dx = px - tx;
        float dy = py - ty;
        float dz = pz - tz;

        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_target_idx = j;
        }
    }

    out_dists_sq[idx] = best_dist_sq;
    out_indices[idx] = best_target_idx;
}