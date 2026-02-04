// OpenCL 1.2

#ifndef K
#define K 15
#endif

__kernel void search_neighbors(
    __global const float* restrict pts,       // n * 3
    int num_points,
    __global int* restrict out_indices,       // n * K
    __global float* restrict out_dists_sq     // n * K
) {
    int gid = get_global_id(0);

    if (gid >= num_points) return;


    float px = pts[gid * 3 + 0];
    float py = pts[gid * 3 + 1];
    float pz = pts[gid * 3 + 2];

    float best_dists[K];
    int best_indices[K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        best_dists[i] = 1.0e30f;
        best_indices[i] = -1;
    }

    for (int j = 0; j < num_points; j++) {
        
        float tx = pts[j * 3 + 0];
        float ty = pts[j * 3 + 1];
        float tz = pts[j * 3 + 2];

        float dx = px - tx;
        float dy = py - ty;
        float dz = pz - tz;

        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < best_dists[K - 1]) {
            int insert_pos = K - 1;

            while (insert_pos > 0 && dist_sq < best_dists[insert_pos - 1]) {
                best_dists[insert_pos] = best_dists[insert_pos - 1];
                best_indices[insert_pos] = best_indices[insert_pos - 1];
                insert_pos--;
            }

            best_dists[insert_pos] = dist_sq;
            best_indices[insert_pos] = j;
        }
    }

    int base_idx = gid * K;
    
    #pragma unroll
    for (int i = 0; i < K; i++) {
        out_indices[base_idx + i] = best_indices[i];
        if (out_dists_sq) {
            out_dists_sq[base_idx + i] = best_dists[i];
        }
    }
}