// OpenCL 1.2

#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFUL
#define P1 73856093UL
#define P2 19349663UL
#define P3 83492791UL


inline ulong compute_hash(int vx, int vy, int vz) {
    return ((ulong)(vx * P1) ^ (ulong)(vy * P2) ^ (ulong)(vz * P3));
}

inline int lookup_table(
    ulong key,
    __global const ulong* restrict table_keys,
    int table_size
) {
    int idx = key % table_size;

    for (int i = 0; i < 50; ++i) {
        ulong k = table_keys[idx];
        if (k == key) return idx;
        if (k == EMPTY_KEY) return -1;
        idx = (idx + 1) % table_size;
    }
    return -1;
}

__kernel void find_nearest_neighbor(
    __global const float* restrict source_pts,
    int num_source,
    float voxel_size,
    __global const ulong* restrict target_keys,
    __global const float* restrict target_centroids,
    __global const int* restrict target_counts,
    __global const int* restrict table_remap,
    int table_size,
    __global int* restrict out_indices,
    __global float* restrict out_dists_sq
) {
    int idx = get_global_id(0);

    if (idx >= num_source) return;

    float px = source_pts[idx * 3 + 0];
    float py = source_pts[idx * 3 + 1];
    float pz = source_pts[idx * 3 + 2];

    int vx = (int)floor(px / voxel_size);
    int vy = (int)floor(py / voxel_size);
    int vz = (int)floor(pz / voxel_size);

    float best_dist_sq = 1.0e30f;
    int best_table_idx = -1;

    int search_range = 1;
    for (int dz = -search_range; dz <= search_range; ++dz) {
        for (int dy = -search_range; dy <= search_range; ++dy) {
            for (int dx = -search_range; dx <= search_range; ++dx) {
                ulong key = compute_hash(vx + dx, vy + dy, vz + dz);

                int t_idx = lookup_table(
                    key,
                    target_keys,
                    table_size
                );

                if (t_idx != -1 && target_counts[t_idx] > 0) {
                    float tx = target_centroids[t_idx * 3 + 0];
                    float ty = target_centroids[t_idx * 3 + 1];
                    float tz = target_centroids[t_idx * 3 + 2];

                    float diff_x = px - tx;
                    float diff_y = py - ty;
                    float diff_z = pz - tz;
                    float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_table_idx = t_idx;
                    }
                }
            }
        }
    }

    out_dists_sq[idx] = best_dist_sq;
    if (best_table_idx != -1) {
        out_indices[idx] = table_remap[best_table_idx];
    } else {
        out_indices[idx] = -1;
    }
}