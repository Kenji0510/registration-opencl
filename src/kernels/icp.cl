// OpenCL 1.
// ICP (Point-to-Plane)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define BLOCK_SIZE 64


static inline int float_as_int(float x) { 
    return as_int(x);
}
static inline float int_as_float(int x) { 
    return as_float(x);
}

static inline float atomic_add_f32_global(volatile __global float* addr, float val) {
    volatile __global int* iaddr = (volatile __global int*)addr;
    int old = *iaddr;
    while (1) {
        float fold = int_as_float(old);
        float fnew = fold + val;
        int inew = float_as_int(fnew);
        int prev = atomic_cmpxchg(iaddr, old, inew);
        if (prev == old) {
            return fold;
        }
        old = prev;
    }
}

__kernel void compute_pt2pl_linear_system(
    __global const float* restrict d_src_pts,  // num_source * 3
    __global const float* restrict d_tgt_pts,  // num_target * 3
    __global const float* restrict d_tgt_normals,  // num_target * 3
    __global const int* restrict d_indices,  // num_source * 3
    __global const float* restrict d_dists_sq,  // num_source * 3
    const int num_source,
    const int num_target,
    const float max_dist_sq,
    __global float* restrict d_H,   // 6x6
    __global float* restrict d_b   // 1x6
) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);

    float local_H[36];
    float local_b[6];
    for (int i = 0; i < 36; i++) local_H[i] = 0.0f;
    for (int i = 0; i < 6; i++) local_b[i] = 0.0f;

    if (gid >= num_source) {
        return;
    }

    int target_idx = d_indices[gid];
    float dist_sq = d_dists_sq[gid];

    if (target_idx >= 0 && (uint)target_idx < (uint)num_target && dist_sq <= max_dist_sq) {
        float ps[3];
        ps[0] = d_src_pts[gid * 3 + 0];
        ps[1] = d_src_pts[gid * 3 + 1];
        ps[2] = d_src_pts[gid * 3 + 2];

        float pt[3];
        pt[0] = d_tgt_pts[target_idx * 3 + 0];
        pt[1] = d_tgt_pts[target_idx * 3 + 1];
        pt[2] = d_tgt_pts[target_idx * 3 + 2];

        float nt[3];
        nt[0] = d_tgt_normals[target_idx * 3 + 0];
        nt[1] = d_tgt_normals[target_idx * 3 + 1];
        nt[2] = d_tgt_normals[target_idx * 3 + 2];

        float cross_x = ps[1] * nt[2] - ps[2] * nt[1];
        float cross_y = ps[2] * nt[0] - ps[0] * nt[2];
        float cross_z = ps[0] * nt[1] - ps[1] * nt[0];

        float J[6];
        J[0] = cross_x;
        J[1] = cross_y;
        J[2] = cross_z;
        J[3] = nt[0];
        J[4] = nt[1];
        J[5] = nt[2];

        // res = (p_t - p_s) . n_t
        float diff[3];
        diff[0] = pt[0] - ps[0];
        diff[1] = pt[1] - ps[1];
        diff[2] = pt[2] - ps[2];

        float res = diff[0] * nt[0] + diff[1] * nt[1] + diff[2] * nt[2];

        for (int i = 0; i < 6; i++) {
            local_b[i] += J[i] * res;
        }

        #pragma unroll
        for (int r = 0; r < 6; r++) {
            float Jr = J[r];
            #pragma unroll
            for (int c = 0; c < 6; c++) {
                local_H[r * 6 + c] += Jr * J[c];
            }
        }
    }
    

    __local float lH[BLOCK_SIZE * 36];
    __local float lb[BLOCK_SIZE * 6];

    int baseH = lid * 36;
    int baseB = lid * 6;
    
    for (int i = 0; i < 36; i++) lH[baseH + i] = local_H[i];
    for (int i = 0; i < 6; i++)  lb[baseB + i] = local_b[i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            int otherH = (lid + offset) * 36;
            int otherB = (lid + offset) * 6;
            
            // Sum H
            for (int i = 0; i < 36; i++) {
                lH[baseH + i] += lH[otherH + i];
            }
            // Sum b
            for (int i = 0; i < 6; i++) {
                lb[baseB + i] += lb[otherB + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        for (int i = 0; i < 36; i++) {
            atomic_add_f32_global(&d_H[i], lH[baseH + i]);
        }
        for (int i = 0; i < 6; i++) {
            atomic_add_f32_global(&d_b[i], lb[baseB + i]);
        }
    }
}