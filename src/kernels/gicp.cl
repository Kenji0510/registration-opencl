// OpenCL 1.2

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

static inline int invert3x3Sym_safe(const float* src, float* dst) {
    float det = 
        src[0] * (src[4] * src[8] - src[5] * src[7]) -
        src[1] * (src[3] * src[8] - src[5] * src[6]) +
        src[2] * (src[3] * src[7] - src[4] * src[6]);

    if (fabs(det) < 1e-12f) {
        return 0;
    }
    float invDet = 1.0f / det;

    dst[0] =  (src[4] * src[8] - src[5] * src[7]) * invDet;
    dst[1] = (src[2] * src[7] - src[1] * src[8]) * invDet;
    dst[2] = (src[1] * src[5] - src[2] * src[4]) * invDet;

    dst[3] = dst[1];
    dst[4] = (src[0] * src[8] - src[2] * src[6]) * invDet;
    dst[5] = (src[2] * src[3] - src[0] * src[5]) * invDet;

    dst[6] = dst[2];
    dst[7] = dst[5];
    dst[8] = (src[0] * src[4] - src[1] * src[3]) * invDet;

    return 1;
}

static inline void mul3(const float* A, const float* v, float* out) {
    out[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];
    out[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];
    out[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];
}

__kernel void compute_gicp_linear_system(
    __global const float* restrict d_src_pts,  // num_source * 3
    __global const float* restrict d_src_covs,  // num_source * 9
    __global const float* restrict d_tgt_pts,  // num_target * 3
    __global const float* restrict d_tgt_covs,  // num_target * 9
    __global const int* restrict d_indices,  // num_source
    __global const float* restrict d_dists_sq,  // num_source
    const int num_source,
    const int num_target,
    const float max_dist_sq,
    __global float* restrict d_H,  // 6 * 6
    __global float* restrict d_b // 6
) {
    int gid = (int)get_global_id(0);
    int lid = (int)get_local_id(0);
    // int lsize = (int)get_local_size(0);
    int lsize = BLOCK_SIZE;

    float local_H[36];
    float local_b[6];
    for (int i = 0; i < 36; i++) local_H[i] = 0.0f;
    for (int i = 0; i < 6; i++) local_b[i] = 0.0f;

    if (gid < num_source) {
        int target_idx = d_indices[gid];
        float dist_sq = d_dists_sq[gid];

        if ((uint)target_idx < (uint)num_target && dist_sq <= max_dist_sq) {
            float ps[3], pt[3];
            ps[0] = d_src_pts[gid * 3 + 0];
            ps[1] = d_src_pts[gid * 3 + 1];
            ps[2] = d_src_pts[gid * 3 + 2];

            pt[0] = d_tgt_pts[target_idx * 3 + 0];
            pt[1] = d_tgt_pts[target_idx * 3 + 1];
            pt[2] = d_tgt_pts[target_idx * 3 + 2];

            float C_sum[9];
            for (int k = 0; k < 9; k++) {
                C_sum[k] = d_tgt_covs[target_idx * 9 + k] + d_src_covs[gid * 9 + k];
            }

            float Omega[9];
            if (invert3x3Sym_safe(C_sum, Omega)) {
                float err[3] = {
                    pt[0] - ps[0],
                    pt[1] - ps[1],
                    pt[2] - ps[2]
                };
                float We[3];
                mul3(Omega, err, We);

                float x = ps[0], y = ps[1], z = ps[2];

                local_b[0] = y * We[2] - z * We[1];
                local_b[1] = z * We[0] - x * We[2];
                local_b[2] = x * We[1] - y * We[0];

                local_b[3] = We[0];
                local_b[4] = We[1];
                local_b[5] = We[2];

                float s0[3] = {0.0f, -z, y};
                float s1[3] = {z, 0.0f, -x};
                float s2[3] = {-y, x, 0.0f};

                float ws0[3], ws1[3], ws2[3];
                mul3(Omega, s0, ws0);
                mul3(Omega, s1, ws1);
                mul3(Omega, s2, ws2);

                float hcol0[3], hcol1[3], hcol2[3];
                hcol0[0] = y*ws0[2] - z*ws0[1];
                hcol0[1] = z*ws0[0] - x*ws0[2];
                hcol0[2] = x*ws0[1] - y*ws0[0];

                hcol1[0] = y*ws1[2] - z*ws1[1];
                hcol1[1] = z*ws1[0] - x*ws1[2];
                hcol1[2] = x*ws1[1] - y*ws1[0];

                hcol2[0] = y*ws2[2] - z*ws2[1];
                hcol2[1] = z*ws2[0] - x*ws2[2];
                hcol2[2] = x*ws2[1] - y*ws2[0];

                // H_rr
                local_H[0*6+0]=hcol0[0]; local_H[0*6+1]=hcol1[0]; local_H[0*6+2]=hcol2[0];
                local_H[1*6+0]=hcol0[1]; local_H[1*6+1]=hcol1[1]; local_H[1*6+2]=hcol2[1];
                local_H[2*6+0]=hcol0[2]; local_H[2*6+1]=hcol1[2]; local_H[2*6+2]=hcol2[2];

                // H_tt = Omega
                local_H[3*6+3]=Omega[0]; local_H[3*6+4]=Omega[1]; local_H[3*6+5]=Omega[2];
                local_H[4*6+3]=Omega[3]; local_H[4*6+4]=Omega[4]; local_H[4*6+5]=Omega[5];
                local_H[5*6+3]=Omega[6]; local_H[5*6+4]=Omega[7]; local_H[5*6+5]=Omega[8];

                // H_rt
                local_H[0*6+3]=ws0[0]; local_H[0*6+4]=ws0[1]; local_H[0*6+5]=ws0[2];
                local_H[1*6+3]=ws1[0]; local_H[1*6+4]=ws1[1]; local_H[1*6+5]=ws1[2];
                local_H[2*6+3]=ws2[0]; local_H[2*6+4]=ws2[1]; local_H[2*6+5]=ws2[2];

                // H_tr = transpose(H_rt)
                local_H[3*6+0]=local_H[0*6+3];
                local_H[3*6+1]=local_H[1*6+3];
                local_H[3*6+2]=local_H[2*6+3];
                local_H[4*6+0]=local_H[0*6+4];
                local_H[4*6+1]=local_H[1*6+4];
                local_H[4*6+2]=local_H[2*6+4];
                local_H[5*6+0]=local_H[0*6+5];
                local_H[5*6+1]=local_H[1*6+5];
                local_H[5*6+2]=local_H[2*6+5];
            }
        }
    }

    __local float lH[BLOCK_SIZE * 36];
    __local float lb[BLOCK_SIZE * 6];

    int baseH = lid * 36;
    int baseB = lid * 6;
    
    // Copy to local
    for (int i=0;i<36;i++) lH[baseH+i] = local_H[i];
    for (int i=0;i<6;i++)  lb[baseB+i] = local_b[i];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            int otherH = (lid + offset) * 36;
            int otherB = (lid + offset) * 6;
            for (int i=0;i<36;i++) lH[baseH+i] += lH[otherH+i];
            for (int i=0;i<6;i++)  lb[baseB+i] += lb[otherB+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- global accumulation ----
    if (lid == 0) {
        for (int i=0;i<36;i++) atomic_add_f32_global(&d_H[i], lH[i]);
        for (int i=0;i<6;i++)  atomic_add_f32_global(&d_b[i], lb[i]);
    }
}