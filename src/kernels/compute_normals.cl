// OpenCL 1.2

#ifndef K
#define K 15
#endif


inline void eigen_decomposition_3x3(
    float A[3][3],
    float evecs[3][3],
    float evals[3]
) {
    evecs[0][0] = 1.0f; evecs[0][1] = 0.0f; evecs[0][2] = 0.0f;
    evecs[1][0] = 0.0f; evecs[1][1] = 1.0f; evecs[1][2] = 0.0f;
    evecs[2][0] = 0.0f; evecs[2][1] = 0.0f; evecs[2][2] = 1.0f;

    const int max_iter = 15;

    for (int iter = 0; iter < max_iter; ++iter) {
        int p = 0, q = 1;

        float a01 = fabs(A[0][1]);
        float a02 = fabs(A[0][2]);
        float a12 = fabs(A[1][2]);

        float max_val;
        if (a01 >= a02 && a01 >= a12) {
            p = 0; q = 1; max_val = a01;
        } else if (a02 >= a01 && a02 >= a12) {
            p = 0; q = 2; max_val = a02;
        } else {
            p = 1; q = 2; max_val = a12;
        }

        if (max_val < 1e-6f) break;

        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];

        float phi = 0.5f * atan2(2.0f * apq, aqq - app);
        float c = cos(phi);
        float s = sin(phi);

        A[p][p] = c*c*app - 2.0f*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2.0f*s*c*apq + c*c*aqq;
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;

        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                float arp = A[r][p];
                float arq = A[r][q];
                A[r][p] = c*arp - s*arq;
                A[p][r] = A[r][p];
                A[r][q] = s*arp + c*arq;
                A[q][r] = A[r][q];
            }
        }

        for (int r = 0; r < 3; ++r) {
            float ervp = evecs[r][p];
            float ervq = evecs[r][q];
            evecs[r][p] = c*ervp - s*ervq;
            evecs[r][q] = s*ervp + c*ervq;
        }
    }

    evals[0] = A[0][0];
    evals[1] = A[1][1];
    evals[2] = A[2][2];
}

__kernel void compute_normals(
    __global const float* restrict points,           // N * 3
    __global const int* restrict neighbor_indices,   // N * K
    const int num_points,
    const float vp_x, const float vp_y, const float vp_z, // Viewpoint
    __global float* restrict out_normals             // N * 3
) {
    const int idx = (int)get_global_id(0);
    if (idx >= num_points) {
        return;
    }

    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    int valid_count = 0;

    float cache_x[K];
    float cache_y[K];
    float cache_z[K];

    const int neighbor_base = idx * K;

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        int n_idx = neighbor_indices[neighbor_base + i];
        
        if (n_idx < 0 || n_idx >= num_points) {
            break; 
        }

        float tx = points[n_idx * 3 + 0];
        float ty = points[n_idx * 3 + 1];
        float tz = points[n_idx * 3 + 2];

        cache_x[valid_count] = tx;
        cache_y[valid_count] = ty;
        cache_z[valid_count] = tz;

        sum_x += tx;
        sum_y += ty;
        sum_z += tz;
        valid_count++;
    }

    if (valid_count < 3) {
        out_normals[idx * 3 + 0] = 0.0f;
        out_normals[idx * 3 + 1] = 0.0f;
        out_normals[idx * 3 + 2] = 0.0f;
        return;
    }

    const float inv_n = 1.0f / (float)valid_count;
    const float mean_x = sum_x * inv_n;
    const float mean_y = sum_y * inv_n;
    const float mean_z = sum_z * inv_n;

    float mat[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };

    for (int i = 0; i < valid_count; ++i) {
        const float dx = cache_x[i] - mean_x;
        const float dy = cache_y[i] - mean_y;
        const float dz = cache_z[i] - mean_z;

        mat[0][0] += dx * dx;
        mat[0][1] += dx * dy;
        mat[0][2] += dx * dz;
        mat[1][1] += dy * dy;
        mat[1][2] += dy * dz;
        mat[2][2] += dz * dz;
    }

    mat[0][0] *= inv_n; mat[0][1] *= inv_n; mat[0][2] *= inv_n;
    mat[1][1] *= inv_n; mat[1][2] *= inv_n; mat[2][2] *= inv_n;
    
    mat[1][0] = mat[0][1];
    mat[2][0] = mat[0][2];
    mat[2][1] = mat[1][2];

    float evecs[3][3];
    float evals[3];
    eigen_decomposition_3x3(mat, evecs, evals);

    int min_idx = 0;
    if (evals[1] < evals[min_idx]) min_idx = 1;
    if (evals[2] < evals[min_idx]) min_idx = 2;

    float nx = evecs[0][min_idx];
    float ny = evecs[1][min_idx];
    float nz = evecs[2][min_idx];

    float vx = vp_x - mean_x;
    float vy = vp_y - mean_y;
    float vz = vp_z - mean_z;

    float dot = nx * vx + ny * vy + nz * vz;
    if (dot < 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    out_normals[idx * 3 + 0] = nx;
    out_normals[idx * 3 + 1] = ny;
    out_normals[idx * 3 + 2] = nz;
}