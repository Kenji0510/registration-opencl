// OpenCL 1.2
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFUL
#define P1 73856093UL
#define P2 19349663UL
#define P3 83492791UL
#define SHARED_TABLE_SIZE 1536
#define SHARED_PROBE 32
#define GLOBAL_PROBE 1000

inline void atomic_add_float_global(volatile __global float* addr_f, float val) {
    volatile __global uint* addr_u = (volatile __global uint*)addr_f;
    union { uint u; float f; } cur, next;

    while (1) {
        cur.u = *addr_u;
        next.f = as_float(cur.u) + val;

        uint prev = atomic_cmpxchg(addr_u, cur.u, as_uint(next.f));
        if (prev == cur.u) break;
    }
}

inline void atomic_add_float_local(volatile __local float* addr_f, float val) {
    volatile __local uint* addr_u = (volatile __local uint*)addr_f;
    union { uint u; float f; } cur, next;

    while (1) {
        cur.u = *addr_u;
        next.f = as_float(cur.u) + val;
        uint prev = atomic_cmpxchg(addr_u, cur.u, as_uint(next.f));
        if (prev == cur.u) break;
    }
}

inline ulong cas_u64_global(volatile __global ulong* p, ulong expected, ulong desired) {
    return atom_cmpxchg(p, expected, desired);
}

inline ulong cas_u64_local(volatile __local ulong* p, ulong expected, ulong desired) {
    return atom_cmpxchg(p, expected, desired);
}

inline unsigned long compute_voxel_hash(float px, float py, float pz, float voxel_size) {
    float inv_voxel = 1.0 / voxel_size;
    int vx = convert_int_rtn(px * inv_voxel);
    int vy = convert_int_rtn(py * inv_voxel);
    int vz = convert_int_rtn(pz * inv_voxel);
    return ((unsigned long)vx * P1) ^ ((unsigned long)vy * P2) ^ ((unsigned long)vz * P3);
}

inline void add_to_global(
    unsigned long hash_key,
    float px, float py, float pz,
    int count,
    volatile __global ulong* table_keys,
    volatile __global float* table_centroids,
    volatile __global int* table_counts,
    int table_size
){
    int table_idx = (int)(hash_key % (unsigned long)table_size);

    for (int i = 0; i < GLOBAL_PROBE; ++i) {
        ulong old_key = cas_u64_global(&table_keys[table_idx], EMPTY_KEY, (ulong)hash_key);

        if (old_key == EMPTY_KEY || old_key == (ulong)hash_key) {
            atomic_add_float_global(&table_centroids[3 * table_idx + 0], px);
            atomic_add_float_global(&table_centroids[3 * table_idx + 1], py);
            atomic_add_float_global(&table_centroids[3 * table_idx + 2], pz);
            atomic_add(&table_counts[table_idx], count);
            return;
        }
        table_idx = (table_idx + 1) % table_size;
    }
}

__kernel void init_table(__global ulong* table_keys, __global int* table_remap, int table_size) {
    int idx = get_global_id(0);
    if (idx >= table_size) return;
    table_keys[idx] = EMPTY_KEY;
    table_remap[idx] = -1;
}

__kernel void insert_points(
    __global const float* points,
    int num_points,
    float voxel_size,
    volatile __global ulong* table_keys,
    volatile __global float* table_centroids,
    volatile __global int* table_counts,
    int table_size
){
    __local ulong  s_keys[SHARED_TABLE_SIZE];
    __local float s_centroids[SHARED_TABLE_SIZE * 3];
    __local int   s_counts[SHARED_TABLE_SIZE];

    int tid = get_local_id(0);
    int ldim = get_local_size(0);
    int idx = get_global_id(0);

    for (int i = tid; i < SHARED_TABLE_SIZE; i += ldim) {
        s_keys[i] = EMPTY_KEY;
        s_centroids[i * 3 + 0] = 0.0f;
        s_centroids[i * 3 + 1] = 0.0f;
        s_centroids[i * 3 + 2] = 0.0f;
        s_counts[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < num_points) {
        float px = points[idx * 3 + 0];
        float py = points[idx * 3 + 1];
        float pz = points[idx * 3 + 2];

        unsigned long hash_key = compute_voxel_hash(px, py, pz, voxel_size);
        int s_idx = (int)(hash_key % (unsigned long)SHARED_TABLE_SIZE);

        int stored = 0;
        for (int probe = 0; probe < SHARED_PROBE; ++probe) {
            // ulong old = atomic_cmpxchg(&s_keys[s_idx], EMPTY_KEY, (ulong)hash_key);
            unsigned long old = cas_u64_local((volatile __local ulong*)&s_keys[s_idx], EMPTY_KEY, (ulong)hash_key);

            if (old == EMPTY_KEY || old == (ulong)hash_key) {
                atomic_add_float_local(&s_centroids[s_idx * 3 + 0], px);
                atomic_add_float_local(&s_centroids[s_idx * 3 + 1], py);
                atomic_add_float_local(&s_centroids[s_idx * 3 + 2], pz);
                atomic_add(&s_counts[s_idx], 1);
                stored = 1;
                break;
            }
            s_idx = (s_idx + 1) % SHARED_TABLE_SIZE;
        }

        if (!stored) {
            add_to_global(hash_key, px, py, pz, 1, table_keys, table_centroids, table_counts, table_size);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = tid; i < SHARED_TABLE_SIZE; i += ldim) {
        ulong key = s_keys[i];
        if (key != EMPTY_KEY) {
            float sx = s_centroids[i * 3 + 0];
            float sy = s_centroids[i * 3 + 1];
            float sz = s_centroids[i * 3 + 2];
            int sc = s_counts[i];
            
            add_to_global((unsigned long)key, sx, sy, sz, sc, table_keys, table_centroids, table_counts, table_size);
        }
    }
}

__kernel void average_table(__global float* table_centroids, __global int* table_counts, int table_size) {
    int idx = get_global_id(0);
    if (idx >= table_size) return;

    int cnt = table_counts[idx];
    if (cnt > 1) {
        float inv = 1.0f / (float)cnt;
        table_centroids[idx * 3 + 0] *= inv;
        table_centroids[idx * 3 + 1] *= inv;
        table_centroids[idx * 3 + 2] *= inv;
    }
}

__kernel void compact_voxels(
    __global const ulong* table_keys,
    __global const float* table_centroids,
    __global const int* table_counts,
    __global int* table_remap,
    int table_size,
    __global float* out_points,
    volatile __global int* out_count
){
    int idx = get_global_id(0);
    if (idx >= table_size) return;

    if (table_keys[idx] != EMPTY_KEY && table_counts[idx] > 0) {
        float sx = table_centroids[idx * 3 + 0];
        float sy = table_centroids[idx * 3 + 1];
        float sz = table_centroids[idx * 3 + 2];

        int w = atomic_add(out_count, 1);
        out_points[w * 3 + 0] = sx;
        out_points[w * 3 + 1] = sy;
        out_points[w * 3 + 2] = sz;

        table_remap[idx] = w;
    }
}
