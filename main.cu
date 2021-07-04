#include <iostream>
#include<functional>
#include "matrix_utils.cpp"
#include <cuda_runtime.h>

#define RANDOM 1
#define FROM_FILE 2

#define BLOCK_DIM 1<<10
#define COL_P_BLK 5
using namespace std;

__device__ void normalize_row(const double *base_row, double *target_row, double scale, size_t n, size_t base_offset,
                              size_t target_offset) {
    for (size_t i = 0; i < n; i++) {
        target_row[i + target_offset] -= base_row[i + base_offset] * scale;
    }
}

__device__ void normalize_self(double *self, double scale, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        self[i + offset] /= scale;
    }
}

__global__ void gje_inverse(double **m, size_t n, size_t cr, double *scl) {
    extern __shared__ double mr[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ofs = COL_P_BLK * bid;

    if (tid == 0) {
        for (size_t i = 0; i < COL_P_BLK; ++i)
            mr[i] = m[cr][ofs + i];
    }
    __syncthreads();
    if (tid == cr) {
        normalize_self(m[tid], scl[tid], COL_P_BLK, ofs);
    } else
        normalize_row(mr, m[tid], scl[tid], COL_P_BLK, 0, ofs);
}

__global__ void gje_scale_calc(double **m, size_t n, size_t cr, double *scl) {
    unsigned int tid = threadIdx.x;
    __shared__ double diag;

    if (tid == 0)
        diag = m[cr][cr];
    __syncthreads();

    if (tid == cr)
        scl[tid] = diag;
    else
        scl[tid] = m[tid][cr] / diag;
}

int main(int argc, char **argv) {
    size_t n = 0;
    int mode = FROM_FILE;
    string path;

    // arg parsing
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                case 'n':
                    n = strtoul(argv[++i], nullptr, 0);
                    break;
                case 'r':
                    mode = RANDOM;
                    break;
                case 'f':
                    mode = FROM_FILE;
                    path = argv[++i];
                    break;
                default:
                    cout << "invalid arguments";
            }
        }
    }
    double **m_h = mxalloc(n, n, malloc);
    double **inv_h = mxalloc(n, n, malloc);
    if (mode == RANDOM) {
        fill_random(n, m_h, pair<float, float>(-1e6, 1e6));
    } else {
        get_from_file(n, m_h, path);
    }
    print_matrix(m_h, n, n);

    double **m_d = nullptr, *scl_d = nullptr;
    cudaMalloc((void **) &scl_d, n * sizeof(double));
    cudaMalloc((void **) &m_d, n * sizeof(double *));
    for (size_t i = 0; i < n; ++i) {
        cudaError_t err = cudaMalloc((void **) &m_d[i], 2 * n * sizeof(double));
        cout << cudaGetErrorString(err);
        cudaMemcpy(&m_d[i], m_h[i], n * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        m_d[i][n + i] = 1;
    }

    unsigned int grid_dim = (2 * n) / COL_P_BLK + ((2 * n) % COL_P_BLK != 0);
    dim3 BL(BLOCK_DIM);
    dim3 GR(grid_dim);
    for (size_t i = 0; i < n; ++i) {
        gje_scale_calc<<<1, BL>>>(m_d, n, i, scl_d);
        gje_inverse<<<GR, BL, COL_P_BLK * sizeof(double)>>>(m_d, n, i, scl_d);
        cudaDeviceSynchronize();
    }
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(&inv_h[i], &m_d[i][n], sizeof(double) * n, cudaMemcpyDeviceToHost);
    }
    mxfree(m_h, n, free);
    mxfree(inv_h, n, free);
    print_matrix(inv_h, n, n);
    cout << inverse_test(m_h, inv_h, n);
}
