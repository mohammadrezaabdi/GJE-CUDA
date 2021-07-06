#include <iostream>
#include <functional>
#include "matrix_utils.cpp"
#include <cuda_runtime.h>

#define RANDOM 1
#define FROM_FILE 2

#define BLOCK_DIM 1<<10
#define COL_PER_BLK 5
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

__global__ void gje_inverse(double *m2d, size_t n, size_t cr, double *scl) {
    size_t m2d_width = 2 * n;
    extern __shared__ double mr[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ofs = COL_PER_BLK * bid;

    if (tid == 0) {
        for (size_t i = 0; i < COL_PER_BLK; ++i)
            mr[i] = m2d[(cr * m2d_width) + (ofs + i)];
    }
    __syncthreads();
    if (tid == cr) {
        normalize_self(&m2d[tid * m2d_width], scl[tid], COL_PER_BLK, ofs);
    } else
        normalize_row(mr, &m2d[tid * m2d_width], scl[tid], COL_PER_BLK, 0, ofs);
}

__global__ void gje_scale_calc(double *m2d, size_t n, size_t current_row, double *scale) {
    size_t m2d_width = 2 * n;
    unsigned int tid = threadIdx.x;
    __shared__
    double diag;

    if (tid == 0)
        diag = m2d[current_row * m2d_width + current_row];
    __syncthreads();

    if (tid == current_row)
        scale[tid] = diag;
    else
        scale[tid] = m2d[tid + m2d_width + current_row] / diag;
}

__global__ void gje_set_identity(double *m2d, size_t n) {
    unsigned int tid = threadIdx.x;
    size_t m2d_width = 2 * n;
    m2d[(tid * m2d_width) + (n + tid)] = 1;
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

    size_t m2d_width = 2 * n;
    double *m2d = nullptr, *scale_d = nullptr;
    int error = 0;
    error |= cudaMalloc((void **) &m2d, n * m2d_width * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        error |= cudaMemcpy(&m2d[i * m2d_width], &m_h[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim((2 * n) / COL_PER_BLK + ((2 * n) % COL_PER_BLK != 0));
    error |= cudaMalloc((void **) &scale_d, n * sizeof(double));
    if (error != cudaSuccess) {
        cout << "couldn't allocate memory in device";
        cout << cudaGetErrorString((cudaError_t) error);
    }
    gje_set_identity<<<dim3(1), block_dim>>>(m2d, n);
    cudaDeviceSynchronize();

    // check identity matrix
    for (size_t i = 0; i < n; ++i) {
        error |= cudaMemcpy(inv_h[i], &m2d[i * m2d_width + n], sizeof(double) * n, cudaMemcpyDeviceToHost);
    }
    print_matrix(inv_h, n, n);

int i=0;
//    for (size_t i = 0; i < n; i++) {

        gje_scale_calc<<<1, block_dim>>>(m2d, n, i, scale_d);
        cudaDeviceSynchronize();
        double *temp = (double *) malloc(sizeof(double) * n);
        error |= cudaMemcpy(temp, scale_d, sizeof(double) * n, cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; ++i)cerr<<temp[i]<<"\t";cerr<<"\n";

        gje_inverse<<<grid_dim, block_dim, COL_PER_BLK * sizeof(double)>>>(m2d, n, i, scale_d);
        cudaDeviceSynchronize();
//    }

    for (size_t i = 0; i < n; ++i) {
        error |= cudaMemcpy(inv_h[i], &m2d[i * m2d_width + n], sizeof(double) * n, cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
        cout << "couldn't retrieve result";
        cout << cudaGetErrorString((cudaError_t) error);
    }
    print_matrix(inv_h, n, n);
    cout << inverse_test(m_h, inv_h, n);
    mxfree(m_h, n, free);
    mxfree(inv_h, n, free);
}
