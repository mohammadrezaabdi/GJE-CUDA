#include <iostream>
#include <functional>
#include "matrix_utils.cpp"
#include <cuda_runtime.h>
#include <iomanip>

#define RANDOM 1
#define FROM_FILE 2

#define BLOCK_DIM 1024
#define COL_PER_BLK 5

#define min(a, b) ((a<b)?a:b)


using namespace std;

__device__ void normalize_row(double *target_row, const double *base_row, double scale, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        double temp = target_row[i + offset] - (base_row[i] * scale);
//        target_row[i + offset] = (IS_ZERO(temp) ? 0 : temp);
        target_row[i + offset] = temp;
    }
}

__device__ void
normalize_self(double *self, double const *self_but_in_share_memory, double scale, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        self[i + offset] = self_but_in_share_memory[i] / scale;
    }
}

// todo: blockDim < n
__global__ void gje_inverse(double *m2, size_t n, size_t base_row_index, double *scale) {
    size_t m2_width = 2 * n;
    extern __shared__ double base_row[];
    unsigned int tid = threadIdx.x;
    unsigned int ofs = COL_PER_BLK * blockIdx.x;

    if (tid > n)
        return;

    if (tid == 0) {
        for (size_t i = 0; i < COL_PER_BLK; i++)
            base_row[i] = m2[(base_row_index * m2_width) + (ofs + i)];
    }
    __syncthreads();

    size_t num_cols = min((2 * n) - ofs, COL_PER_BLK);

    if (tid == base_row_index) {
        normalize_self(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);
    } else
        normalize_row(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);
}

// todo: blockDim < n
__global__ void gje_scale_calc(const double *m2d, size_t n, size_t current_row, double *scale) {
    unsigned int tid = threadIdx.x;
    unsigned int ofs = COL_PER_BLK * blockIdx.x;
    __shared__ double diag;
    __shared__ size_t m2d_width;
    double base = 0;
    if (tid == 0) {
        m2d_width = 2 * n;
    }
    __syncthreads();

    if (tid > n)
        return;

    if (tid == current_row)
        diag = m2d[current_row * m2d_width + current_row];
    else
        base = m2d[tid * m2d_width + current_row];
    __syncthreads();

    if (tid == current_row)
        scale[tid] = diag;
    else
        scale[tid] = base / diag;
}

// todo: blockDim < n
__global__ void gje_set_identity(double *m2d, size_t n) {
    unsigned int tid = threadIdx.x;
    __shared__ size_t m2d_width;
    if (tid == 0) {
        m2d_width = 2 * n;
    }
    __syncthreads();

    unsigned int idx = COL_PER_BLK * blockIdx.x + tid;
    m2d[idx * m2d_width + (n + idx)] = 1.0;
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
    if (mode == RANDOM) {
        fill_random(n, m_h, pair<float, float>(-1e6, 1e6));
    } else {
        get_from_file(n, m_h, path);
    }
    double **cpu_inv = mxalloc(n, n, malloc);
    inverse(m_h, n, cpu_inv);
//    print_matrix(m_h, n, n);

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim((2 * n) / COL_PER_BLK + ((2 * n) % COL_PER_BLK != 0));

    size_t m2_width = 2 * n;
    double *m2_d = nullptr, *scale_d = nullptr, *temp2_h = nullptr;;
    int error = 0;
    error |= cudaMallocHost((void **) &temp2_h, sizeof(double) * n * m2_width);
    if (error != cudaSuccess) {
        cout << "couldn't allocate memory in host" << endl;
        cout << cudaGetErrorString((cudaError_t) error) << endl;
    }
    error |= cudaMalloc((void **) &m2_d, n * m2_width * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        error |= cudaMemcpy(&m2_d[i * m2_width], m_h[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }
    error |= cudaMalloc((void **) &scale_d, n * sizeof(double));
    if (error != cudaSuccess) {
        cout << "couldn't allocate memory in device" << endl;
        cout << cudaGetErrorString((cudaError_t) error) << endl;
    }

    unsigned int grid_dim1 = n / COL_PER_BLK + (n % COL_PER_BLK != 0);
    gje_set_identity<<<    dim3(grid_dim1), dim3(COL_PER_BLK)>>>(m2_d, n);
    cudaDeviceSynchronize();

    error |= cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "kernel error" << endl;
        cout << cudaGetErrorString((cudaError_t) error) << endl;
    }

    for (size_t i = 0; i < n; i++) {
        gje_scale_calc<<<1, block_dim>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();

        error |= cudaGetLastError();
        if (error != cudaSuccess) {
            cout << "kernel error" << endl;
            cout << cudaGetErrorString((cudaError_t) error) << endl;
        }

        gje_inverse<<<grid_dim, block_dim, COL_PER_BLK * sizeof(double)>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();

        error |= cudaGetLastError();
        if (error != cudaSuccess) {
            cout << "kernel error" << endl;
            cout << cudaGetErrorString((cudaError_t) error) << endl;
        }
    }

    error |= cudaMemcpy(temp2_h, m2_d, sizeof(double) * n * m2_width, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cout << "couldn't retrieve result" << endl;
        cout << cudaGetErrorString((cudaError_t) error) << endl;
    }
//    print_matrix(temp2_h, n, 2 * n);

    double **inv_h = mxalloc(n, n, malloc);
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(inv_h[i], &temp2_h[i * m2_width + n], sizeof(double) * n, cudaMemcpyHostToHost);
    }
//    print_matrix(inv_h, n, n);

    cout << "cpu " << inverse_test(m_h, cpu_inv, n) << endl;
    cout << "gpu " << inverse_test(m_h, inv_h, n) << endl;
    cudaFree(m2_d);
    cudaFree(scale_d);
    mxfree(m_h, n, free);
    mxfree(inv_h, n, free);
    cudaFreeHost(temp2_h);
}
