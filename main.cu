#include <iostream>
#include <functional>
#include "matrix_utils.cpp"
#include <cuda_runtime.h>
#include <iomanip>

#define RANDOM 1
#define FROM_FILE 2

#define BLOCK_DIM 1<<10
#define COL_PER_BLK 3

#define min(a, b) ((a<b)?a:b)

#define MARGIN 1e-8
#define IS_ZERO(x) (abs(x)<=MARGIN)
#define NOT_ZERO(x) (abs(x)>MARGIN)


using namespace std;

__device__ void normalize_row(double *target_row, const double *base_row, double scale, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        double temp = target_row[i + offset] - (base_row[i] * scale);
        target_row[i + offset] = (IS_ZERO(temp) ? 0 : temp);
    }
}

__device__ void
normalize_self(double *self, double const *self_but_in_share_memory, double scale, size_t n, size_t offset) {
    for (size_t i = 0; i < n; i++) {
        self[i + offset] = self_but_in_share_memory[i] / scale;
    }
}

__global__ void gje_inverse(double *m2, size_t n, size_t base_row_index, double *scale) {
    size_t m2_width = 2 * n;
    extern __shared__ double base_row[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ofs = COL_PER_BLK * bid;

    if (tid > n)return;


    if (tid == 0) {
        for (size_t i = 0; i < COL_PER_BLK; i++)
            base_row[i] = m2[(base_row_index * m2_width) + (ofs + i)];
    }
    __syncthreads();

    size_t max_cols=min((2*n)-ofs,3);
//    size_t max_cols = 3;

    if (tid == base_row_index) {
        normalize_self(&m2[tid * m2_width], base_row, scale[tid], max_cols, ofs);
    } else
        normalize_row(&m2[tid * m2_width], base_row, scale[tid], max_cols, ofs);
}

__global__ void gje_scale_calc(double *m2d, size_t n, size_t current_row, double *scale) {
    size_t m2_width = 2 * n;
    unsigned int tid = threadIdx.x;

    __shared__ double diag;
    double base = 0;

    if (tid == current_row)
        diag = m2d[current_row * m2_width + current_row];
    else
        base = m2d[tid * m2_width + current_row];
    __syncthreads();

    if (tid == current_row)
        scale[tid] = diag;
    else
        scale[tid] = base / diag;
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
    double **m2_h = mxalloc(n, n, malloc);
    double **inv_h = mxalloc(n, n, malloc);
    if (mode == RANDOM) {
        fill_random(n, m2_h, pair<float, float>(-1e6, 1e6));
    } else {
        get_from_file(n, m2_h, path);
    }
    print_matrix(m2_h, n, n);

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim((2 * n) / COL_PER_BLK + ((2 * n) % COL_PER_BLK != 0));

    size_t m2_width = 2 * n;
    double *m2_d = nullptr, *scale_d = nullptr;
    int error = 0;
    error |= cudaMalloc((void **) &m2_d, n * m2_width * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        error |= cudaMemcpy(m2_d + i * m2_width, m2_h[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }
    error |= cudaMalloc((void **) &scale_d, n * sizeof(double));
    if (error != cudaSuccess) {
        cout << "couldn't allocate memory in device";
        cout << cudaGetErrorString((cudaError_t) error);
    }
    gje_set_identity<<<dim3(1), block_dim>>>(m2_d, n);
    cudaDeviceSynchronize();

//    // check identity matrix
//    for (size_t i = 0; i < n; ++i) {
//        error |= cudaMemcpy(inv_h[i], &m2_d[i * m2_width + n], sizeof(double) * n, cudaMemcpyDeviceToHost);
//    }
//    print_matrix(inv_h, n, n);
    double **temp2_h = mxalloc(n, 2 * n, malloc);

    for (size_t i = 0; i < n; i++) {

        gje_scale_calc<<<1, block_dim>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();
        double *temp = (double *) malloc(sizeof(double) * n);
        error |= cudaMemcpy(temp, scale_d, sizeof(double) * n, cudaMemcpyDeviceToHost);
        for (int j = 0; j < n; ++j)cerr << std::setprecision(2) << temp[j] << "\t";
        cerr << "\n";

        // check matrix before
        for (size_t j = 0; j < n; ++j) {
            error |= cudaMemcpy(temp2_h[j], &m2_d[j * m2_width], sizeof(double) * 2 * n, cudaMemcpyDeviceToHost);
        }
        cerr << "print M before:\n";
        print_matrix(temp2_h, n, 2 * n);


        gje_inverse<<<grid_dim, block_dim, COL_PER_BLK * sizeof(double)>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();

        // check matrix
        for (size_t j = 0; j < n; ++j) {
            error |= cudaMemcpy(temp2_h[j], &m2_d[j * m2_width], sizeof(double) * 2 * n, cudaMemcpyDeviceToHost);
        }
        cerr << "print M:\n";
        print_matrix(temp2_h, n, 2 * n);
        cerr << "\n\n\n";


    }

    for (size_t i = 0; i < n; ++i) {
        error |= cudaMemcpy(inv_h[i], &m2_d[i * m2_width + n], sizeof(double) * n, cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
        cout << "couldn't retrieve result";
        cout << cudaGetErrorString((cudaError_t) error);
    }
    print_matrix(inv_h, n, n);
    cout << inverse_test(m2_h, inv_h, n);
    mxfree(m2_h, n, free);
    mxfree(inv_h, n, free);
}
