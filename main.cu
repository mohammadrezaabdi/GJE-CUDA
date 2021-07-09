#include <iostream>
#include <functional>
#include "matrix_utils.cpp"
#include <cuda_runtime.h>
#include <iomanip>

#define RANDOM 1
#define FROM_FILE 2

#define BLOCK_DIM 1024
#define COL_PER_BLK 5

#define min(a, b) ((a < b) ? a : b)
#define ceil(a, b) (a / b + (a % b != 0))

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
    __shared__ size_t m2_width;
    extern __shared__ double base_row[];
    unsigned int tid = threadIdx.x;
    unsigned int ofs = COL_PER_BLK * blockIdx.x;

    if (tid >= n)
        return;

    if (tid == 0) {
        m2_width = 2 * n;
        for (size_t i = 0; i < COL_PER_BLK; i++)
            base_row[i] = m2[(base_row_index * m2_width) + (ofs + i)];
    }
    __syncthreads();

    size_t num_cols = min(m2_width - ofs, COL_PER_BLK);

    if (tid == base_row_index) {
        normalize_self(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);
    } else
        normalize_row(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);
}

// todo: blockDim < n
__global__ void gje_scale_calc(const double *m2d, size_t n, size_t current_row, double *scale) {
    unsigned int tid = threadIdx.x;
    __shared__ double diag;
    size_t m2d_width = 2 * n;
    double base = 0;

    if (tid >= n)
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

// ** num of threads per block = COL_PER_BLOCK
__global__ void gje_set_identity(double *m2d, size_t n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + tid;
    size_t m2d_width = n * 2;

    if (idx >= n)
        return;

    m2d[idx * m2d_width + (n + idx)] = 1.0;
}

void cuda_check_err(const string &msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << msg << ":" << endl << cudaGetErrorString((cudaError_t) error) << endl;
        cudaDeviceReset();
        exit(1);
    }
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

//    double **cpu_inv = mxalloc(n, n, malloc);
//    inverse(m_h, n, cpu_inv);
//    cout << "cpu " << inverse_test(m_h, cpu_inv, n) << endl;
//    print_matrix(m_h, n, n);

    size_t m2_width = 2 * n;
    double *m2_d = nullptr, *scale_d = nullptr, *temp2_h = nullptr;;

    cudaMallocHost((void **) &temp2_h, sizeof(double) * n * m2_width);
    cuda_check_err("couldn't allocate memory in host");

    cudaMalloc((void **) &m2_d, n * m2_width * sizeof(double));
    cuda_check_err("couldn't allocate memory in device");

    for (size_t i = 0; i < n; i++) {
        cudaMemcpy(&m2_d[i * m2_width], m_h[i], n * sizeof(double), cudaMemcpyHostToDevice);
        cuda_check_err("couldn't copy data from host to device");
    }

    cudaMalloc((void **) &scale_d, n * sizeof(double));
    cuda_check_err("couldn't allocate memory in device");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gje_set_identity<<<    ceil(n, COL_PER_BLK), COL_PER_BLK>>>(m2_d, n);
    cudaDeviceSynchronize();
    cuda_check_err("error in set_identity");

    for (size_t i = 0; i < n; i++) {
        stringstream str_i;
        gje_scale_calc<<<1, BLOCK_DIM>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();
        str_i.str(string());
        str_i << "iter " << i << ") error in scale_calc";
        cuda_check_err(str_i.str());

        gje_inverse<<<ceil(m2_width, COL_PER_BLK), BLOCK_DIM, COL_PER_BLK * sizeof(double)>>>(m2_d, n, i, scale_d);
        cudaDeviceSynchronize();
        str_i.str(string());
        str_i << "iter " << i << ") error in inverse";
        cuda_check_err(str_i.str());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(temp2_h, m2_d, sizeof(double) * n * m2_width, cudaMemcpyDeviceToHost);
    cuda_check_err("couldn't copy data from device to host");
//    print_matrix(temp2_h, n, 2 * n);

    double **inv_h = mxalloc(n, n, malloc);
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(inv_h[i], &temp2_h[i * m2_width + n], sizeof(double) * n, cudaMemcpyHostToHost);
        cuda_check_err("couldn't copy data from host to host");
    }
//    print_matrix(inv_h, n, n);

    cout << "gpu :";
    cout << "\ttime: " << milliseconds/1000 << endl;
    cout << "\terror:" << inverse_test(m_h, inv_h, n) << endl;

    cudaFree(m2_d);
    cudaFree(scale_d);
    cudaFreeHost(temp2_h);
    mxfree(m_h, n, free);
    mxfree(inv_h, n, free);
    cudaDeviceReset();
    return 0;
}
