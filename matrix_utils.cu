#include "matrix_utils.cuh"
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

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

__global__ void gje_inverse(double *m2, size_t n, size_t base_row_index, double *scale) {
    size_t m2_width = 2 * n;
    extern __shared__ double base_row[];
    unsigned int tid = threadIdx.x;
    unsigned int ofs = COL_PER_BLK * blockIdx.x;

    if (tid >= n)
        return;

    if (tid == 0)
        for (size_t i = 0; i < COL_PER_BLK; i++)
            base_row[i] = m2[(base_row_index * m2_width) + (ofs + i)];
    __syncthreads();

    size_t num_cols = min(m2_width - ofs, COL_PER_BLK);
    size_t step = blockDim.x;
    while (tid < n) {

        if (tid == base_row_index)
            normalize_self(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);
        else
            normalize_row(&m2[tid * m2_width], base_row, scale[tid], num_cols, ofs);

        tid += step;
    }
}

__global__ void gje_scale_calc(const double *m2d, size_t n, size_t current_row, double *scale) {
    unsigned int tid = threadIdx.x;
    __shared__ double diag;
    size_t m2d_width = 2 * n;
    double base = 0;

    if (tid >= n)
        return;

    if (tid == 0)
        diag = m2d[current_row * m2d_width + current_row];
    __syncthreads();

    size_t step = blockDim.x;
    while (tid < n) {

        if (tid == current_row)
            scale[tid] = diag;
        else {
            base = m2d[tid * m2d_width + current_row];
            scale[tid] = base / diag;
        }
        tid += step;
    }


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

__host__ void gpu_inverse(double **matrix, size_t n, double **inverse, float *runtime) {
    size_t m2_width = 2 * n;
    double *m2_d = nullptr, *scale_d = nullptr;

    cudaMalloc((void **) &m2_d, n * m2_width * sizeof(double));
    cuda_check_err("couldn't allocate memory in device");

    for (size_t i = 0; i < n; i++) {
        cudaMemcpy(&m2_d[i * m2_width], matrix[i], n * sizeof(double), cudaMemcpyHostToDevice);
        cuda_check_err("couldn't copy data from host to device");
    }

    cudaMalloc((void **) &scale_d, n * sizeof(double));
    cuda_check_err("couldn't allocate memory in device");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gje_set_identity<<<ceil(n, COL_PER_BLK), COL_PER_BLK>>>(m2_d, n);
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
        str_i << "iter " << i << ") error in cpu_inverse";
        cuda_check_err(str_i.str());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(runtime, start, stop);

    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(inverse[i], &m2_d[i * m2_width + n], sizeof(double) * n, cudaMemcpyDeviceToHost);
        cuda_check_err("couldn't copy data from host to host");
    }

    cudaFree(m2_d);
    cudaFree(scale_d);
}

__host__ void cuda_check_err(const string &msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << msg << ":" << endl << cudaGetErrorString((cudaError_t) error) << endl;
        cudaDeviceReset();
        exit(1);
    }
}