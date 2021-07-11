#ifndef GJE_MATRIX_UTILS_CUH
#define GJE_MATRIX_UTILS_CUH

#include <string>

#define BLOCK_DIM (1<<10)
#define COL_PER_BLK 5

#define min(a, b) ((a < b) ? a : b)
#define ceil(a, b) (a / b + (a % b != 0))

using namespace std;

__device__ void normalize_row(double *target_row, const double *base_row, double scale, size_t n, size_t offset);

__device__ void
normalize_self(double *self, double const *self_but_in_share_memory, double scale, size_t n, size_t offset);

__global__ void gje_inverse(double *m2, size_t n, size_t base_row_index, double *scale);

__global__ void gje_scale_calc(const double *m2d, size_t n, size_t current_row, double *scale);

__global__ void gje_set_identity(double *m2d, size_t n);

__host__ void gpu_inverse(double **matrix, size_t n, double **inverse, float *runtime);

void cuda_check_err(const string &msg);

#endif
