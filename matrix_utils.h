#ifndef GJE_MATRIX_UTILS_H
#define GJE_MATRIX_UTILS_H

using namespace std;
typedef void* malloc_func_t (size_t);
typedef void free_func_t(void *);

void get_from_file(size_t, double **, const string &);

void fill_random(size_t, double **, pair<float, float>);

double inverse_test(double **, double **, size_t);

double** mxalloc(size_t,size_t,malloc_func_t);

double** mxcalloc(size_t,size_t,malloc_func_t);

void mxfree(double **, size_t, free_func_t);

void mxcpy(double** src,double** dst,size_t n,size_t m,size_t offset);

void print_matrix(double **, size_t, size_t);

void cpu_inverse(double **matrix, size_t n, double **inverse, float *runtime);

double norm2(double **, size_t, size_t);

void square_multiple(double **, double **, size_t, double **);

void square_subtract(double **, double **, size_t, double **);

void make_identity(size_t, double ***);

void save_to_file(size_t n, double **matrix, const string &path);

#endif //GJE_MATRIX_UTILS_H
