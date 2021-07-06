#include "matrix_utils.h"

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <cstring>

#define MARGIN 1e-8
#define IS_ZERO(x) (abs(x)<=MARGIN)
#define NOT_ZERO(x) (abs(x)>MARGIN)

typedef void* malloc_func_t (size_t);
typedef void free_func_t(void *);

inline void normalize_row(const double *base_row, double *target_row, size_t n, double scale) {
    for (size_t i = 0; i < n; i++) {
        target_row[i] -= base_row[i] * scale;
    }
}

void inverse(double **matrix, size_t n, double **inverse) {
    // Create the augmented matrix
    double **m = mxalloc(n, 2 * n, malloc);
    mxcpy(matrix, m, n, n, 0);

    // Add the identity matrix
    for (int i = 0; i < n; i++) {
        memset(&m[i][n], 0, n);
        m[i][n + i] = 1;
    }

    // generating the inverse matrix
    for (size_t i = 0; i < n; i++) {
        if (IS_ZERO(m[i][i]))//m[i][i] ==0
            for (size_t k = i + 1; k < n; k++)
                if (NOT_ZERO(m[k][i]))
                    swap(m[k], m[i]);

        for (size_t k = 0; k < n; k++)
            if (i != k)
                normalize_row(m[i], m[k], 2 * n, m[k][i] / m[i][i]);

        // normalizing current row
        double scale = m[i][i];
        for (size_t j = 0; j < 2 * n; j++)
            m[i][j] /= scale;
    }

    // copy the inverse matrix (augmented) to the result array
    mxcpy(m, inverse, n, n, n);
    mxfree(m, n, free);
}

double norm2(double **matrix, size_t n, size_t m) {
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            sum += sqrt(pow(matrix[i][j], 2));
    return sum;
}

double inverse_test(double **matrix, double **inverse, size_t n) {
    double **I, **tmp = mxcalloc(n, n, malloc);
    make_identity(n, &I);
    square_multiple(matrix, inverse, n, tmp);
    square_subtract(I, tmp, n, tmp);
    double res = norm2(tmp, n, n);
    mxfree(I, n, free);
    mxfree(tmp, n, free);
    return res;
}

double **mxcalloc(size_t n, size_t m, malloc_func_t malloc_f) {
    double **matrix = mxalloc(n, m, malloc_f);
    for (size_t i = 0; i < n; i++)
        memset(matrix[i], 0, m * sizeof(double));
    return matrix;
}

void square_multiple(double **m1, double **m2, size_t n, double **result) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++)
                result[i][j] += m1[i][k] * m2[k][j];
}

void square_subtract(double **m1, double **m2, size_t n, double **result) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            result[i][j] = m1[i][j] - m2[i][j];
}

void make_identity(size_t n, double ***result) {
    (*result) = mxcalloc(n, n, malloc);
    for (size_t i = 0; i < n; i++)
        (*result)[i][i] = 1.0;
}

void get_from_file(size_t n, double **matrix, const string &path) {
    ifstream fin(path);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            fin >> matrix[i][j];
            if (!fin)goto done;
        }
    }
    done:
    fin.close();
}

void fill_random(size_t n, double **matrix, pair<float, float> range) {
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> uni(range.first, range.second);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            matrix[i][j] = uni(eng);
}

double **mxalloc(size_t n, size_t m, malloc_func_t malloc_f) {
    double **matrix = (double **) malloc_f(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++)
        matrix[i] = (double *) malloc_f(sizeof(double) * m);
    return matrix;
}

void mxfree(double **matrix_host, size_t rows_count, free_func_t free_f) {
    for (size_t i = 0; i < rows_count; i++)
        free_f(matrix_host[i]);
    free_f(matrix_host);
}

void mxcpy(double **src, double **dst, size_t n, size_t m, size_t offset) {
    for (size_t i = 0; i < n; i++)
        memcpy(dst[i], src[i] + offset, sizeof(double) * m);
}

void print_matrix(double **ar, size_t n, size_t m) {
    cerr << "printing matrix:\n";
    for (size_t i = 0; i < n; i++, cerr << endl)
        for (size_t j = 0; j < m; j++)
            cerr << setprecision(2) << ar[i][j] << "\t";
}


