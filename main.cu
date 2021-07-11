#include "matrix_utils.cpp"
#include "matrix_utils.cu"
#include <iostream>
#include <cuda_runtime.h>

#define RANDOM 1
#define FROM_FILE 2
#define WITH_CPU 1
#define WITH_GPU 2

int main(int argc, char **argv) {
    size_t n = 0;
    int get_mode = RANDOM;
    int exec_mode = WITH_GPU;
    string in_path;
    string out_path;

    // arg parsing
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                case 'n':
                    n = strtoul(argv[++i], nullptr, 0);
                    break;
                case 'r':
                    get_mode = RANDOM;
                    break;
                case 'f':
                    get_mode = FROM_FILE;
                    in_path = argv[++i];
                    break;
                case 'g':
                    exec_mode = WITH_GPU;
                    break;
                case 'c':
                    exec_mode = WITH_CPU;
                    break;
                case 'o':
                    out_path = argv[++i];
                    break;
                default:
                    cout << "invalid arguments";
            }
        }
    }
    double **matrix = mxalloc(n, n, malloc);
    double **inverse = mxalloc(n, n, malloc);
    float runtime = 0.0;

    if (get_mode == RANDOM)
        fill_random(n, matrix, pair<float, float>(-1e6, 1e6));
    else
        get_from_file(n, matrix, in_path);

    if (exec_mode == WITH_GPU)
        gpu_inverse(matrix, n, inverse, &runtime);
    else
        cpu_inverse(matrix, n, inverse, &runtime);

    if (!out_path.empty())
        save_to_file(n, inverse, out_path);

    cout << "time: " << runtime << "(ms)" << endl << "err: " << inverse_test(matrix, inverse, n) << endl;

    mxfree(matrix, n, free);
    mxfree(inverse, n, free);
    cudaDeviceReset();
    return 0;
}
