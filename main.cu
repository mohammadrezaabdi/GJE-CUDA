#include <iostream>
#include "matrix_utils.cpp"

#define RANDOM 1
#define FROM_FILE 2

using namespace std;

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
    double **matrix = mxalloc(n,n,malloc);
    if (mode == RANDOM) {
        fill_random(n, matrix, pair<float, float>(-1e6, 1e6));
    } else {
        get_from_file(n, matrix, path);
    }
    print_matrix(matrix, n, n);

    double **inv = mxalloc(n,n,malloc);
    inverse(matrix,n,inv);
    print_matrix(inv,n,n);
    cout<< inverse_test(matrix,inv,n);

}
