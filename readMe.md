
# Gauss-Jordan Elimination In CUDA

Gauss-Jordan Elimination is a way of calculating the inverse of a matrix and solving many linear systems. this is an Implement matrix inversion using Gauss-Jordan Elimination in CUDA.

## build

first you should build the project with cmake

```bash
cmake CMakeLists.txt
```

after a Makefile built, compile the project by following command:

```bash
make
```

## run

after building the project, you should execute `./GJE` command with following flags:

```bash
./GJE -n <edge_length> [-f <input_matrix_file> | -r <random_uniform_matrix>] -o <calculated_inverse_matrix_path> [-c <execute_on_cpu> | -g <execute_on_gpu>]
```

the program writes the calculation runtime in on stdout (in milliseconds). for example:

```out
calculation time: 120.31(ms)
```

## methodology
** important --> 1024 >= block size > COL_PER_BLK

## benchmark

you can run the python benchmark program by following command:

```bash
sudo python test.py 
```

on this benchmark, we've executed the program with random matrix with exponential binary lengths from 2^1 till 2^16.
after each execution on both cpu & gpu, we capture its computation runtime and error.
we calculate matrix error by norm2(Frobenius) method. assume that we've calculated the inverse of matrix A, then we have:

$error = norm2(I-A*A^{-1})$

we've executed the benchmark on Nvidia RTX2080 & intel i9900k.
