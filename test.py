import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from numpy import linalg as lng
from subprocess import Popen, PIPE
import os

num_of_tests = 15
sample_range = 1e6


def main():
    Popen('make').wait()
    if not os.path.exists('/tests'):
        os.makedirs('/tests')
    gpu_norms = []
    gpu_runtimes = []
    cpu_norms = []
    cpu_runtimes = []
    samples = [2 ** i for i in range(1, num_of_tests + 1)]
    for i, n in enumerate(samples, start=1):
        input_path = f'tests/in{i}.txt'
        output_cpu_path = f'tests/out{i}_cpu.txt'
        output_gpu_path = f'tests/out{i}_gpu.txt'
        arr = np.random.uniform(low=-sample_range, high=sample_range, size=(n, n))
        np.savetxt(input_path, arr, delimiter=' ')

        print(f'cpu test{i}\tstarted (n={n}).')
        p1 = Popen(['./GJE', '-c', '-n', str(n), '-f', str(input_path), '-o', str(output_cpu_path)],
                   stdout=PIPE, stderr=PIPE)
        stdout, stderr = p1.communicate()
        stdout_str, stderr_str = (stdout.decode("utf-8"), stderr.decode("utf-8"))
        print(stdout_str)
        print(stderr_str, file=sys.stderr)
        cpu_runtimes.append(float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stdout_str)[0]))
        print(f'cpu test{i}\tfinished.')

        print(f'gpu test{i}\tstarted (n={n}).')
        p2 = Popen(['./GJE', '-g', '-n', str(n), '-f', str(input_path), '-o', str(output_gpu_path)],
                   stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        stdout_str, stderr_str = (stdout.decode("utf-8"), stderr.decode("utf-8"))
        print(stdout_str)
        print(stderr_str, file=sys.stderr)
        cpu_runtimes.append(float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stdout_str)[0]))
        print(f'gpu test{i}\tfinished.')

        cpu_inv = np.array([[float(i) for i in line.split()] for line in open(output_cpu_path)])
        gpu_inv = np.array([[float(i) for i in line.split()] for line in open(output_gpu_path)])
        cpu_norms.append(lng.norm(np.matmul(arr, cpu_inv) - np.identity(n)))
        gpu_norms.append(lng.norm(np.matmul(arr, gpu_inv) - np.identity(n)))

    plt.figure(figsize=(10, 7.5))
    plt.plot(cpu_runtimes)
    plt.plot(gpu_runtimes)
    plt.legend(['cpu', 'gpu'], loc='upper left')
    plt.title('runtime plot')
    plt.xlabel('n')
    plt.ylabel('ms')
    plt.show()

    plt.figure(figsize=(10, 7.5))
    plt.plot(cpu_norms)
    plt.plot(gpu_norms)
    plt.legend(['cpu', 'gpu'], loc='upper left')
    plt.title('precision plot')
    plt.xlabel('n')
    plt.ylabel('norm2')
    plt.show()


if __name__ == '__main__':
    main()
