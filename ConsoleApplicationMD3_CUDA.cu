/*
   使用Verlet和动态近邻表方法计算液态氩的分子动力学 - CUDA加速版本
   这个版本用的是笨办法：在设备端和host端反复拷贝粒子的信息，没有用CUDA的共享内存
   或者流式显存分配，效率并不是很高。
   
   实际上最需要通过CUDA并行化的就是更新近邻表，因为它的时间复杂度最高，
   每10次更新都需要重新计算，所以效率很低。
   下一步就是优化它。

   现在起码能用了，可以叫它V-0.1版本

   如需执行，请在终端输入如下两行命令：
   nvcc -arch=sm_61 ConsoleApplicationMD3_CUDA.cu -o ConsoleApplicationMD3_CUDA
   ./ConsoleApplicationMD3_CUDA
   其中-arch=sm_61是GPU架构，这里对应的是1050显卡，可自行修改

   难受的是，我现在还没搞明白为什么VS内置的CUDA编译选项要求文件中不能出现中文注释，
   所以说暂时不能直接一键编译。
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 仿真参数
int N = 864;              // 氩原子数
double rho = 0.8;         // 密度
double T = 1.0;           // 温度
double L;                 // 仿真空间的边长（通过N和RHO计算）

// 主机端数据
double** h_r, ** h_v, ** h_a;     // 位置，速度，加速度（主机端）

// 设备端数据
double *d_r, *d_v, *d_a;          // 位置，速度，加速度（设备端）

// 用于实现动态近邻表的各种参数
double rCutOff = 2.5;     // 力计算的截断距离
double rMax = 3.3;        // 近邻表内粒子对最大距离
int nPairs;               // 当前近邻表粒子对数量
int** h_pairList;         // 近邻表（主机端）
double** h_drPair;        // 每个对的朝向 (i,j)（主机端）
double* h_rSqdPair;       // 每个对的距离(i,j)（主机端）

// 设备端近邻表数据
int* d_pairList;          // 近邻表（设备端）
double* d_drPair;         // 每个对的朝向 (i,j)（设备端）
double* d_rSqdPair;       // 每个对的距离(i,j)（设备端）

int updateInterval = 10;  // 近邻表更新周期

// 错误检查宏
#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 声明CPU函数
void initPositions();
void initVelocities();
void rescaleVelocities();
double instantaneousTemperature();
void computeSeparation(int, int, double[], double&);
void updatePairList();
void updatePairSeparations();
void allocateMemory();
void freeMemory();
void copyDataToDevice();
void copyDataFromDevice();

// 声明CUDA核函数
__global__ void computeAccelerationsKernel(double* d_r, double* d_a, int* d_pairList, 
                                          double* d_drPair, double* d_rSqdPair, 
                                          int nPairs, double rCutOff, int N);
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N);
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N);

// 初始化，这里是整个程序的第一步
void initialize() {
    allocateMemory();
    initPositions();
    initVelocities();
}

// 分配内存
void allocateMemory() {
    // 主机内存分配
    h_r = new double* [N];
    h_v = new double* [N];
    h_a = new double* [N];
    for (int i = 0; i < N; i++) {
        h_r[i] = new double[3];
        h_v[i] = new double[3];
        h_a[i] = new double[3];
    }

    // 设备内存分配
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_r, N * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_v, N * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_a, N * 3 * sizeof(double)));

    // 近邻表内存分配
    int maxPairs = N * (N - 1) / 2;
    h_pairList = new int* [maxPairs];
    h_drPair = new double* [maxPairs];
    for (int p = 0; p < maxPairs; p++) {
        h_pairList[p] = new int[2];
        h_drPair[p] = new double[3];
    }
    h_rSqdPair = new double[maxPairs];

    // 设备端近邻表内存分配 - 将在updatePairList后根据实际nPairs分配
}

// 释放内存
void freeMemory() {
    // 释放主机内存
    for (int i = 0; i < N; i++) {
        delete[] h_r[i];
        delete[] h_v[i];
        delete[] h_a[i];
    }
    delete[] h_r;
    delete[] h_v;
    delete[] h_a;

    int maxPairs = N * (N - 1) / 2;
    for (int p = 0; p < maxPairs; p++) {
        delete[] h_pairList[p];
        delete[] h_drPair[p];
    }
    delete[] h_pairList;
    delete[] h_drPair;
    delete[] h_rSqdPair;

    // 释放设备内存
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_pairList);
    cudaFree(d_drPair);
    cudaFree(d_rSqdPair);
}

// 将数据从主机复制到设备
void copyDataToDevice() {
    // 将位置、速度和加速度数据各自转换为一维数组，形式为：粒子1号的rx,ry,rz，粒子2号的rx,ry,rz，...
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_r_flat[i * 3 + d] = h_r[i][d];
            h_v_flat[i * 3 + d] = h_v[i][d];
            h_a_flat[i * 3 + d] = h_a[i][d];
        }
    }

    // 复制到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_r, h_r_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_v, h_v_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_a, h_a_flat, N * 3 * sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_r_flat;
    delete[] h_v_flat;
    delete[] h_a_flat;
}

// 将数据从设备复制回主机
void copyDataFromDevice() {
    // 分配临时一维数组
    double* h_r_flat = new double[N * 3];
    double* h_v_flat = new double[N * 3];
    double* h_a_flat = new double[N * 3];

    // 从设备复制数据
    CUDA_CHECK_ERROR(cudaMemcpy(h_r_flat, d_r, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_v_flat, d_v, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_a_flat, d_a, N * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    // 转换回二维数组格式
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            h_r[i][d] = h_r_flat[i * 3 + d];
            h_v[i][d] = h_v_flat[i * 3 + d];
            h_a[i][d] = h_a_flat[i * 3 + d];
        }
    }

    delete[] h_r_flat;
    delete[] h_v_flat;
    delete[] h_a_flat;
}

// 距离计算
void computeSeparation(int i, int j, double dr[], double& rSqd) {
    // 计算周期边界条件下的距离
    rSqd = 0;
    for (int d = 0; d < 3; d++) {
        dr[d] = h_r[i][d] - h_r[j][d]; // 距离
        if (dr[d] >= 0.5 * L)
            dr[d] -= L;
        if (dr[d] < -0.5 * L)
            dr[d] += L;
        rSqd += dr[d] * dr[d]; // 距离的平方
    }
}

// 更新近邻表
// 这实际上是本程序唯一一个计算时间需求以N²规律剧烈增长的部分
// TODO：将其放入CUDA当中计算
void updatePairList() {
    nPairs = 0;
    double dr[3];

    // 在最大的粒子对列表当中进行遍历，寻找其中处于有效距离内的粒子对
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++) {
            double rSqd;
            computeSeparation(i, j, dr, rSqd);
            if (rSqd < rMax * rMax) {
                h_pairList[nPairs][0] = i;
                h_pairList[nPairs][1] = j;
                ++nPairs;
            }
        }

    // 为设备端近邻表分配内存
    if (d_pairList != nullptr) cudaFree(d_pairList);
    if (d_drPair != nullptr) cudaFree(d_drPair);
    if (d_rSqdPair != nullptr) cudaFree(d_rSqdPair);

    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pairList, nPairs * 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_drPair, nPairs * 3 * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_rSqdPair, nPairs * sizeof(double)));

    // 将近邻表数据转换为一维数组
    int* h_pairList_flat = new int[nPairs * 2];
    for (int p = 0; p < nPairs; p++) {
        h_pairList_flat[p * 2] = h_pairList[p][0];
        h_pairList_flat[p * 2 + 1] = h_pairList[p][1];
    }

    // 复制近邻表到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_pairList, h_pairList_flat, nPairs * 2 * sizeof(int), cudaMemcpyHostToDevice));
    
    delete[] h_pairList_flat;
}

// 更新近邻表内粒子的距离
void updatePairSeparations() {
    double dr[3];
    for (int p = 0; p < nPairs; p++) {
        int i = h_pairList[p][0];
        int j = h_pairList[p][1];
        double rSqd;
        computeSeparation(i, j, dr, rSqd);
        for (int d = 0; d < 3; d++)
            h_drPair[p][d] = dr[d];
        h_rSqdPair[p] = rSqd;
    }

    // 将更新后的距离数据转换为一维数组
    double* h_drPair_flat = new double[nPairs * 3];
    for (int p = 0; p < nPairs; p++) {
        for (int d = 0; d < 3; d++) {
            h_drPair_flat[p * 3 + d] = h_drPair[p][d];
        }
    }

    // 复制到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_drPair, h_drPair_flat, nPairs * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_rSqdPair, h_rSqdPair, nPairs * sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_drPair_flat;
}

// CUDA核函数：计算加速度
__global__ void computeAccelerationsKernel(double* d_r, double* d_a, int* d_pairList, 
                                          double* d_drPair, double* d_rSqdPair, 
                                          int nPairs, double rCutOff, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化加速度为0
    if (idx < N) {
        d_a[idx * 3] = 0.0;
        d_a[idx * 3 + 1] = 0.0;
        d_a[idx * 3 + 2] = 0.0;
    }
    __syncthreads();

    // 计算粒子对之间的力
    if (idx < nPairs) {
        int i = d_pairList[idx * 2];
        int j = d_pairList[idx * 2 + 1];
        double rSqd = d_rSqdPair[idx];

        if (rSqd < rCutOff * rCutOff) {
            double r2Inv = 1.0 / rSqd;
            double r6Inv = r2Inv * r2Inv * r2Inv;
            double f = 24.0 * r2Inv * r6Inv * (2.0 * r6Inv - 1.0);

            // 使用原子操作更新加速度（使用CAS循环实现double类型的原子加法）
            for (int d = 0; d < 3; d++) {
                double force = f * d_drPair[idx * 3 + d];
                // 对粒子i的加速度进行原子加法
                unsigned long long int* address_as_ull = (unsigned long long int*)&d_a[i * 3 + d];
                unsigned long long int old = *address_as_ull;
                unsigned long long int assumed;
                do {
                    assumed = old;
                    old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(force + __longlong_as_double(assumed)));
                } while (assumed != old);
                
                // 对粒子j的加速度进行原子加法
                address_as_ull = (unsigned long long int*)&d_a[j * 3 + d];
                old = *address_as_ull;
                do {
                    assumed = old;
                    old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(-force + __longlong_as_double(assumed)));
                } while (assumed != old);
            }
        }
    }
}

// CUDA核函数：Verlet第一步（用上一次迭代的数据更新位置，完成速度更新的一半）
__global__ void velocityVerletStep1Kernel(double* d_r, double* d_v, double* d_a, 
                                         double dt, double L, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {

            // 首先用上一次迭代的速度和加速度数据更新位置
            d_r[idx * 3 + d] += d_v[idx * 3 + d] * dt + 0.5 * d_a[idx * 3 + d] * dt * dt;
            
            // 应用周期边界条件，将超出边界的粒子传送到另一侧
            if (d_r[idx * 3 + d] < 0)
                d_r[idx * 3 + d] += L;
            if (d_r[idx * 3 + d] >= L)
                d_r[idx * 3 + d] -= L;
            
            // 根据上一个迭代的加速度数据的一半，来更新速度
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// CUDA核函数：Verlet第二步（根据本次迭代的加速度完成速度更新的另一半）
__global__ void velocityVerletStep2Kernel(double* d_v, double* d_a, double dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        for (int d = 0; d < 3; d++) {
            // 速度更新的另一半
            d_v[idx * 3 + d] += 0.5 * d_a[idx * 3 + d] * dt;
        }
    }
}

// 使用CUDA加速的Verlet积分
void velocityVerlet(double dt) {
    // 设置CUDA线程块和网格大小
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;

    // 第一步：更新位置和速度的一半
    velocityVerletStep1Kernel<<<numBlocks, blockSize>>>(d_r, d_v, d_a, dt, L, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 更新粒子对距离（需要从设备复制数据到主机）
    copyDataFromDevice();
    updatePairSeparations();

    // 计算新的加速度
    computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, rCutOff, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 第二步：更新速度的另一半
    velocityVerletStep2Kernel<<<numBlocks, blockSize>>>(d_v, d_a, dt, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// 位置初始化函数
void initPositions() {
    // 通过总粒子数N和粒子数密度RHO计算正方体边长
    L = pow(N / rho, 1.0 / 3);

    // 找到一个足够大的晶胞数，M立方，以容纳全部N个原子
    int M = 1;
    while (4 * M * M * M < N) // 一个面心立方晶胞的原子数是4
        ++M;
    double a = L / M;           // 计算得到每个晶胞的边长a

    // 晶胞中 4 个原子的位置
    double xCell[4] = { 0.25, 0.75, 0.75, 0.25 };
    double yCell[4] = { 0.25, 0.75, 0.25, 0.75 };
    double zCell[4] = { 0.25, 0.25, 0.75, 0.75 };

    int n = 0;                  // 放置全部的N个原子
    for (int x = 0; x < M; x++)
        for (int y = 0; y < M; y++)
            for (int z = 0; z < M; z++)
                for (int k = 0; k < 4; k++)
                    if (n < N) {
                        h_r[n][0] = (x + xCell[k]) * a;
                        h_r[n][1] = (y + yCell[k]) * a;
                        h_r[n][2] = (z + zCell[k]) * a;
                        ++n;
                    }
}

// 高斯随机数生成函数
double gasdev() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;
        return v2 * fac;
    }
    else {
        available = false;
        return gset;
    }
}

// 速度初始化函数
void initVelocities() {
    // 单位标准差的高斯分布
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] = gasdev();
    
    // 将平均速度置零
    double vCM[3] = { 0, 0, 0 };
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vCM[i] += h_v[n][i];
    for (int i = 0; i < 3; i++)
        vCM[i] /= N;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] -= vCM[i];

    // 重新等比例缩放所有粒子的速度以达到想要的温度
    rescaleVelocities();
}

// 根据目标温度重新设置速度
void rescaleVelocities() {
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += h_v[n][i] * h_v[n][i];
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            h_v[n][i] *= lambda;
    
    // 更新设备端速度数据
    copyDataToDevice();
}

// 即时温度测量函数
double instantaneousTemperature() {
    // 从设备复制速度数据到主机
    copyDataFromDevice();
    
    double sum = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            sum += h_v[i][k] * h_v[i][k];
    return sum / (3 * (N - 1));
}

// 主函数与主循环
int main() {
    // 初始化CUDA设备
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "没有找到支持CUDA的设备！" << endl;
        return -1;
    }
    
    // 选择第一个CUDA设备
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    
    // 初始化指针为nullptr
    d_r = d_v = d_a = nullptr;
    d_pairList = nullptr;
    d_drPair = nullptr;
    d_rSqdPair = nullptr;
    
    // 初始化模拟
    initialize();
    
    // 将数据复制到设备
    copyDataToDevice();
    
    // 更新近邻表
    updatePairList();
    updatePairSeparations();
    
    // 设置CUDA线程块和网格大小
    int blockSize = 256;
    int numBlocksPairs = (nPairs + blockSize - 1) / blockSize;
    
    // 计算初始加速度
    computeAccelerationsKernel<<<numBlocksPairs, blockSize>>>(d_r, d_a, d_pairList, d_drPair, d_rSqdPair, nPairs, rCutOff, N);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    double dt = 0.01;
    ofstream file("T3_CUDA.data");
    
    // 主循环
    for (int i = 0; i < 1000; i++) {

        // 主物理运算，包含三个设备端函数
        velocityVerlet(dt);

        // 在文本文件中写入温度
        file << instantaneousTemperature() << '\n';
        
        if (i % 200 == 0)
            rescaleVelocities();
        
        if (i % updateInterval == 0) {// 每10步更新一次近邻表
			updatePairList();// 这是一个O（n^2)操作，应当放入CUDA当中，但是还没做
            updatePairSeparations();
        }
    }
    
    file.close();
    
    // 释放内存
    freeMemory();
    
    return 0;
} 