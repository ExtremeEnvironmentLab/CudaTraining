/*
   使用Verlet和动态近邻表方法计算液态氩的分子动力学
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

// 仿真参数
int N = 864;              // 氩原子数
double rho = 0.8;         // 密度
double T = 1.0;           // 温度
double L;                 // 仿真空间的边长（通过N和RHO计算）

double** r, ** v, ** a;     // 位置，速度，加速度

// 声明一部分函数
void initPositions();
void initVelocities();
void rescaleVelocities();
double instantaneousTemperature();

// 用于实现动态近邻表的各种参数
double rCutOff = 2.5;     // 力计算的截断距离
double rMax = 3.3;        // 近邻表内粒子对最大距离
int nPairs;               // 当前近邻表粒子对数量
int** pairList;           // 近邻表（存有所有近邻两个粒子的ID）
double** drPair;          // 每个对的朝向 (i,j)
double* rSqdPair;         // 每个对的距离(i,j)
int updateInterval = 10;  // 近邻表更新周期

// 声明实现近邻表的函数
void computeSeparation(int, int, double[], double&);
void updatePairList();
void updatePairSeparations();

//初始化，这里是整个程序的第一步
void initialize() {
    r = new double* [N];
    v = new double* [N];
    a = new double* [N];
    for (int i = 0; i < N; i++) {
        r[i] = new double[3];
        v[i] = new double[3];
        a[i] = new double[3];
    }
    initPositions();
    initVelocities();


    nPairs = N * (N - 1) / 2;
    pairList = new int* [nPairs];
    drPair = new double* [nPairs];
    for (int p = 0; p < nPairs; p++) {
        pairList[p] = new int[2];
        drPair[p] = new double[3];
    }
    rSqdPair = new double[nPairs];
}

//距离计算（实际上是计算距离的平方）
void computeSeparation(int i, int j, double dr[], double& rSqd) {

    // 计算周期边界条件下的距离
    rSqd = 0;
    for (int d = 0; d < 3; d++) {
        dr[d] = r[i][d] - r[j][d];//距离
        if (dr[d] >= 0.5 * L)
            dr[d] -= L;
        if (dr[d] < -0.5 * L)
            dr[d] += L;
        rSqd += dr[d] * dr[d];//距离的平方
    }
}

//更新近邻表
//是本程序时间复杂度最高的操作，达到O(N²)
void updatePairList() {

    //清空近邻表
    nPairs = 0;
    double dr[3];

    // 在最大的粒子对列表当中进行遍历，寻找其中处于有效距离内的粒子对
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++) 
        {   
            //这两层循环相当于是构建了一个上三角矩阵
            double rSqd;

            //距离计算函数
            computeSeparation(i, j, dr, rSqd);
            if (rSqd < rMax * rMax) //判别是否应该存入近邻表
            {
                //存入近邻表
                pairList[nPairs][0] = i;
                pairList[nPairs][1] = j;
                ++nPairs;
            }
        }
}

//更新近邻表内粒子的距离
void updatePairSeparations() {
    double dr[3];
    for (int p = 0; p < nPairs; p++) {
        int i = pairList[p][0];
        int j = pairList[p][1];
        double rSqd;
        computeSeparation(i, j, dr, rSqd);
        for (int d = 0; d < 3; d++)
            drPair[p][d] = dr[d];
        rSqdPair[p] = rSqd;
    }
}

//计算加速度
void computeAccelerations() {

    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            a[i][k] = 0;

    for (int p = 0; p < nPairs; p++) {
        int i = pairList[p][0];
        int j = pairList[p][1];
        if (rSqdPair[p] < rCutOff * rCutOff) {
            double r2Inv = 1 / rSqdPair[p];
            double r6Inv = r2Inv * r2Inv * r2Inv;
            double f = 24 * r2Inv * r6Inv * (2 * r6Inv - 1);
            for (int d = 0; d < 3; d++) {
                a[i][d] += f * drPair[p][d];
                a[j][d] -= f * drPair[p][d];
            }
        }
    }
}

//速度Verlet，就像龙格库塔法
//这是整个程序的主物理循环的核心
void velocityVerlet(double dt) {
    for (int i = 0; i < N; i++)//对粒子id遍历
        for (int k = 0; k < 3; k++) {//对每个坐标遍历

            //首先更新位置
            r[i][k] += v[i][k] * dt + 0.5 * a[i][k] * dt * dt;

            //使用周期边界条件
            //将超出边界的传送回边界另一端
            if (r[i][k] < 0)
                r[i][k] += L;
            if (r[i][k] >= L)
                r[i][k] -= L;

            //速度更新的第一步
            //在加速度更新前，先加上上一次迭代求得的加速度的一半
            v[i][k] += 0.5 * a[i][k] * dt;
            //这是Verlet算法的标准操作
        }

    //计算近邻表内的粒子距离
    updatePairSeparations();
    //根据距离计算本次迭代的加速度
    computeAccelerations();
    //对于单个粒子来说，与其有互动的粒子只有距离比较近的十几个
    //所以近邻表内粒子距离更新和加速度更新时间复杂度是O（N）

    //在完成加速度更新以后，进行速度更新的第二步
    for (int i = 0; i < N; i++)//对粒子id遍历
        for (int k = 0; k < 3; k++)//对每个坐标遍历
            //同样是只加上加速度的一半，但是取的是本次迭代的加速度值
            v[i][k] += 0.5 * a[i][k] * dt;
}

//主函数与主循环
int main() {
    initialize();
    updatePairList();
    updatePairSeparations();
    computeAccelerations();
    double dt = 0.01;
    ofstream file("T3.data");
    for (int i = 0; i < 1000; i++) {
        //主物理运算Verlet
        velocityVerlet(dt);
        file << instantaneousTemperature() << '\n';
        if (i % 200 == 0)
            rescaleVelocities();
        if (i % updateInterval == 0) {

            //每过特定步数更新一次近邻表，默认是10步
            updatePairList();
            //更新近邻表是本程序时间复杂度最高的操作，达到O(N²)

            updatePairSeparations();
        }
    }
    file.close();
    return(0);
}

//位置初始化函数
void initPositions() {

    // 通过总粒子数N和粒子数密度RHO计算正方体边长
    L = pow(N / rho, 1.0 / 3);

    // 找到一个足够大的晶胞数，M立方，以容纳全部N个原子
    int M = 1;
    while (4 * M * M * M < N)//一个面心立方晶胞的原子数是4
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
                        r[n][0] = (x + xCell[k]) * a;
                        r[n][1] = (y + yCell[k]) * a;
                        r[n][2] = (z + zCell[k]) * a;
                        ++n;
                    }
}

//高斯随机数生成函数
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

//速度初始化函数
void initVelocities() {

    // 单位标准差的高斯分布
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] = gasdev();
    // 将平均速度置零
    double vCM[3] = { 0, 0, 0 };//velocity center of mass
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vCM[i] += v[n][i];
    for (int i = 0; i < 3; i++)
        vCM[i] /= N;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] -= vCM[i];

    // 重新等比例缩放所有粒子的速度以达到想要的温度
    rescaleVelocities();
}

//根据目标温度重新缩放速度（由于仿真过程精度损失温度总是会漂变）
void rescaleVelocities() {
    double vSqdSum = 0;
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            vSqdSum += v[n][i] * v[n][i];
    double lambda = sqrt(3 * (N - 1) * T / vSqdSum);
    for (int n = 0; n < N; n++)
        for (int i = 0; i < 3; i++)
            v[n][i] *= lambda;
}

//即时温度测量函数
double instantaneousTemperature() {
    double sum = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            sum += v[i][k] * v[i][k];
    return sum / (3 * (N - 1));
}
