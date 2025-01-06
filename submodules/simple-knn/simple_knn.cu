/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cfloat>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// CUDA内核函数和辅助函数，用于计算点云数据的K近邻（K-Nearest Neighbors, KNN）
namespace cg = cooperative_groups;
// CustomMin和CustomMax结构体定义了设备内联函数，用于计算两个float3向量的逐元素最小值和最大值。
struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};
struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};
// prepMorton函数用于将输入转化为Morton编码（将多维数据转化为一维数据的编码，常用于空间排序）
__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}
// coord2Morton函数将3D坐标转换为Morton编码，用于空间排序
__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	//  (coord.x - minn.x) / (maxx.x - minn.x)将分量归一化到[0, 1]范围。((1 << 10) - 1)将归一化的x分量缩放到[0, 1023]范围（10位Morton编码）
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}
// coord2Morton函数的CUDA内核函数版本，用于将将点云数据中的每个点换为Morton编码，并存储在 codes 数组中
__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}
// MinMax 结构体用于存储点云数据的最小和最大边界
struct MinMax
{
	float3 minn;
	float3 maxx;
};
// boxMinMax计算每个块（block）中点的最小和最大边界，并存储在boxes数组中
__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
// uint32_t P：点的数量。float3* points：点的坐标数组。uint32_t* indices：点的索引数组。MinMax* boxes：用于存储每个块的最小和最大边界的数组。
{
	auto idx = cg::this_grid().thread_rank();// 获取当前线程在整个网格中的索引
	// 初始化MinMax结构体me：如果当前线程的索引小于点的数量P，则将me.minn和me.maxx初始化为对应点的坐标。否则，将me.minn 初始化为最大浮点数，将me.maxx初始化为最小浮点数
	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}
	// 定义一个共享内存数组redResult，用于更新存储每个线程的MinMax结果
	__shared__ MinMax redResult[BOX_SIZE];
	// 计算每个块中点的最小和最大边界
	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)//循环变量off从BOX_SIZE/2开始，每次减半，直到off等于1
	{
		// 在每次循环中，前2 * off个线程将me的值写入共享内存redResult
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();//同步所有线程，确保共享内存中的数据已更新
		// 前off个线程从共享内存中读取相邻线程的MinMax结果，并更新自己的me值
		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}
// boxMeanDist计算每个点到其K近邻的平均距离，并存储在dists数组中
__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
// updateKBest用于更新K近邻的距离数组knn
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
// const float3& ref：参考点的坐标。const float3& point：当前点的坐标。float* knn：存储K近邻距离的数组。
{
	// 计算当前点与参考点之间的欧氏距离平方dist
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	// 遍历K近邻数组knn，如果当前距离小于数组中的某个值，则更新数组，并将较大的值移到后面
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}
// boxMeanDist用于计算每个点到其K近邻的平均距离，并存储在dists数组中
__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
// uint32_t P：点的数量。float3* points：点的坐标数组。uint32_t* indices：点的索引数组。MinMax* boxes：用于存储每个块的最小和最大边界的数组。float* dists：用于存储每个点到其K近邻的平均距离的数组。
{
	// 获取线程索引
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	// 初始化变量
	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
	// 遍历当前点的邻域（前后各3个点），更新K近邻的距离数组best
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}
	// 计算拒绝距离reject，即当前点到其第3近邻的距离。重置best数组
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;
	// 遍历所有块，计算当前点到每个块的距离 dist。如果距离大于拒绝距离reject或大于当前的第3近邻距离，则跳过该块。否则，遍历块中的所有点，更新K近邻的距离数组best。
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;//计算当前点到其 K 近邻的平均距离，并存储在 dists 数组中
}
// SimpleKNN类的knn方法，用于计算点云数据的K近邻（K-Nearest Neighbors, KNN），使用cub库和thrust库进行并行计算和排序。计算每个点的Morton编码，并对其进行排序。计算每个块的最小和最大边界。计算每个点到其K近邻的平均距离，并存储在meanDists数组中
void SimpleKNN::knn(int P, float3* points, float* meanDists)
// int P：点的数量。float3* points：点的坐标数组。float* meanDists：用于存储每个点到其 K 近邻的平均距离的数组。
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;
	// 计算点云数据的最小边界minn和最大边界maxx
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);
	// 声明morton和morton_sorted，用于存储Morton编码及其排序结果。调用coord2Morton内核函数计算每个点的Morton编码，并存储在 morton 数组中
	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());
	// 排序Morton编码和索引：声明indices和indices_sorted，用于存储点的索引及其排序结果。调用CUB库的DeviceRadixSort::SortPairs函数对Morton编码和索引进行排序。传入nullptr以获取所需的临时内存大小，并将其存储在temp_storage_bytes中。调整temp_storage的大小以适应所需的临时内存。再次调用SortPairs函数进行排序。
	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	// 计算每个块的最小和最大边界：声明boxes和meanDists，用于存储每个块的最小和最大边界以及每个点到其K近邻的平均距离。调用boxMinMax和boxMeanDist内核函数计算每个块的最小和最大边界以及每个点到其K近邻的平均距离。
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}