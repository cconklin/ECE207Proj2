#pragma once
#include "cuda_fp16.h"
#define PI_QROOT 1.331325547571923
#define MAJORITY(a, b, c) (a + b + c) & 2

__device__ __host__ float mexican_hat_point(float, float);

__global__ void mexican_hat(float *, float, float, float);

__global__ void to_float(float *, half *, int);

__device__ __host__ void
cross_correlate_point_with_wavelet(float *, float *, float *, int, int, int);

__global__ void
cross_correlate_with_wavelet(float *, float *, float *, int, int);

__global__ void
threshold(int *, float *, float);

__global__ void
edge_detect(int *, int *, int);

__global__ void
merge_leads(int *, int *, int,  int *, int, int *, int);

__global__ void
index_of_peak(int *, int *, int *);

__global__ void
nonzero(int *, int *, int);

__global__ void
moving_average(int *, int *, int, int);

__global__ void
scatter(int *, int *, int *, int *, int);

__global__ void
get_compact_rr(int *, int *, int, int);

__global__ void
clean_result(int *, int, int, int, int);

__global__ void
get_rr(int *, int *, int *, int, int, int);
