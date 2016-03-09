#include "cuda_fp16.h"
#define PI_QROOT 1.331325547571923

__device__ __host__ float mexican_hat_point(float sigma, float t) {
  float sigma_sq = sigma * sigma;
  float t_sq = t * t;
  float term1 = 2.0 / (PI_QROOT * sqrt(3.0 * sigma));
  float term2 = 1.0 - (t_sq / sigma_sq);
  float term3 = expf(-1.0 * (t_sq/(2.0 * sigma_sq)));
  return term1 * term2 * term3;
}

__global__ void mexican_hat(float * out_signal, float sigma, float t_start, float t_step) {
  int sample_number = threadIdx.x + blockIdx.x * blockDim.x;
  float t = t_start + ((float) sample_number) * t_step;
  out_signal[sample_number] = mexican_hat_point(sigma, t);
}

__global__ void to_float(float * out, half * in, int length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < length) {
    out[idx] = __half2float(in[idx]);
  }
}

__device__ __host__ void
cross_correlate_point_with_wavelet(
  float * result_point,
  float * signal,
  float * wavelet,
  int point,
  int signal_size,
  int points_per_wavelet)
{
  float result = 0.0;
  for (int i = -points_per_wavelet / 2; i < points_per_wavelet / 2; i++) {
    if (point + i >= signal_size) {
      break;
    } else if (point + i < 0) {
      continue;
    }
    result += signal[point + i] * wavelet[i + points_per_wavelet / 2];
  }
  * result_point = result;
}

__global__ void
cross_correlate_with_wavelet(
  float * out_signal,
  float * in_signal,
  float * wavelet,
  int signal_size,
  int points_per_wavelet)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Could dynamically allocate, but we won't use wavelets larger than this
  __shared__ float sh_wavelet[8];
  if (threadIdx.x < points_per_wavelet) {
    sh_wavelet[threadIdx.x] = wavelet[threadIdx.x];
  }
  __syncthreads();
  cross_correlate_point_with_wavelet(&out_signal[idx],
                                     in_signal,
                                     sh_wavelet,
                                     idx,
                                     signal_size,
                                     points_per_wavelet);
}

__global__ void
threshold(
  int * out_signal,
  float * in_signal,
  float threshold)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  out_signal[idx] = in_signal[idx] >= threshold;
}

__global__ void
edge_detect(
  int * out_signal,
  int * in_signal,
  int num_points)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == (num_points - 1)) {
    out_signal[idx] = 0;
  }
  out_signal[idx] = (!in_signal[idx] && in_signal[idx+1]);
}

#define MAJORITY(a, b, c) (a + b + c) & 2

__global__ void
merge_leads(
  int * merged, // Output
  int * in1,    // Thresholded input signal
  int offset1,  // Synchronization offset
  int * in2,
  int offset2,
  int * in3,
  int offset3)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int val1 = in1[tid + offset1];
  int val2 = in2[tid + offset2];
  int val3 = in3[tid + offset3];
  merged[tid] = MAJORITY(val1, val2, val3);
}

__device__ int
int_3_median_index(
  int * in_signal,
  int index)
{
  // Cardinality (3 is largest) -> (compare1, compare2, compare3) [position]
  // 1 2 3 -> 1 1 1 [1] 01
  // 1 3 2 -> 1 1 0 [2] 10
  // 2 1 3 -> 0 1 1 [0] 00
  // 2 3 1 -> 1 0 0 [0] 00
  // 3 1 2 -> 0 0 1 [2] 10
  // 3 2 1 -> 0 0 0 [1] 01
  int lookup[8] = { 1, 2, 0, 0, 0, 0, 2, 1};
  int compare1 = in_signal[index] < in_signal[index + 1];
  int compare2 = in_signal[index] < in_signal[index + 2];
  int compare3 = in_signal[index + 1] < in_signal[index + 2];
  int compare_mask = (compare1 << 2) | (compare2 << 1) | compare3;
  return lookup[compare_mask];
}

__device__ int
int_3_median_value(
  int * in_signal,
  int index
  )
{
  return in_signal[index + int_3_median_index(in_signal, index)];
}

__global__ void
int_3_median_reduction(
  int * out_signal,
  int * out_index,
  int * in_signal,
  int * in_index)
{
  // Reduce 3-1 by getting the median of 3 element chunks
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int median_index = int_3_median_index(in_signal, tid * 3);
  out_signal[tid] = in_signal[tid * 3 + median_index];
  out_index[tid] = in_index[tid * 3 + median_index];
}

// Assumes that the input has been padded with the last element 2 times
// e.g. [1, 3, 4, 2] -> [1, 3, 4, 2, 2, 2]
__global__ void
int_3_median_filter(
  int * out_signal,
  int * in_signal)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  out_signal[tid] = int_3_median_value(in_signal, tid);
}

__global__ void
index_of_peak(
  int * out_signal,
  int * mask_signal,
  int * in_signal)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int which_block = tid >> 5; // tid / 32
  if (in_signal[tid]) {
    out_signal[which_block] = tid;
    mask_signal[which_block] = 1;
  }
}

__global__ void
nonzero(
  int * out_ary,
  int * in_ary,
  int length)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < length) {
    out_ary[idx] = in_ary[idx] != 0;
  }
}

__global__ void
scatter(
  int * out_ary,
  int * values,
  int * indecies,
  int * present,
  int length)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int new_idx;
  if (idx < length) {
    if (present[idx]) {
      new_idx = indecies[idx];
      out_ary[new_idx] = values[idx];
    }
  }
}

__global__ void
get_compact_rr(
  int * out,
  int * samples,
  int sampling_rate,
  int length)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < length - 1) {    
    out[tid+1] = (60 * sampling_rate) / (samples[tid+1] - samples[tid]);  
  }
}
__global__ void
get_rr(
  int * out_signal,
  int * in_signal,
  int * mask_signal,
  int window_size,
  int sampling_rate,
  int size)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int myval = in_signal[tid];
  int valid = mask_signal[tid];
  int i;
  int end;

  if (valid) {
    end = (((tid + window_size) > size) ? size : tid + window_size);
    for (i = tid + 1; i < end; i++) {
      if (mask_signal[i]) {
        out_signal[tid] = (60 * sampling_rate) / (in_signal[i] - myval);
        break;
      }
    }
  }
}
