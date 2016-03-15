#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <pthread.h>
#include "cuda_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <tgmath.h>
#include <sys/time.h>
#include <assert.h>

#include "half.hpp"

#include "plotecg.h"
#include "kernels.h"

#define checkCuda(result) _checkCuda(result, __LINE__, __FILE__)
#define KERNEL(func) func<<<num_blocks, threads_per_block>>>

inline
cudaError_t _checkCuda(cudaError_t result, int l, const char * f)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s (%s: line %d)\n", cudaGetErrorString(result), f, l);
    assert(result == cudaSuccess);
  }
  return result;
}

void threshold_ecg(float * output1,
                   float * output2,
                   float * output3,
                   float * samples,
                   int * output_len,
                   float * input1,
                   float * input2,
                   float * input3,
                   int input_len,
                   float threshold)
{
  float neg_threshold = - threshold;
  int i = 0;
  int idx = 0;
  for (i = 0; i < input_len; i++) {
    float val1 = input1[i];
    float val2 = input2[i];
    float val3 = input3[i];
    if (val1 < neg_threshold || val1 > threshold) {
      output1[idx] = val1;
      output2[idx] = val2;
      output3[idx] = val3;
      samples[idx++] = i;
    }
  }
  * output_len = idx;
}

double get_time(void) {
  struct timeval t;

  gettimeofday(&t, NULL);
  return (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
}

double elapsed_time(double start_time, double end_time) {
  // Get the elapsed time
  return ((end_time - start_time) / 1000.0);
}

void turning_point_compress(uint16_t * output,
                            float * input,
                            int input_len)
{
  int idx;
  int output_len = input_len / 2;
  output[0] = input[0];
  for (idx = 1; idx < output_len; idx++) {
    if ((input[2*idx]-output[idx-1])*(input[2*idx+1]-input[2*idx]) < 0) {
      output[idx] = half_float::detail::float2half<std::round_indeterminate>(input[2*idx]);
    } else {
      output[idx] = half_float::detail::float2half<std::round_indeterminate>(input[2*idx+1]);
    }
  }
}

void * tp_worker(void * _args) {
  struct tp_arg * args = (struct tp_arg *) _args;
  uint16_t * output = args -> output;
  float * input = args -> input;
  int len = args -> len;
  turning_point_compress(output, input, len);
  pthread_exit(NULL);
}

void parallel_turning_point_compress(uint16_t * output,
                                     float * input,
                                     int input_len)
{
  int num_threads = 8;
  int tid;
  struct tp_arg thread_args[num_threads];
  pthread_t threads[num_threads];
  pthread_attr_t th_attr;
  pthread_attr_init(&th_attr);
  pthread_attr_setdetachstate(&th_attr, PTHREAD_CREATE_JOINABLE);
  int chunk_size = input_len / num_threads;
  for (tid = 0; tid < num_threads; tid++) {
    (&thread_args[tid]) -> output = & output[chunk_size * tid / 2];
    (&thread_args[tid]) -> input = & input[chunk_size * tid];
    (&thread_args[tid]) -> len = chunk_size;
    pthread_create(&threads[tid], &th_attr, tp_worker, (void *) & thread_args[tid]);
  }
  for (tid = 0; tid < num_threads; tid++) {
    pthread_join(threads[tid], NULL);
  }
  pthread_attr_destroy(&th_attr);
}

void inclusive_scan(int * out, int * in, int len) {
  thrust::device_ptr<int> in_p = thrust::device_pointer_cast(in);
  thrust::device_ptr<int> out_p = thrust::device_pointer_cast(out);
  thrust::inclusive_scan(in_p, in_p+len, out_p);
}

void exclusive_scan(int * out, int * in, int len) {
  thrust::device_ptr<int> in_p = thrust::device_pointer_cast(in);
  thrust::device_ptr<int> out_p = thrust::device_pointer_cast(out);
  thrust::exclusive_scan(in_p, in_p+len, out_p);
}

void device_index(int * ary, int * last_val, int idx) {
  cudaMemcpy(last_val, & ary[idx], sizeof(int), cudaMemcpyDeviceToHost);
}

void synchronize_and_merge(int ** merged_out,
                           int * merged_length_out,
                           int * d_lead1,
                           int * d_lead2,
                           int * d_lead3,
                           int lead_length,
                           int chunk_length)
{
  int * lead1, * lead2, * lead3;
  size_t chunk_size = chunk_length * sizeof(int);
  int start1 = 0, start2 = 0, start3 = 0;
  int offset1, offset2, offset3;
  int minstart, maxstart;
  int i;
  int sync_length;
  int threads_per_block = 256;
  int num_blocks;
  // Allocate small chunks
  lead1 = malloc(chunk_size);
  assert(lead1);
  lead2 = malloc(chunk_size);
  assert(lead2);
  lead3 = malloc(chunk_size);
  assert(lead3);
  // Copy back
  checkCuda( cudaMemcpy(lead1, d_lead1, chunk_size, cudaMemcpyDeviceToHost) );
  checkCuda( cudaMemcpy(lead2, d_lead2, chunk_size, cudaMemcpyDeviceToHost) );
  checkCuda( cudaMemcpy(lead3, d_lead3, chunk_size, cudaMemcpyDeviceToHost) );
  // Find the index of the max element
  for (i = 0; i < chunk_length; i++) {
    if (lead1[i] && !start1) {
      start1 = i;
    }
    if (lead2[i] && !start2) {
      start2 = i;
    }
    if (lead3[i] && !start3) {
      start3 = i;
    }
    if (start1 && start2 && start3) {
      break;
    }
  }
  minstart = std::min(std::min(start1, start2), start3);
  maxstart = std::max(std::max(start1, start2), start3);
  // Get offsets and prospective new length
  offset1 = start1 - minstart;
  offset2 = start2 - minstart;
  offset3 = start3 - minstart;
  sync_length = lead_length - (maxstart - minstart);

  // Merge
  num_blocks = sync_length / threads_per_block;
  * merged_length_out = num_blocks * threads_per_block;
  // Allocate the output
  checkCuda( cudaMalloc((void **) merged_out), * merged_length_out * sizeof(int) );
  // Merge kernel
  KERNEL(merge_leads)(* merged_out, d_lead1, offset1, d_lead2, offset2, d_lead3, offset3);
}

void get_hr(float ** out_samples,
            float ** out_rr_values,
            uint16_t * compressed_lead_1,
            uint16_t * compressed_lead_2,
            uint16_t * compressed_lead_3,
            int lead_length,
            float * d_wavelet,
            int wavelet_length,
            int sampling_rate)
{
  size_t lead_size = lead_length * sizeof(float);
  size_t int_lead_size = lead_length * sizeof(int);
  size_t compressed_lead_size = lead_length * sizeof(uint16_t);
  size_t wavelet_size = wavelet_length * sizeof(float);
  uint16_t * d_clead1, * d_clead2, * d_clead3;
  float * d_lead1, * d_lead2, * d_lead3;
  float * d_corr1, * d_corr2, * d_corr3;
  int * d_thresh1, * d_thresh2, * d_thresh3;
  float * d_merged;
  int * d_edge;
  int * d_masks;
  int * d_indecies;
  int * d_scan;
  int * d_scatter;
  int * d_rr;
  int * d_filtered;
  int merged_length;
  // Still hardcoded...
  float threshold_value = 0.5;
  int threads_per_block = 200;
  int num_blocks = lead_size / threads_per_block;
  int chunk_length = sampling_rate * 2;
  int reduce_by = 32;
  int reduced_length;
  size_t reduced_size;
  int compacted_length;
  size_t compacted_size;

  // Allocate leads
  // Compressed
  checkCuda( cudaMalloc((void **) & d_clead1, compressed_lead_size) );
  checkCuda( cudaMalloc((void **) & d_clead2, compressed_lead_size) );
  checkCuda( cudaMalloc((void **) & d_clead3, compressed_lead_size) );
  // Decompressed
  checkCuda( cudaMalloc((void **) & d_lead1, lead_size) );
  checkCuda( cudaMalloc((void **) & d_lead2, lead_size) );
  checkCuda( cudaMalloc((void **) & d_lead3, lead_size) );
  // Correlated
  checkCuda( cudaMalloc((void **) & d_corr1, lead_size) );
  checkCuda( cudaMalloc((void **) & d_corr2, lead_size) );
  checkCuda( cudaMalloc((void **) & d_corr3, lead_size) );

  // Transfer leads
  // TODO add streaming
  checkCuda( cudaMemcpy(d_clead1, compressed_lead_1, compressed_lead_size, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_clead2, compressed_lead_2, compressed_lead_size, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_clead3, compressed_lead_3, compressed_lead_size, cudaMemcpyHostToDevice) );

  // Preprocess kernels

  // "Decompress" on GPU (16 bit float to 32 bit float)
  KERNEL(to_float)(d_lead1, d_clead1, lead_length);
  KERNEL(to_float)(d_lead1, d_clead1, lead_length);
  KERNEL(to_float)(d_lead1, d_clead1, lead_length);

  // Free unneeded memory
  cudaFree(d_clead1);
  cudaFree(d_clead2);
  cudaFree(d_clead3);

  // Cross-Correlate with wavelet
  KERNEL(cross_correlate_with_wavelet)(d_corr1, d_lead1, d_wavelet, lead_length, wavelet_length);
  KERNEL(cross_correlate_with_wavelet)(d_corr2, d_lead2, d_wavelet, lead_length, wavelet_length);
  KERNEL(cross_correlate_with_wavelet)(d_corr3, d_lead3, d_wavelet, lead_length, wavelet_length);
  // Free unneeded memory
  cudaFree(d_lead1);
  cudaFree(d_lead2);
  cudaFree(d_lead3);
  cudaFree(d_wavelet);

  // Threshold
  // Allocate output
  checkCuda( cudaMalloc((void **) & d_thresh1, int_lead_size) );
  checkCuda( cudaMalloc((void **) & d_thresh2, int_lead_size) );
  checkCuda( cudaMalloc((void **) & d_thresh3, int_lead_size) );
  // Threshold Kernel
  KERNEL(threshold)(d_thresh1, d_corr1, threshold_value);
  KERNEL(threshold)(d_thresh2, d_corr2, threshold_value);
  KERNEL(threshold)(d_thresh3, d_corr3, threshold_value);
  // Free unneeded memory
  cudaFree(d_corr1);
  cudaFree(d_corr2);
  cudaFree(d_corr3);

  // Synchronize and Merge 3 Leads
  // TODO define this
  synchronize_and_merge(& d_merged, & merged_length, d_thresh1, d_thresh2, d_thresh3, lead_length, chunk_length);
  // Free unneeded memory
  cudaFree(d_thresh1);
  cudaFree(d_thresh2);
  cudaFree(d_thresh3);

  // Heartrate kernels
  reduced_length = merged_length / reduce_by;
  reduced_size = reduced_length * sizeof(int);
  num_blocks = merged_length / threads_per_block;
  KERNEL(edge_detect)(d_edge, d_merged, merged_length);
  // Free unneeded memory
  cudaFree(d_merged);

  // Pre-Collapse Sparse (32x reduction)
  // Allocate and Zero output
  checkCuda( cudaMalloc((void **) & d_masks, reduced_size) );
  checkCuda( cudaMalloc((void **) & d_indecies, reduced_size) );
  checkCuda( cudaMemset(d_masks, 0, reduced_size) );
  // TODO: is this needed?
  checkCuda( cudaMemset(d_indecies, 0, reduced_size) );
  // reduction kernel
  KERNEL(index_of_peak)(d_indecies, d_masks, d_edge);
  // Free unneeded memory
  cudaFree(d_edge);

  // Collapse Sparse (Stream Compaction)
  threads_per_block = 64;
  num_blocks = (reduced_length / threads_per_block) + 1;
  cudaMalloc((void **) & d_scan, reduced_size);
  // Scan
  exclusive_scan(d_scan, d_masks, reduced_length);
  // Get new length
  device_index(d_scan, & compacted_length, reduced_length - 1);
  compacted_size = compacted_length * sizeof(int);
  // Scatter
  num_blocks = (compacted_length / threads_per_block) + 1;
  cudaMalloc((void **) & d_scatter, compacted_size);
  KERNEL(scatter)(d_scatter, d_indecies, d_scan, d_masks, compacted_length);
  // Free unneeded memory
  cudaFree(d_scan);
  cudaFree(d_mask);

  // Get heartrate
  cudaMalloc((void **) & d_rr, compacted_size);
  KERNEL(get_compact_rr)(d_rr, d_scatter, sampling_rate, compacted_length);
  // Remove all values outside the range (40..140) starting at point 1 (i.e. ignore point 0)
  KERNEL(clean_result)(d_rr, 140, 40, 1, compacted_length);

  // Moving average filter
  cudaMalloc((void **) & d_filtered, compacted_size);
  // Reuse memory in d_scatter (it's just the right size)
  d_scan = d_scatter;
  exclusive_scan(d_scan, d_rr, compacted_length);
  // use a 250 point window for the moving average
  KERNEL(moving_average)(d_filtered, d_scan, 250, compacted_length);
  // Free unneeded memory
  cudaFree(d_scan);
  cudaFree(d_rr);

  // Transfer back to host
  // Allocate space on host
  * out_samples = (int *) malloc(compacted_size);
  assert(* out_samples);
  * out_rr_values = (int *) malloc(compacted_size);
  assert(* out_rr_values);
  // Copy back
  checkCuda( cudaMemcpy(out_samples, d_indecies, compacted_size, cudaMemcpyDeviceToHost) );
  checkCuda( cudaMemcpy(out_rr_values, d_filtered, compacted_size, cudaMemcpyDeviceToHost) );
  // Free unneeded memory
  cudaFree(d_indecies);
  cudaFree(d_filtered);
  // Correct first value of output heartrate (it's always wrong)
  (* out_rr_values)[0] = (* out_rr_values)[1];
}
