#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <pthread.h>
#include "cuda_runtime.h"

#include <stdio.h>
#include <tgmath.h>
#include <sys/time.h>
#include <assert.h>

#include "plotecg.h"
#include "kernels.h"

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

void turning_point_compress(float * output,
                            float * input,
                            int input_len)
{
  int idx;
  int output_len = input_len / 2;
  output[0] = input[0];
  for (idx = 1; idx < output_len; idx++) {
    if ((input[2*idx]-output[idx-1])*(input[2*idx+1]-input[2*idx]) < 0) {
      output[idx] = input[2*idx];
    } else {
      output[idx] = input[2*idx+1];
    }
  }
}

void * tp_worker(void * _args) {
  struct tp_arg * args = (struct tp_arg *) _args;
  float * output = args -> output;
  float * input = args -> input;
  int len = args -> len;
  turning_point_compress(output, input, len);
  pthread_exit(NULL);
}

void parallel_turning_point_compress(float * output,
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
