#pragma once
#include "kernels.h"
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

void threshold_ecg(float *, float *, float *, float *, int *, float *, float *, float *, int, float);

double get_time(void);

double elapsed_time(double, double);

void turning_point_compress(float *, float *, int);

struct tp_arg {
  float * output;
  float * input;
  int len;
};

void * tp_worker(void *);

void parallel_turning_point_compress(float, float *, int);

void inclusive_scan(int *, int *, int);

void exclusive_scan(int *, int *, int);

void device_index(int *, int *, int)