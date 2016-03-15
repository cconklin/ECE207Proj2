#pragma once

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

void parallel_turning_point_compress(float *, float *, int);

void inclusive_scan(int *, int *, int);

void exclusive_scan(int *, int *, int);

void device_index(int *, int *, int);
