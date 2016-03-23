#pragma once
#include <stdint.h>

void threshold_ecg(float *, float *, float *, float *, int *, float *, float *, float *, int, float);

double get_time(void);

void * tp_worker(void *);

void parallel_turning_point_compress(float *, float *, int);

void inclusive_scan(int *, int *, int);

void exclusive_scan(int *, int *, int);

void device_index(int *, int *, int);
