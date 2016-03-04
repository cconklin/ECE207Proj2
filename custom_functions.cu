#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

extern "C" {
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
}

