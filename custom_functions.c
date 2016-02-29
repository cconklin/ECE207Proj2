#include <stdlib.h>

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
