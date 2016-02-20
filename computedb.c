// gcc computedb.c -o test -lm to compile
// various steps: 0.125 0.0625 0.03125 0.015625


#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>

#define SQRT2 1.41421356237309504880
#define SQRT3 1.73205080757
int counter = 0;

float hcoeff(int x){

	float result;
	float array[8] = {0.2303778133, 0.7148465706, 0.6308807679, -0.0279837694, -0.1870348117, 0.0308413818,0.0328830117,-0.0105974018};
	if (x > (sizeof(array)/sizeof(float)) || x < 0){
		result = 0;
	}
	else {
		result = array[x];
	}
	//printf("hcoeff = %f\n", result);
	return result;
}
/**
float phi (float x,int N){
	float term;
	float term2;
	int i = 0;
	if (x < 0 || x > 2*N-1){
		term = 0;
	}
	else{
	while(i <= 2*N - 1){
		if (2*x-i == x){
			return 1;
		}
		term2 = term + hcoeff(i) * phi(2*x-i,N);
		if ((term2 - term) < 0.00001){
			term = term2;
			i++;
		}
		else{
			term = term2;
		}
	}
	}
	printf("(%d,%d) = %f\n", x,N,SQRT2* term);
	return SQRT2 * term;
	
}
**/
float phi(float x){//WARNING: recursive nature see last else clause
    counter++;
	int upperx = ceilf(x);
    int lowerx = floorf(x);
    float result;	
    if ((upperx - x) < 0.00000001){ // 8 digits accuracy is single precision
		//is close enough to be considered an integer   
		switch (upperx){
			case 1 : 
				result = 1.007169977725601; //to be arithmetized replace with phi(x,N)
				break;
			case 2 : 
				result = -0.03383695405283447; //to be arithmetized
				break;
			case 3 : 
				result = 0.03961046271590321;
				break;
			case 4 : 
				result = -0.01176435820572669;
				break;
			case 5 : 
				result = -0.001197957596176928;
				break;
			case 6 : 
				result = 0.00001882941323353892;
				break;
			default : // case 7 and 8 is also 0.
				result = 0;
				break;
		}
	}
	else if ((x - lowerx) < 0.00000001){ // 8 digits accuracy is single precision
		//is close enough to be considered an integer   
		switch (lowerx){
			case 1 : 
				result = 1.007169977725601; //to be arithmetized replace with phi(x,N)
				break;
			case 2 : 
				result = -0.03383695405283447; //to be arithmetized
				break;
			case 3 : 
				result = 0.03961046271590321;
				break;
			case 4 : 
				result = -0.01176435820572669;
				break;
			case 5 : 
				result = -0.001197957596176928;
				break;
			case 6 : 
				result = 0.00001882941323353892;
				break;
			default : // case 7 and 8 is also 0.
				result = 0;
				break;
		}
	}
    else{//fix this ASAP probably not even executable
	result = SQRT2 * (hcoeff(0)*phi(2*x) + hcoeff(1)*phi(2*x-1) + hcoeff(2)*phi(2*x-2) + hcoeff(3)*phi(2*x-3) + hcoeff(4)*phi(2*x-4) + hcoeff(5)*phi(2*x-5) + hcoeff(6)*phi(2*x-6) + hcoeff(7)*phi(2*x-7));
    }
    return result;
}

float db4_point(float x){//evaluates db4 at a specific point
    float term1 = hcoeff(7) * phi(2 * x - 0);
	float term2 = -hcoeff(6) * phi(2 * x - 1);
	float term3 = hcoeff(5) * phi(2 * x - 2);
	float term4 = -hcoeff(4) * phi(2 * x - 3);
	float term5 = hcoeff(3) * phi(2 * x - 4);
	float term6 = -hcoeff(2) * phi(2 * x - 5);
	float term7 = hcoeff(1) * phi(2 * x - 6);
	float term8 = -hcoeff(0) * phi(2 * x - 7);
	//should have 8 of these terms
    float term9 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8;
    return SQRT2 * term9;
}

void generate_wavelet(float * wavelet, int num_points) {
  float i, step;
  step = 8.0 / (float) num_points;
	for (i = 0; i < 8; i = i + step){
		int j = (int) (i/step);
		wavelet[j] = db4_point(i); //db4 implied
	}
}

// int main(int argc, char* argv[]){
//
//   if (argc != 2){
//     printf("Improper usage! argc = %d should be 2", argc);
//     exit(EXIT_FAILURE);
//   }
//
//     struct timeval     t;
//     float             StartTime, EndTime;
//     float             TimeElapsed;
//
//   float step = atof (argv[1]);
//   printf("input is %f\n", step);
//   float i = 0;
//   float result[8000];
//   printf("[");
//   int j,k = 0;
//
//   gettimeofday(&t, NULL);
//     StartTime = (float)t.tv_sec*1000000.0 + ((float)t.tv_usec);
//
//   for (i = 0; i < 8; i = i + step){
//     int j = (int) (i/step);
//     result[j] = db4_point(i); //db4 implied
//     //result2[j] = phi(j,4);
//     printf("%f ", result[j]);
//     k++;
//   }
//
//     gettimeofday(&t, NULL);
//     EndTime = (float)t.tv_sec*1000000.0 + ((float)t.tv_usec);
//   TimeElapsed=(EndTime-StartTime)/1000.00;
//
//   printf("]\nnumber of data points=%d\nnumber of recursive calls = %d",k, counter);
//   printf("Execution Time: %10.4f ms\n",TimeElapsed);
//   return 0;
// }
