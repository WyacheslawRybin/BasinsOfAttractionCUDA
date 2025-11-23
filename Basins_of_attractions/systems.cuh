#ifndef SYSTEMS_CUH
#define SYSTEMS_CUH


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

typedef float numb;


#define USE_SYSTEM_FOR_BASINS

#ifdef USE_CHAMELEON_MODEL
__device__ inline void calcDiscreteModel(numb* X, const numb* a, numb h) {
    numb h1 = a[0] * h;
    numb h2 = (1 - a[0]) * h;
    X[0] = __fma_rn(h1, -a[6] * X[1], X[0]);
    X[1] = __fma_rn(h1, a[6] * X[0] + a[1] * X[2], X[1]);
    numb cos_term = cos(a[5] * X[1]);
    X[2] = __fma_rn(h1, a[2] - a[3] * X[2] + a[4] * cos_term, X[2]);

    X[2] = __fma_rn(h2, (a[2] + a[4] * cos_term), X[2]) / (1 + a[3] * h2);
    X[1] = __fma_rn(h2, (a[6] * X[0] + a[1] * X[2]), X[1]);
    X[0] = __fma_rn(h2, -a[6] * X[1], X[0]);
}
#define SIZE_X 3
#define SIZE_A 7
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_ROSSLER_MODEL
__device__ inline void calcDiscreteModel(numb* x, const numb* a, numb h) {
    numb h1 = 0.5 * h + a[0];
    numb h2 = 0.5 * h - a[0];

    x[0] = h1 * (-x[1] - x[2]) + x[0];
    x[1] = h1 * (x[0] + a[1] * x[1]) + x[1];
    x[2] = h1 * (a[2] + x[2] * (x[0] - a[3])) + x[2];

    numb temp = -h2 * (x[0] - a[3]) + 1.0;
    x[2] = (h2 * a[2] + x[2]) / temp;

    temp = -h2 * a[1] + 1.0;
    x[1] = (h2 * x[0] + x[1]) / temp;

    x[0] = h2 * (-x[1] - x[2]) + x[0];
}
#define SIZE_X 3
#define SIZE_A 4
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#define CALC_DISCRETE_MODEL_FF(X, a, h) calcDiscreteModelFloatFloat(X, a, h)
#endif



#ifdef USE_SYSTEM_FOR_BASINS
__device__ inline void calcDiscreteModel(numb* X, const numb* a, numb h) {
    numb h1 = h * a[0];
    numb h2 = h * (1 - a[0]);

    X[0] = X[0] + h * (sin(X[1]) - a[1] * X[0]);
    X[1] = X[1] + h * (sin(X[2]) - a[1] * X[1]);
    X[2] = X[2] + h * (sin(X[0]) - a[1] * X[2]);

    X[2] = (X[2] + h2 * sin(X[0])) / (1 + h2 * a[1]);
    X[1] = (X[1] + h2 * sin(X[2])) / (1 + h2 * a[1]);
    X[0] = (X[0] + h2 * sin(X[1])) / (1 + h2 * a[1]);
}
#define SIZE_X 3
#define SIZE_A 5
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_SYSTEM_FOR_BASINS_2
__device__ inline void calcDiscreteModel(numb* X, const numb* a, numb h) {
    numb h1 = h * a[0];
    numb h2 = h * (1 - a[0]);

    X[0] = X[0] + h1 * (-X[1]);
    X[1] = X[1] + h1 * (a[1] * X[0] + sin(X[1]));

    numb z = X[1];

    X[1] = z + h2 * (a[1] * X[0] + sin(X[1]));
    X[0] = X[0] + h2 * (-X[1]);
}
#define SIZE_X 2
#define SIZE_A 2
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif

#ifdef USE_CUSTOM_SYSTEM
__device__ inline void calcDiscreteModel(numb* X, const numb* a, numb h) {
    //
    // PLACE YOUR CODE HERE
    //
}
#define SIZE_X /*Number of dimensions*/
#define SIZE_A /*Number of parameters*/
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif




#endif // SYSTEMS_CUH
