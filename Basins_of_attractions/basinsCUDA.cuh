#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cudaMacros.cuh"
#include "systems.cuh"

#include <iomanip>
#include <string>

namespace basinsGPU {
    __host__ void basinsOfAttraction_2(
        const numb	tMax,				
        const int		nPts,								
        const numb	h,								
        const int		amountOfInitialConditions,			
        const numb* initialConditions,					
        const numb* ranges,								
        const int* indicesOfMutVars,					
        const int		writableVar,						
        const numb	maxValue,							
        const numb	transientTime,						
        const numb* values,								
        const int		amountOfValues,						
        const int		preScaller,							
        const numb	eps,
        const int block_size,
        std::string		OUT_FILE_PATH,
        int time[3]);
            __host__ void basinsOfAttraction_2_only_dbscan(
        const numb	tMax,				
        const int		nPts,								
        const numb	h,								
        const int		amountOfInitialConditions,			
        const numb* initialConditions,					
        const numb* ranges,								
        const int* indicesOfMutVars,					
        const int		writableVar,						
        const numb	maxValue,							
        const numb	transientTime,						
        const numb* values,								
        const int		amountOfValues,						
        const int		preScaller,							
        const numb	eps,
        const int block_size,
        std::string		OUT_FILE_PATH,
        int time[3]);

    __global__ void calculateDiscreteModelICCUDA(
        numb* ranges,
        int* indicesOfMutVars,
        numb* initialConditions,
        const numb* values,
        numb* data,
        int* maxValueCheckerArray);

    __global__ void calculateTransTimeCUDA(
        numb* ranges,
        int* indicesOfMutVars,
        numb* initialConditions,
        const numb* values,
        numb* semi_result,
        int* maxValueCheckerArray);

    __global__ void calculateTransferResultCUDA(
        numb* semi_result);

    __global__ void calculateDiscreteModelCUDA(
        numb* ranges,
        int* indicesOfMutVars,
        numb* initialConditions,
        const numb* values,
        numb* data,
        numb* semi_result,
        int* maxValueCheckerArray);


    __global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
        numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h = 0);

    __device__ __host__ numb getValueByIdx(
        const int idx,
        const int nPts,
        const numb startRange,
        const numb finishRange,
        const int valueNumber);


    __device__ int loopCalculateDiscreteModel_int(
        numb* x, const numb* values,
        const numb h, const int amountOfIterations, const int amountOfX, const int preScaller = 0,
        const int writableVar = 0, const numb maxValue = 0,
        numb* data = nullptr, const int startDataIndex = 0,
        const int writeStep = 1);

    // Function for finding peaks in time series data
    __device__ int peakFinder(
        numb* data,
        const int startDataIndex,
        const int amountOfPoints,
        numb* outPeaks,
        numb* timeOfPeaks,
        numb h = 0);


    __global__ void CUDA_dbscan_kernel(numb* data, numb* intervals, int* labels,
        const int amountOfData, const numb eps, int amountOfClusters,
        int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray);



    __global__ void CUDA_dbscan_search_clear_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
        const int amountOfData, int* res);



    __global__ void CUDA_dbscan_search_fixed_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
        const int amountOfData, int* res);


}