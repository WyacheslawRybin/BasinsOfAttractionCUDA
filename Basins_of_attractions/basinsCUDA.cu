#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "basinsCUDA.cuh"
#include "cudaMacros.cuh"
#include "systems.cuh"
#include <fstream>
#include <chrono>

#define DEBUG



namespace basinsGPU {
	__constant__ numb d_tMax;
	__constant__ int d_nPts;
	__constant__ numb d_h;
	__constant__ int d_amountOfInitialConditions;

	__constant__ int d_writableVar;
	__constant__ numb d_maxValue;
	__constant__ numb d_transientTime;

	__constant__ int d_amountOfValues;
	__constant__ int d_preScaller;
	__constant__ numb d_eps;


	__constant__ int d_sizeOfBlock;
	__constant__ int d_dimension;
	__constant__ int d_amountOfIterations;


	__constant__ int d_nPtsLimiter;

	__constant__ int d_amountOfPointsInBlock;
	__constant__ int d_amountOfPointsForSkip;
	__constant__ int d_originalNPtsLimiter;

	__constant__ int d_amountOfCalculatedPoints;


	void CUDA_dbscan(numb* data, numb* intervals, int* labels, int* helpfulArray, const int amountOfData, const numb eps, int block_size)
	{
		int resultClusters = 0;
		int amountOfClusters = 0;               // Number of clusters
		int amountOfNegativeClusters = 0;
		int* amountOfNeighbors = new int[1];     // Helper variable - how many neighbors were found for a point
		*amountOfNeighbors = 0;
		int* neighbors = new int[amountOfData]; // Helper variable - indices of found neighbors
		int* d_amountOfNeighbors;               // Helper variable - how many neighbors were found for a point (device/GPU)
		int* d_neighbors;                       // Helper variable - indices of found neighbors (device/GPU)

		cudaMalloc((void**)&d_amountOfNeighbors, sizeof(int));
		cudaMalloc((void**)&d_neighbors, sizeof(int) * amountOfData);

		cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

		int amountOfVisitedPoints = 0;

		int blockSize1;
		int minGridSize1;
		int gridSize1;


		//cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, amountOfData);

		//blockSize1 = blockSize1 > 512 ? 512 : blockSize1;
		blockSize1 = block_size;
		gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

		int blockSize2;
		int minGridSize2;
		int gridSize2;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, amountOfData);

		//blockSize2 = blockSize2 > 512 ? 512 : blockSize2;
		blockSize2 = block_size;

		gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;


		for (int i = 0; i < amountOfData; i++)
		{
			int* clearIdx = new int[1];
			*clearIdx = -1;

			int* d_clearIdx;

			cudaMalloc((void**)&d_clearIdx, sizeof(int));

			cudaMemcpy(d_clearIdx, clearIdx, sizeof(int), cudaMemcpyHostToDevice);

			CUDA_dbscan_search_fixed_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
				amountOfData, d_clearIdx);

			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

			if (*clearIdx == -1)
			{
				CUDA_dbscan_search_clear_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
					amountOfData, d_clearIdx);

				++amountOfClusters;
				resultClusters = amountOfClusters;
				if (cudaGetLastError() != cudaSuccess)
				{
					fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
				}

				//gpuGlobalErrorCheck();
				cudaDeviceSynchronize();

				cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

				if (*clearIdx == -1) {
					cudaFree(d_clearIdx);
					delete[] clearIdx;
					break;
				}
			}
			else
			{
				--amountOfNegativeClusters;
				resultClusters = amountOfNegativeClusters;
			}

			*amountOfNeighbors = 0;
			for (size_t i = 0; i < amountOfData; ++i)
				neighbors[i] = 0;

			cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

			CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
				resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, *clearIdx, helpfulArray);

			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);


			for (size_t i = 0; i < *amountOfNeighbors; ++i)
			{
				CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
					resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, neighbors[i], helpfulArray);

				if (cudaGetLastError() != cudaSuccess)
				{
					fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
				}

				//gpuGlobalErrorCheck();
				cudaDeviceSynchronize();

				cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);

				++amountOfVisitedPoints;
			}

			cudaFree(d_clearIdx);
			delete[] clearIdx;
		}

		delete[] amountOfNeighbors;
		delete[] neighbors;

		cudaFree(d_amountOfNeighbors);
		cudaFree(d_neighbors);

	}
	__device__ void calculateDiscreteModel(numb* X, const numb* a, const numb h)
	{
		CALC_DISCRETE_MODEL(X, a, h);
	}

	__device__ __host__ numb getValueByIdx(const int idx, const int nPts,
		const numb startRange, const numb finishRange, const int valueNumber)
	{
		return startRange + (((int)((int)idx / pow((numb)nPts, (numb)valueNumber)) % nPts) * ((numb)(finishRange - startRange) / (numb)(nPts - 1)));
	}

	__global__ void calculateDiscreteModelICCUDA(
		numb* ranges,
		int* indicesOfMutVars,
		numb* initialConditions,
		const numb* values,
		numb* data,
		int* maxValueCheckerArray)
	{

		extern __shared__ numb s[];

		numb* localX = s + (threadIdx.x * d_amountOfInitialConditions);
		numb* localValues = s + (blockDim.x * d_amountOfInitialConditions) + (threadIdx.x * d_amountOfValues);

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= d_nPtsLimiter)	return;

		for (int i = 0; i < d_amountOfInitialConditions; ++i)
			localX[i] = initialConditions[i];

		for (int i = 0; i < d_amountOfValues; ++i)
			localValues[i] = values[i];

		for (int i = 0; i < d_dimension; ++i)
			localX[indicesOfMutVars[i]] = getValueByIdx(d_amountOfCalculatedPoints + idx,
				d_nPts, ranges[i * 2], ranges[i * 2 + 1], i);

		int flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfPointsForSkip,
			d_amountOfInitialConditions, 1, 0, 0, nullptr, idx * d_amountOfPointsInBlock);

		if (flag == 1 || flag == -1)
			flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfPointsInBlock,
				d_amountOfInitialConditions, d_preScaller, d_writableVar, d_maxValue, data, idx * d_amountOfPointsInBlock);


		if (maxValueCheckerArray != nullptr) {
			maxValueCheckerArray[idx] = flag;
		}

		return;
	}

	__global__ void calculateTransTimeCUDA(
		numb* ranges,
		int* indicesOfMutVars,
		numb* initialConditions,
		const numb* values,
		numb* semi_result,
		int* maxValueCheckerArray)
	{
		numb localX[SIZE_X];
		numb localValues[SIZE_A];

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= d_nPtsLimiter) return;

		for (int i = 0; i < d_amountOfInitialConditions; ++i)
			localX[i] = initialConditions[i];

		for (int i = 0; i < d_amountOfValues; ++i)
			localValues[i] = values[i];

		for (int i = 0; i < d_dimension; ++i) {
			int valueIdx = d_amountOfCalculatedPoints + idx;
			int nPts = d_nPts;
			numb startRange = ranges[i * 2];
			numb finishRange = ranges[i * 2 + 1];
			int valueNumber = i;
			localX[indicesOfMutVars[i]] = startRange +
				(((int)((int)valueIdx / pow((numb)nPts, (numb)valueNumber)) % nPts) *
					((numb)(finishRange - startRange) / (numb)(nPts - 1)));
		}

		{
#pragma unroll 4
			for (int i = 0; i < d_amountOfPointsForSkip; ++i) {
				calculateDiscreteModel(localX, localValues, d_h);


			}
		}

#pragma unroll
		for (int i = 0; i < d_amountOfInitialConditions; ++i) {
			semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + i] = localX[i];
		}
#pragma unroll
		for (int i = 0; i < d_amountOfValues; ++i)
			semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + d_amountOfInitialConditions + i] = localValues[i];
	}

	__global__ void calculateDiscreteModelCUDA(
		numb* ranges,
		int* indicesOfMutVars,
		numb* initialConditions,
		const numb* values,
		numb* data,
		numb* semi_result,
		int* maxValueCheckerArray)
	{
		numb localX[SIZE_X];
		numb localValues[SIZE_A];

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= d_nPtsLimiter)	return;

		for (int i = 0; i < d_amountOfInitialConditions; ++i)
			localX[i] = semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + i];

		for (int i = 0; i < d_amountOfValues; ++i)
			localValues[i] = semi_result[idx * (d_amountOfInitialConditions + d_amountOfValues) + d_amountOfInitialConditions + i];

		int flag;

		flag = loopCalculateDiscreteModel_int(localX, localValues, d_h, d_amountOfPointsInBlock,
			d_amountOfInitialConditions, d_preScaller, d_writableVar, d_maxValue, data, idx * d_amountOfPointsInBlock);

		if (maxValueCheckerArray != nullptr) {
			maxValueCheckerArray[idx] = flag;
		}


		return;
	}


	//__device__ __host__ int loopCalculateDiscreteModel_int(
	__device__ int loopCalculateDiscreteModel_int(
		numb* x,
		const numb* values,
		const numb h,
		const int amountOfIterations,
		const int amountOfX,
		const int preScaller,
		int writableVar,
		const numb maxValue,
		numb* data,
		const int startDataIndex,
		const int writeStep)
	{
		numb xPrev[SIZE_X];

		for (int i = 0; i < amountOfIterations; ++i)
		{
			for (int j = 0; j < amountOfX; ++j)
			{
				xPrev[j] = x[j];
			}


			if (data != nullptr)
				data[startDataIndex + i * writeStep] = (x[writableVar]);


			for (int j = 0; j < preScaller - 1; ++j)
				calculateDiscreteModel(x, values, h);

			calculateDiscreteModel(x, values, h);

			if (isnan(x[writableVar]) || isinf(x[writableVar]))
			{
				delete[] xPrev;
				return 0;
			}

			if (maxValue != 0)
				if (fabsf(x[writableVar]) > maxValue)
				{
					delete[] xPrev;
					return 0;
				}
		}

		numb tempResult = 0;

		for (int j = 0; j < amountOfX; ++j)
		{
			tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
		}


		if (sqrt(abs(tempResult)) < 1e-9)
		{
			return -1;
		}

		return 1;
	}

	__device__ __host__ int peakFinder(numb* data, const int startDataIndex,
		const int amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h)
	{
		int amountOfPeaks = 0;

		for (int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i)
		{
			if (data[i] - data[i - 1] > 1e-13 && data[i] >= data[i + 1]) //&&data[j] > 0.2
			{
				for (int j = i; j < startDataIndex + amountOfPoints - 1; ++j)
				{
					if (data[j] < data[j + 1])
					{
						i = j + 1;
						break;
					}
					if (data[j] - data[j + 1] > 1e-13) //&&data[j] > 0.2
					{
						if (outPeaks != nullptr)
							outPeaks[startDataIndex + amountOfPeaks] = data[j];
						if (timeOfPeaks != nullptr)
							timeOfPeaks[startDataIndex + amountOfPeaks] = trunc(((numb)j + (numb)i) / (numb)2);
						++amountOfPeaks;
						i = j + 1;
						break;
					}
				}
			}
		}
		if (amountOfPeaks > 1) {
			for (size_t i = 0; i < amountOfPeaks - 1; i++)
			{
				if (outPeaks != nullptr)
					outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
				if (timeOfPeaks != nullptr)
					timeOfPeaks[startDataIndex + i] = (numb)(timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i]) * h;
			}
			amountOfPeaks = amountOfPeaks - 1;
		}
		else {
			amountOfPeaks = 0;
		}


		return amountOfPeaks;
	}

	__host__ void basinsOfAttraction_2(
		const numb  tMax,                            // Time for modeling the system
		const int     nPts,                            // Resolution of the diagram
		const numb   h,                              // Integration step
		const int     amountOfInitialConditions,       // Number of initial conditions (equations in the system)
		const numb* initialConditions,               // Array with initial conditions
		const numb* ranges,                          // Ranges for varying parameters
		const int* indicesOfMutVars,                  // Indices of mutable parameters
		const int     writableVar,                     // Index of the equation for which the diagram will be constructed
		const numb  maxValue,                        // Maximum value (in absolute terms) above which the system is considered "diverged"
		const numb  transientTime,                   // Time to be modeled before calculating the diagram
		const numb* values,                          // Parameters
		const int     amountOfValues,                  // Number of parameters
		const int     preScaller,                      // Multiplier that reduces time and computational load (only every 'preScaller' point will be calculated)
		const numb  eps,                             // Epsilon for the DBSCAN algorithm
		const int block_size,						// custom block size
		std::string   OUT_FILE_PATH,
		int time[3])                   // Output file path
	{
		time[0] = 0;
		time[1] = 0;
		time[2] = 0;

		int amountOfPointsInBlock = tMax / h / preScaller;

		int amountOfPointsForSkip = transientTime / h;

		size_t freeMemory;                              // Variable to hold the free memory on the GPU
		size_t totalMemory;                             // Variable to hold the total memory on the GPU

		gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory)); // Get the free and total memory on the GPU

		freeMemory *= 0.9;                             // Memory limiter (only a portion of available GPU memory will be used)      

		// --- Calculate the number of systems we can model in parallel at one moment in time ---
		// TODO: Implement memory requirement calculation
		size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);

		nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter; // If we can calculate more systems than required, limit it to the maximum (nPts)

		size_t originalNPtsLimiter = nPtsLimiter;      // Store the original value of nPts for further calculations (getValueByIdx)

		numb* d_data;                              // Pointer to the array in GPU memory for storing the trajectory of the system
		numb* d_ranges;                            // Pointer to the array with the range of the variable changes
		int* d_indicesOfMutVars;                     // Pointer to the array with indices of mutable variables in the values array
		numb* d_initialConditions;                  // Pointer to the array with initial conditions
		numb* d_values;                             // Pointer to the array with parameters

		int* d_amountOfPeaks;                         // Pointer to the GPU array with the number of peaks in each system
		numb* d_intervals;                          // Pointer to the GPU array with the peak intervals
		int* d_dbscanResult;                          // Pointer to the GPU array for the resulting matrix (diagram)
		int* d_helpfulArray;                          // Pointer to the GPU array for auxiliary data

		numb* d_avgPeaks;
		numb* d_avgIntervals;


		gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
		gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(numb)));
		gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
		gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(numb)));
		gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));

		gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));
		gpuErrorCheck(cudaMalloc((void**)&d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
		gpuErrorCheck(cudaMalloc((void**)&d_dbscanResult, nPts * nPts * sizeof(int)));
		gpuErrorCheck(cudaMalloc((void**)&d_helpfulArray, nPts * nPts * sizeof(int)));


		gpuErrorCheck(cudaMalloc((void**)&d_avgPeaks, nPts * nPts * sizeof(numb)));
		gpuErrorCheck(cudaMalloc((void**)&d_avgIntervals, nPts * nPts * sizeof(numb)));

		gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

		size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

		gpuErrorCheck(cudaMemcpyToSymbol(d_h, &h, sizeof(numb)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_tMax, &tMax, sizeof(numb)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_nPts, &nPts, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfIterations, &amountOfIteration, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfInitialConditions, &amountOfInitialConditions, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfValues, &amountOfValues, sizeof(int)));

		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsInBlock, &amountOfPointsInBlock, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfPointsForSkip, &amountOfPointsForSkip, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_writableVar, &writableVar, sizeof(int)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_maxValue, &maxValue, sizeof(numb)));
		gpuErrorCheck(cudaMemcpyToSymbol(d_preScaller, &preScaller, sizeof(int)));

		int dimension = 2;
		gpuErrorCheck(cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(int)));





		std::ofstream outFileStream;
		outFileStream.open(OUT_FILE_PATH);

		// ------------------------------------------------------

#ifdef DEBUG
		printf("Basins of attraction\n");
		printf("nPtsLimiter : %zu\n", nPtsLimiter);
		printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

		int stringCounter = 0;

		outFileStream << std::setprecision(15);

		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
		//stringCounter = 0;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



		int blockSize;
		int minGridSize;
		int gridSize;

		for (int i = 0; i < amountOfIteration; ++i)
		{
			auto start = std::chrono::high_resolution_clock::now();

			if (i == amountOfIteration - 1)
				nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);


			blockSize = block_size;

			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


			gpuErrorCheck(cudaMemcpyToSymbol(d_nPtsLimiter, &nPtsLimiter, sizeof(int)));
			int calculatedPoints = i * originalNPtsLimiter;
			gpuErrorCheck(cudaMemcpyToSymbol(d_amountOfCalculatedPoints, &calculatedPoints, sizeof(int)));




			numb* d_semi_result;
			gpuErrorCheck(cudaMalloc((void**)&d_semi_result, nPtsLimiter * (amountOfInitialConditions + amountOfValues) * sizeof(numb)));

			calculateTransTimeCUDA << <gridSize, blockSize >> > (
				d_ranges,
				d_indicesOfMutVars,
				d_initialConditions,
				d_values,
				d_semi_result,
				d_helpfulArray + (i * originalNPtsLimiter));
			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			calculateDiscreteModelCUDA << <gridSize, blockSize >> > (
				d_ranges,
				d_indicesOfMutVars,
				d_initialConditions,
				d_values,
				d_data,
				d_semi_result,
				d_helpfulArray + (i * originalNPtsLimiter));

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());
			gpuErrorCheck(cudaFree(d_semi_result));



			//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, nPtsLimiter);
			blockSize = block_size;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


			avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,
					amountOfPointsInBlock,
					nPtsLimiter,
					d_avgPeaks + (i * originalNPtsLimiter),
					d_avgIntervals + (i * originalNPtsLimiter),
					d_data,
					d_intervals,
					d_helpfulArray + (i * originalNPtsLimiter),
					h * preScaller);
			gpuGlobalErrorCheck();

			gpuErrorCheck(cudaDeviceSynchronize());


			auto end = std::chrono::high_resolution_clock::now();
			time[0] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
#ifdef DEBUG
			printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
		}
		auto start2 = std::chrono::high_resolution_clock::now();

		CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps, block_size);
		auto end2 = std::chrono::high_resolution_clock::now();
		time[1] += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();

		numb* h_avgPeaks = new numb[nPts * nPts];
		numb* h_avgIntervals = new numb[nPts * nPts];
		int* h_helpfulArray = new int[nPts * nPts];
		int* h_dbscanResult = new int[nPts * nPts];

		gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		stringCounter = 0;
		outFileStream.open(OUT_FILE_PATH, std::ios::app);
		for (size_t i = 0; i < nPts * nPts; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				outFileStream << h_dbscanResult[i];
				++stringCounter;
			}
			else
			{
#ifdef DEBUG
				printf("\nOutput file open error\n");
#endif
				exit(1);
			}
		outFileStream.close();

		stringCounter = 0;
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
		for (size_t i = 0; i < nPts * nPts; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				if (h_avgPeaks[i] != NAN)
					outFileStream << h_avgPeaks[i];
				else
					outFileStream << 999;
				++stringCounter;
			}
		outFileStream.close();

		stringCounter = 0;
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
		for (size_t i = 0; i < nPts * nPts; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}
				if (h_avgIntervals[i] != NAN)
					outFileStream << h_avgIntervals[i];
				else
					outFileStream << 999;
				++stringCounter;
			}
		outFileStream.close();


		stringCounter = 0;
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
		for (size_t i = 0; i < nPts * nPts; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}

				if (h_helpfulArray[i] != NAN)
					outFileStream << h_helpfulArray[i];
				else
					outFileStream << 999;
				++stringCounter;
			}
		outFileStream.close();



		gpuErrorCheck(cudaFree(d_data));
		gpuErrorCheck(cudaFree(d_ranges));
		gpuErrorCheck(cudaFree(d_indicesOfMutVars));
		gpuErrorCheck(cudaFree(d_initialConditions));
		gpuErrorCheck(cudaFree(d_values));

		gpuErrorCheck(cudaFree(d_amountOfPeaks));
		gpuErrorCheck(cudaFree(d_intervals));
		gpuErrorCheck(cudaFree(d_dbscanResult));
		gpuErrorCheck(cudaFree(d_helpfulArray));

		delete[] h_dbscanResult;
		delete[] h_avgPeaks;
		delete[] h_avgIntervals;
		delete[] h_helpfulArray;
		cudaDeviceReset();


		// ---------------------------
		time[2] = time[0] + time[1];
	}

__global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
		numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h)
	{
		// ---   ,     ---
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= amountOfBlocks)		//     ,   -   
			return;

		if (systemCheker[idx] == 0) // unbound solution
		{
			outAvgPeaks[idx] = 999;
			AvgTimeOfPeaks[idx] = 999;
			return;
		}

		if (systemCheker[idx] == -1) //fixed point
		{
			outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock - 1];
			AvgTimeOfPeaks[idx] = -1.0;
			return;
		}


		outAvgPeaks[idx] = 0;
		AvgTimeOfPeaks[idx] = 0;


		int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

		if (amountOfPeaks <= 0)
		{
			outAvgPeaks[idx] = 1000;
			AvgTimeOfPeaks[idx] = 1000;
			return;
		}

		for (int i = 0; i < amountOfPeaks; ++i)
		{
			outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
			AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
		}

		outAvgPeaks[idx] /= amountOfPeaks;
		AvgTimeOfPeaks[idx] /= amountOfPeaks;

		return;
	}

	__global__ void CUDA_dbscan_search_clear_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
		const int amountOfData, int* res)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
		if (idx >= amountOfData)								//    -   
			return;

		if (labels[idx] == 0 && helpfulArray[idx] == 1)
		{
			*res = idx;
			return;
		}
	}

	__global__ void CUDA_dbscan_search_fixed_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
		const int amountOfData, int* res)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
		if (idx >= amountOfData)								//    -   
			return;

		if (helpfulArray[idx] == -1 && labels[idx] == 0)
		{
			*res = idx;
			return;
		}
	}

	__global__ void CUDA_dbscan_kernel(numb* data, numb* intervals, int* labels,
		const int amountOfData, const numb eps, int amountOfClusters,
		int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;		//    
		if (idx >= amountOfData)								//    -   
			return;



		labels[idxCurPoint] = amountOfClusters;

		if (labels[idx] != 0)
			return;

		if (idx == idxCurPoint)
			return;


		if (helpfulArray[idxCurPoint] == 0) {
			labels[idxCurPoint] = 0;
			return;
		}

		if (sqrt((data[idxCurPoint] - data[idx]) * (data[idxCurPoint] - data[idx]) + (intervals[idxCurPoint] - intervals[idx]) * (intervals[idxCurPoint] - intervals[idx])) <= eps)
		{
			labels[idx] = labels[idxCurPoint];
			neighbors[atomicAdd(amountOfNeighbors, 1)] = idx;
		}
	}
}