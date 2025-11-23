#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "basinsCUDA.cuh"
#include "cudaMacros.cuh"

#include <iomanip>
#include <string>

namespace Basins {

   void basinsOfAttraction_2(
        const numb tMax,                              
        const int nPts,                                 
        const numb h,                                 
        const int amountOfInitialConditions,            
        const numb* initialConditions,                
        const numb* ranges,                           
        const int* indicesOfMutVars,                   
        const int writableVar,                         
        const numb maxValue,                          
        const numb transientTime,                    
        const numb* values,                           
        const int amountOfValues,                       
        const int preScaller,                          
        const numb eps,
        const int block_size,              
        std::string OUT_FILE_PATH,
        int time[3]);
   void basinsOfAttraction_2_only_dbscan(
       const numb tMax,
       const int nPts,
       const numb h,
       const int amountOfInitialConditions,
       const numb* initialConditions,
       const numb* ranges,
       const int* indicesOfMutVars,
       const int writableVar,
       const numb maxValue,
       const numb transientTime,
       const numb* values,
       const int amountOfValues,
       const int preScaller,
       const numb eps,
       const int block_size,
       std::string OUT_FILE_PATH,
       int time[3]);

}