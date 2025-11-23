#include "basinsHOST.h"
#include "basinsCUDA.cuh"
#include "systems.cuh"

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
        int time[3]
        )
    {
        // Delegate to the CUDA implementation
        basinsGPU::basinsOfAttraction_2(
            tMax,
            nPts,
            h,
            amountOfInitialConditions,
            initialConditions,
            ranges,
            indicesOfMutVars,
            writableVar,
            maxValue,
            transientTime,
            values,
            amountOfValues,
            preScaller,
            eps,
            block_size,
            OUT_FILE_PATH,
            time
        );
    }

}