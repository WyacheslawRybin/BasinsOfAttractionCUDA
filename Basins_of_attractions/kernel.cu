#include "basinsHOST.h"
#include "systems.cuh"

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <conio.h>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>

const std::string BASINS_OUTPUT_PATH = "C:/Users/user/Documents";

int main()
{
	size_t startTime = std::clock();
	numb h = (numb)0.01;

#ifdef USE_SYSTEM_FOR_BASINS
	numb params[5]{ 0.5, 0.1665, 1.4,  15.552, 2 };
	numb init[3]{ 0, 0, 0, };
	numb ranges[4]{ -6, 6, -6, 6 };
	int indicesOfMutVars[2]{ 0, 1 };
	const int custom_block_size = 1024;
	 {
	 std::cout << "start basins" << std::endl;
	 auto start = std::chrono::high_resolution_clock::now();
	 int time[3];
	 Basins::basinsOfAttraction_2(
	 	700,       // ct
	 	300,       // resolution
	 	h,         // time step
	 	sizeof(init) / sizeof(numb),   // amount of init conditions
	 	init,         // init conditions
	 	ranges,			// parameters range
	 	indicesOfMutVars, // indices of butual variables
	 	1,          // index of the equation to use for plotting the diagram
	 	100000000,  // maximum value (by absolute value); above this the system is considered "diverged"
	 	500,       // time that will be simulated before computing the diagram
	 	params,     // parameters
	 	sizeof(params) / sizeof(numb),  // number of parameters
	 	1,          // multiplier that reduces time and computation load (only every 'prescaller' point will be computed)
	 	0.05,       // epsilon for the dbscan algorithm
		custom_block_size,
	 	std::string(BASINS_OUTPUT_PATH) + "/bas.csv",
		time
	 );
	 auto end = std::chrono::high_resolution_clock::now();
	 auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	 std::cout << "time taken: " << duration << " milliseconds" << std::endl;
	 std::cout << "time taken: for system " << time[0] << " ms --- for dbscan " << time[1] << " ms --- at all " << time[2] << " ms" << std::endl;

	  }

#endif

#ifdef USE_SYSTEM_FOR_BASINS_2
	numb params[5]{ 0.5, 0.1, 1.4,  15.552, 2 };
	numb init[3]{ 0, 0, 0, };

	{
		std::cout << "Start basins" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		Basins::basinsOfAttraction_2(
			200,        // Simulation time of the system
			100,        // Diagram resolution
			0.01,       // Integration step
			sizeof(init) / sizeof(numb),   // Number of initial conditions (equations in the system)
			init,       // Array of initial conditions
			new numb[4] { -200, 200, -60, 60 },
			new int[2] { 0, 1 },
			1,          // Index of the equation to use for plotting the diagram
			100000000,  // Maximum value (by absolute value); above this the system is considered "diverged"
			1000,       // Time that will be simulated before computing the diagram
			params,     // Parameters
			sizeof(params) / sizeof(numb),  // Number of parameters
			1,          // Multiplier that reduces time and computation load (only every 'preScaller' point will be computed)
			0.05,       // Epsilon for the DBSCAN algorithm
			std::string(BASINS_OUTPUT_PATH) + "/bas_2.csv"
		);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
	}
#endif


	std::cout << "Time taken: " << (std::clock() - startTime) / (numb)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

	return 0;
}