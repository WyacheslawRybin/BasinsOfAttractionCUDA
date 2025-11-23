# CUDA_basins_of_attraction
CUDA-based basins of attraction calculation

# USER MANUAL  
**User Guide for the Software Suite for Constructing Basins of Attraction of Dynamical Systems**

---

## 🖥️ System Requirements

- **Microsoft Visual Studio 2022** (or a compatible version) must be installed.  
- A **compatible version of the NVIDIA CUDA Toolkit** must be installed.  

> **Note:** The CUDA Toolkit should be installed **after** Visual Studio to ensure proper integration of build components.

---

## 📂 Project Setup and Execution Instructions

1. **Clone or Download** the project  
  
2. **Launch Microsoft Visual Studio.**

3. From the main menu, select **“Open Project or Solution”**.

4. Navigate to the `Basins_of_attractions` folder and select the solution file **`Basins_of_attractions.sln`**.  
   Upon successful loading, the project structure will appear in the **Solution Explorer**.

5. **CUDA Toolkit Configuration:**  
   - Go to **Project → Build Customizations**  
     → Ensure that the correct version of the installed CUDA Toolkit is selected.  
   - It is recommended to enable full file visibility in the Solution Explorer via **Project → Show All Files**.

---

## 🔧 Key Files for User Customization

- **`systems.cuh`**: Definition of finite-difference schemes (discrete dynamical models).  
- **`kernel.cu`**: Implementation of the computational kernel and the program entry point (`main` function).

---

## 🧮 Defining a Custom Finite-Difference Scheme (`systems.cuh`)

To incorporate a new dynamical system, replicate the following template and adapt it to your model:

```cpp
#ifdef SYSTEM_NAME
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
    // Implementation of a single integration step
}
#define SIZE_X <number_of_state_variables>
#define SIZE_A <number_of_parameters>
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif
```
- Replace `SYSTEM_NAME` with a **unique identifier** (e.g., `USE_SYSTEM_FOR_BASINS`) for conditional compilation.  
- The function `calcDiscreteModel` implements one discrete-time evolution step with integration step size `h`.  
- `SIZE_X` = dimensionality of the phase space (number of state variables).  
- `SIZE_A` = number of model parameters.

### ✅ Example Implementation

```cpp
#ifdef USE_SYSTEM_FOR_BASINS
__device__ inline void calcDiscreteModel(double* X, const double* a, double h) {
    float h1 = h * a[0];
    float h2 = h * (1 - a[0]);

    X[0] = X[0] + h1 * (sin(X[1]) - a[1] * X[0]);
    X[1] = X[1] + h1 * (sin(X[2]) - a[1] * X[1]);
    X[2] = X[2] + h1 * (sin(X[0]) - a[1] * X[2]);

    X[2] = (X[2] + h2 * sin(X[0])) / (1 + h2 * a[1]);
    X[1] = (X[1] + h2 * sin(X[2])) / (1 + h2 * a[1]);
    X[0] = (X[0] + h2 * sin(X[1])) / (1 + h2 * a[1]);
}
#define SIZE_X 3
#define SIZE_A 2
#define CALC_DISCRETE_MODEL(X, a, h) calcDiscreteModel(X, a, h)
#endif
```
To activate a specific scheme, replace the top-level `#define` directive in `systems.cuh` with the corresponding identifier, e.g.: 

```cpp
#define USE_SYSTEM_FOR_BASINS
```
To change the precision of calculations between float/double, modify this line.
```cpp
typedef float numb;
```

### Configuring the Computational Kernel (kernel.cu):

1. Specify the output directory at the top of kernel.cu:
```cpp
   const std::string BASINS_OUTPUT_PATH = "C:/Users/user/Documents"
```
2. In the main() function, invoke the basin construction algorithm using conditional compilation:

```cpp
#ifdef USE_SYSTEM_FOR_BASINS
    double params[2]{ 0.5, 0.1665 };
    double init[3]{ 0, 0, 0 };

    {
        std::cout << "Start basins" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        Basins::basinsOfAttraction_2(
            200,                                    // Total simulation time
            600,                                    // Diagram resolution (pixels per side)
            0.01,                                   // Integration step size
            sizeof(init) / sizeof(double),          // Phase space dimensionality
            init,                                   // Initial conditions
            new double[4]{ -200, 200, -60, 60 },    // Parameter ranges [x_min, x_max, y_min, y_max]
            new int[2]{ 0, 1 },                     // Indices of variables varied across the grid
            1,                                      // Index of the state variable used for attractor classification
            100000000,                              // Divergence threshold (trajectory considered unstable if |x| exceeds this value)
            1000,                                   // Transient time (discarded before analysis)
            params,                                 // System parameters
            sizeof(params) / sizeof(double),        // Number of parameters
            1,                                      // Subsampling factor (only every 'preScaller'-th point is processed)
            0.05,                                   // eps parameter for DBSCAN clustering
            std::string(BASINS_OUTPUT_PATH) + "/bas_2.csv"
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
    }
#endif
```

**Critical Note**: The identifier used in the #ifdef directive in kernel.cu must exactly match the one activated via #define in systems.cuh.  
This design ensures modularity and facilitates rapid switching between distinct dynamical models without modifying the core computational logic.

### Program Execution:

After completing the configuration, launch the program by clicking “Local Windows Debugger” or pressing F5.  
The output will be saved as a CSV file in the specified directory and can be visualized using external tools such as Python (e.g., Matplotlib, Pandas), MATLAB, or Excel.

Note: Correct program operation requires a CUDA-compatible GPU.