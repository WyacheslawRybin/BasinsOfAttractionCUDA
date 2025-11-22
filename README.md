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

1. **Download** the project  
  

2. **Extract** the archive contents to a convenient location on your local drive.  
   The primary working directory is named `Basins_of_attractions`.

3. **Launch Microsoft Visual Studio.**

4. From the main menu, select **“Open Project or Solution”**.

5. Navigate to the `Basins_of_attractions` folder and select the solution file **`Basins_of_attractions.sln`**.  
   Upon successful loading, the project structure will appear in the **Solution Explorer**.

6. **CUDA Toolkit Configuration:**  
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
