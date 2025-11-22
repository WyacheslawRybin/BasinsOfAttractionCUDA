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
