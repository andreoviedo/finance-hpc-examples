#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric> // Required for std::accumulate
#include <cuda_runtime.h>
#include <curand_kernel.h> // For cuRAND
#include <map> // To store results easily

using namespace std::chrono;

// --- Constants ---
#define NUM_PATHS 1000000 // Number of Monte Carlo paths per option (>= 1 million)
#define NUM_OPTIONS_PER_T 100 // 100 strikes per T
#define START_STRIKE 50.0f
#define STRIKE_STEP 1.0f
#define NUM_T_VALUES 5
#define TOTAL_OPTIONS (NUM_OPTIONS_PER_T * NUM_T_VALUES) // 500 total options

// Structure to hold market parameters for each T

struct MarketParams {
    float T; // Time to maturity
    float r; // Risk-free rate
    float v; // Volatilitys
};

// Market data provided in the instructions
const std::vector<MarketParams> marketData = {
    {0.5f, 0.03f, 0.30f},
    {0.75f, 0.04f, 0.29f},
    {1.0f, 0.05f, 0.28f},
    {1.25f, 0.06f, 0.27f},
    {1.5f, 0.07f, 0.26f}
};

// Kernel to initialize cuRAND states
__global__ void setupRandStates_kernel(curandState_t* states, unsigned long long seed, int numStates) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates) {
        // Initialize each state with a unique sequence based on the index
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel to fill an array with standard normal random numbers (for CRN)
__global__ void fillZ_kernel(float* d_Z, curandState_t* states, int numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curandState_t state = states[idx];
        d_Z[idx] = curand_normal(&state);
        states[idx] = state;
    }
}

// Modified Monte Carlo kernel to use precomputed Z values (CRN)
__global__ void monteCarloCallPrice_kernel_CRN(
    float* d_optionPrices,
    const float* d_Z, // Precomputed Z array
    float S0,
    const float* d_K_batch,
    const float* d_T_batch,
    const float* d_r_batch,
    const float* d_v_batch,
    int numOptions,
    int numPaths)
{
    int optionIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (optionIdx < numOptions) {
        float K = d_K_batch[optionIdx];
        float T = d_T_batch[optionIdx];
        float r = d_r_batch[optionIdx];
        float v = d_v_batch[optionIdx];
        float sumPayoffs = 0.0f;
        float drift = (r - 0.5f * v * v) * T;
        float diffusion = v * sqrtf(T);
        for (int i = 0; i < numPaths; ++i) {
            float Z = d_Z[i]; // Use precomputed Z
            float ST = S0 * expf(drift + diffusion * Z);
            float payoff = fmaxf(ST - K, 0.0f);
            sumPayoffs += payoff;
        }
        float avgPayoff = sumPayoffs / static_cast<float>(numPaths);
        float price = expf(-r * T) * avgPayoff;
        d_optionPrices[optionIdx] = price;
    }
}

// --- Calculation Function (Handles a single S0) ---
// Modified to use CRN
float calculate_portfolio_value_for_S0_CRN(
    float S0,
    float* d_optionPrices,
    std::vector<float>& h_optionPrices,
    const float* d_K_batch,
    const float* d_T_batch,
    const float* d_r_batch,
    const float* d_v_batch,
    const float* d_Z // Precomputed Z array
    )
{
    int threadsPerBlockMC = 256;
    int blocksPerGridMC = (TOTAL_OPTIONS + threadsPerBlockMC - 1) / threadsPerBlockMC;
    size_t optionParamsSize = TOTAL_OPTIONS * sizeof(float);
    monteCarloCallPrice_kernel_CRN<<<blocksPerGridMC, threadsPerBlockMC>>>(
        d_optionPrices, d_Z, S0,
        d_K_batch, d_T_batch, d_r_batch, d_v_batch,
        TOTAL_OPTIONS, NUM_PATHS);
    cudaMemcpy(h_optionPrices.data(), d_optionPrices, optionParamsSize, cudaMemcpyDeviceToHost);
    float portfolioValue = std::accumulate(h_optionPrices.begin(), h_optionPrices.end(), 0.0f);
    return portfolioValue;
}

// Running all steps in the main function
int main() {
    std::cout << std::fixed << std::setprecision(2); // Format output

    // Populate input data on the CPU

    // We will valuate the portfolio for all T (and v) values
    // We will do this by populating the vectors with the correct values
    // and then copying them to the GPU

    std::vector<float> h_K_batch(TOTAL_OPTIONS);
    std::vector<float> h_T_batch(TOTAL_OPTIONS);
    std::vector<float> h_r_batch(TOTAL_OPTIONS);
    std::vector<float> h_v_batch(TOTAL_OPTIONS);

    int currentIdx = 0;
    for (const auto& marketParams : marketData) { // For all T (and v) values
        for (int i = 0; i < NUM_OPTIONS_PER_T; ++i) {
            h_K_batch[currentIdx] = START_STRIKE + static_cast<float>(i) * STRIKE_STEP; // Strikes 50 to 149
            h_T_batch[currentIdx] = marketParams.T;
            h_r_batch[currentIdx] = marketParams.r;
            h_v_batch[currentIdx] = marketParams.v;
            currentIdx++;
        }
    }

    // Allocate memory on the GPU

    // We will allocate memory for the strike prices, time to maturity, risk-free rate, and volatility
    // We will also allocate memory for the option prices and the cuRAND states

    float *d_K_batch, *d_T_batch, *d_r_batch, *d_v_batch;
    float *d_optionPrices;

    size_t optionParamsSize = TOTAL_OPTIONS * sizeof(float);
    
    cudaMalloc(&d_K_batch, optionParamsSize);
    cudaMalloc(&d_T_batch, optionParamsSize);
    cudaMalloc(&d_r_batch, optionParamsSize);
    cudaMalloc(&d_v_batch, optionParamsSize);
    cudaMalloc(&d_optionPrices, optionParamsSize); // For results

    // Host buffer for results
    std::vector<float> h_optionPrices(TOTAL_OPTIONS);

    // Copy static data to GPU
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(d_K_batch, h_K_batch.data(), optionParamsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_batch, h_T_batch.data(), optionParamsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_batch, h_r_batch.data(), optionParamsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_batch, h_v_batch.data(), optionParamsSize, cudaMemcpyHostToDevice);

    // Allocate memory for precomputed Z array
    float* d_Z;
    cudaMalloc(&d_Z, NUM_PATHS * sizeof(float));

    // Map to store results: (Stock Price -> Portfolio Value)

    std::map<int, float> portfolioResults;

    

    // Compute portfolio value for each required scenario using CRN

    // Part a): S0 = 100
    float S0_a = 100.0f;
    float value_a = calculate_portfolio_value_for_S0_CRN(
        S0_a,
        d_optionPrices,
        h_optionPrices,
        d_K_batch,
        d_T_batch,
        d_r_batch,
        d_v_batch,
        d_Z
    );
    portfolioResults[static_cast<int>(S0_a)] = value_a;

    // Part b): S0 from 99 down to 95
    for (float S0_b = 99.0f; S0_b >= 95.0f; S0_b -= 1.0f) {
        float value_b = calculate_portfolio_value_for_S0_CRN(
            S0_b,
            d_optionPrices,
            h_optionPrices,
            d_K_batch,
            d_T_batch,
            d_r_batch,
            d_v_batch,
            d_Z
        );
        portfolioResults[static_cast<int>(S0_b)] = value_b;
    }

    // Part c): S0 from 101 up to 105
    for (float S0_c = 101.0f; S0_c <= 105.0f; S0_c += 1.0f) {
        float value_c = calculate_portfolio_value_for_S0_CRN(
            S0_c,
            d_optionPrices,
            h_optionPrices,
            d_K_batch,
            d_T_batch,
            d_r_batch,
            d_v_batch,
            d_Z
        );
        portfolioResults[static_cast<int>(S0_c)] = value_c;
    }

    auto t2 = high_resolution_clock::now();

    // Calculate elapsed time
    auto duration = duration_cast<microseconds>(t2 - t1);

    // Free GPU Memory
    cudaFree(d_K_batch);
    cudaFree(d_T_batch);
    cudaFree(d_r_batch);
    cudaFree(d_v_batch);
    cudaFree(d_optionPrices);
    cudaFree(d_Z);

    // Write results to Console for all parts (it shows a cute usual graph for an option portfolio: as S0 increases, the portfolio value increases)

    std::cout << "Stock price      Portfolio Value" << std::endl;
    for (const auto& result : portfolioResults) { // Map iterates in key order (95 to 105)
        std::cout << result.first << "                           " << result.second << std::endl;
    }
    std::cout << std::endl;

    // Show Time
    std::cout << "elapsed time = " << duration.count() << " (microseconds)" << std::endl;

    return 0;
}

// Compilation command:
// nvcc -O3 -lcurand final-exam.cu -o final-exam
