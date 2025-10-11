# Derivatives examples in C++

## Introduction

This is work done mostly for a HPC in C++ and Python course. This repo houses the C++ code (and their Python/Julia implementations for comparison)

## Index

### Option Pricing Models

- **[Black-Scholes European Call](black-scholes-european-call/)** - Implementation of the Black-Scholes-Merton formula for European call option pricing
- **[Binomial Tree](binomial-tree/)** - Binomial tree model for option pricing
- **[Monte Carlo Brownian Motion](monte-carlo-bm/)** - Monte Carlo simulation using Brownian motion for option pricing

### GPU-Accelerated Computing (CUDA)

- **[Monte Carlo CUDA](monte-carlo-cuda/)** - GPU-accelerated Monte Carlo simulation for option pricing using CUDA
- **[Greeks CUDA](greeks-cuda/)** - GPU computation of option Greeks (Delta, Gamma, Theta, etc.)

### Parallel Computing

- **[Multi-threaded Fork-Join](multi-threaded/)** - Implementation of fork-join and map-reduce patterns using std::thread for parallel option pricing

### Mathematical Computations

- **[Matrix Multiplication](matrix-multiplication/)** - Optimized matrix multiplication implementations
- **[Monte Carlo Pi](montecarlo-pi/)** - Monte Carlo method for estimating π

## Technologies Used

- **C++**: Core implementation language
- **CUDA**: GPU acceleration for computationally intensive tasks
- **std::thread**: Multi-threading for parallel processing
