# Methodology

The usual MC + Black Scholes: use BSM underlying BM to generate NUM_PATH paths, value them using the call formula and the average them to find the price for a given case. Save all output for each stock price and then print the corresponding portfolio value.

Major difference with respect to the old implementations: here we use CUDA and cuRAND to generate the pseudo-random walks of the price
to then value the option given each path.

# Compile command

nvcc -O3 -lcurand final-exam.cu -o final-exam

# Preliminary results

- On my personal laptop (RTX 3060 67C 55W 6GB VRAM): 0.6 ms