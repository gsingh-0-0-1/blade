#include "blade/memory/base.hh"

using namespace Blade;

// organized by powers of two starting at 8
// eg index 0 is 2^(0 + 8) = 256
// and in ascending order of stddev
float SKLIM_VALS[] = {
    // STD 3, CHUNK 256
    0.698159, 1.49597,
    // STD 3, CHUNK 512
    0.775046, 1.32542,
    // STD 3, CHUNK 1024
    0.834186, 1.21695,

    // STD 4, CHUNK 256
    0.613738, 1.784,
    // STD 4, CHUNK 512
    0.711612, 1.48684,
    // STD 5, CHUNK 1024
    0.786484, 1.31218,

    // STD 5, CHUNK 256
    0.526881, 2.18694,
    // STD 5, CHUNK 512
    0.649093, 1.69044,
    // STD 5, CHUNK 1024
    0.740405, 1.42332
}

// CUDA kernel to compute sk_array
__global__ void computeSkArray(
    comp_float_t* block,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS) {//, int m) {


    // Compute indices
    int ant = blockIdx.x;    // Antenna index
    int chan = blockIdx.y;   // Channel index
    int pol = threadIdx.x;   // Polarization index

    float sklim_lower, sklim_upper;
    // let's assume STDDEV of 5 for now
    if (N_SAMPS == 256) {
        sklim_lower = 0.526881;
        sklim_upper = 2.18694;
    }
    if (N_SAMPS == 512) {
        sklim_lower = 0.649093;
        sklim_upper = 1.69044;
    }
    if (N_SAMPS == 1024) {
        sklim_lower = 0.740405;
        sklim_upper = 1.42332;
    }

    if (ant < N_ANTS && chan < N_CHANS && pol < N_POLS) {
        // Initialize sums
        float s1 = 0.0f;
        float s2 = 0.0f;

        // Compute s1 (sum of elements) and s2 (sum of squares)
        for (int samp = 0; samp < N_SAMPS; samp++) {
            int idx = ((ant * N_CHANS + chan) * N_SAMPS + samp) * N_POLS + pol;
            comp_float_t value = block[idx];

            float v2 = value.real * value.real + value.imag * value.imag;

            s1 += v2;
            s2 += v2 * v2;
        }

        // Compute sk value
        float sk = ((N_SAMPS + 1.0f) / (N_SAMPS - 1.0f)) * ((N_SAMPS * (s2 / (s1 * s1))) - 1.0f);

        // based on sk we can zap the channel
        if (sk > sklim_upper || sk < sklim_lower) {
            int chan_start = ((ant * N_CHANS + chan) * N_SAMPS + 0) * N_POLS + pol;
            for (int j = chan_start; j < chan_start + N_SAMPS * N_POLS; j = j + N_POLS) {
                block[j].real = 0.0;
                block[j].imag = 0.0;
            }
        }

        // Write the result to the output array
        // int out_idx = ((ant * N_CHANS + chan) * 1 + 0) * N_POLS + pol;
        // output[out_idx] = sk;
    }
}

// Host function to call the kernel
void calculateSkArray(
    comp_float_t* d_block,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS) {//, int m) {

    dim3 gridDim(N_ANTS, N_CHANS);    // One block per antenna and channel
    dim3 blockDim(N_POLS);           // One thread per polarization

    computeSkArray<<<gridDim, blockDim>>>(
        d_block, N_ANTS, N_CHANS, N_SAMPS, N_POLS);//, m);
}