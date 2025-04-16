#ifndef BLADE_BENCHMARK_KURTOSIS_GENERIC_HH
#define BLADE_BENCHMARK_KURTOSIS_GENERIC_HH

#include "blade/modules/kurtosis.hh"

#include "../../helper.hh"

#include <random>

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class KurtosisTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;

    int nants = 28;
    int nchans = 192;
    int nsamps = 8192;
    int npols = 2; 

    Result run(benchmark::State& state) {
        // const U64 A = state.range(20);
        // const U8 M = state.range(1);

        InitAndProfile([&](){
            // config.inputPolarization = POL::XY;
            // config.outputPolarization = static_cast<POL>(M);
            config.blockSize = 192;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({nants, nchans, nsamps, npols}, true);
            //memset(&deviceInputBuf, 100, 28 * 192 * 8192 * 2);
            // printf("%.5f %.5f\n", deviceInputBuf[0].real(), deviceInputBuf[0].imag());

            //BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf, 
            }, this->getStream());
            //BL_ENABLE_PRINT();
        }, state);

        for (auto _ : state) {
            if (!Profiler::IsCapturing()) {
                for (int i = 0; i < nants; i++) {
                    for (int j = 0; j < nchans; j++) {
                        for (int k = 0; k < nsamps; k++) {
                            deviceInputBuf[{i, j, k, 0}] = std::complex<float>(100.0f, 100.0f);
                            deviceInputBuf[{i, j, k, 1}] = std::complex<float>(100.0f, 100.0f);
                            //deviceInputBuf[i * 10 * 256 * 2 + j * 256 * 2 + k * 2 + 0] = std::complex<float>(100.0f, 100.0f);
                            //deviceInputBuf[i * 10 * 256 * 2 + j * 256 * 2 + k * 2 + 1] = std::complex<float>(100.0f, 100.0f);
                        }
                    }
                }
            }
            BL_CHECK(this->startIteration());
            BL_CHECK(module->process(0, this->getStream()));
            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
