#ifndef BLADE_BENCHMARK_KURTOSIS_GENERIC_HH
#define BLADE_BENCHMARK_KURTOSIS_GENERIC_HH

#include "blade/modules/kurtosis.hh"

#include "../../helper.hh"

namespace Blade {

template<template<typename, typename> class MUT, typename IT, typename OT>
class KurtosisTest : CudaBenchmark {
 public:
    typename MUT<IT, OT>::Config config;
    std::shared_ptr<MUT<IT, OT>> module;
    ArrayTensor<Device::CUDA, IT> deviceInputBuf;

    Result run(benchmark::State& state) {
        // const U64 A = state.range(20);
        // const U8 M = state.range(1);

        InitAndProfile([&](){
            // config.inputPolarization = POL::XY;
            // config.outputPolarization = static_cast<POL>(M);
            config.blockSize = 512;

            deviceInputBuf = ArrayTensor<Device::CUDA, IT>({20, 192, 256, 2});

            BL_DISABLE_PRINT();
            Create(module, config, {
                .buf = deviceInputBuf, 
            }, this->getStream());
            BL_ENABLE_PRINT();
        }, state);

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            BL_CHECK(module->process(0, this->getStream()));
            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
