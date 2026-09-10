#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNN_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNN_h

// Inline MLP with compile-time weights, evaluated per triplet in Kernel_connect and gated by
// useTripletDNN. The weights come from the per-iteration banks in CATripletDNNWeights_*.h.

#include <alpaka/alpaka.hpp>
#include <cmath>

#include "CADnnBank.h"  // DnnBank (per-iteration weight-bank selector)
#include "CATripletDNNWeights.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTripletDNN_eval {

    // Returns P(real) in [0,1] from the requested bank.
    template <DnnBank BANK>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float score(const float* feat) {
      if constexpr (BANK == DnnBank::kPrompt) {
        using namespace caTripletDNN_prompt;
        float xn[kNFeat];
        for (int i = 0; i < kNFeat; ++i)
          xn[i] = (feat[i] - kMean[i]) / kScale[i];
        float h1[kNH1];
        for (int j = 0; j < kNH1; ++j) {
          float s = kB1[j];
          for (int i = 0; i < kNFeat; ++i)
            s += xn[i] * kW1[i * kNH1 + j];  // kW1 row-major [kNFeat][kNH1]
          h1[j] = s > 0.f ? s : 0.f;         // relu
        }
        float h2[kNH2];
        for (int k = 0; k < kNH2; ++k) {
          float s = kB2[k];
          for (int j = 0; j < kNH1; ++j)
            s += h1[j] * kW2[j * kNH2 + k];  // kW2 row-major [kNH1][kNH2]
          h2[k] = s > 0.f ? s : 0.f;         // relu
        }
        float o = kB3[0];
        for (int k = 0; k < kNH2; ++k)
          o += h2[k] * kW3[k];              // kW3 [kNH2][1]
        return 1.f / (1.f + std::exp(-o));  // sigmoid
      } else {
        using namespace caTripletDNN_displaced;
        float xn[kNFeat];
        for (int i = 0; i < kNFeat; ++i)
          xn[i] = (feat[i] - kMean[i]) / kScale[i];
        float h1[kNH1];
        for (int j = 0; j < kNH1; ++j) {
          float s = kB1[j];
          for (int i = 0; i < kNFeat; ++i)
            s += xn[i] * kW1[i * kNH1 + j];  // kW1 row-major [kNFeat][kNH1]
          h1[j] = s > 0.f ? s : 0.f;         // relu
        }
        float h2[kNH2];
        for (int k = 0; k < kNH2; ++k) {
          float s = kB2[k];
          for (int j = 0; j < kNH1; ++j)
            s += h1[j] * kW2[j * kNH2 + k];  // kW2 row-major [kNH1][kNH2]
          h2[k] = s > 0.f ? s : 0.f;         // relu
        }
        float o = kB3[0];
        for (int k = 0; k < kNH2; ++k)
          o += h2[k] * kW3[k];              // kW3 [kNH2][1]
        return 1.f / (1.f + std::exp(-o));  // sigmoid
      }
    }

  }  // namespace caTripletDNN_eval
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
