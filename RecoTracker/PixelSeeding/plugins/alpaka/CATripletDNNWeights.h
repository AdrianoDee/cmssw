#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h

// Dispatcher for the per-iteration triplet-DNN weight banks:
//   - CATripletDNNWeights_displaced.h : namespace caTripletDNN_displaced (default bank)
//   - CATripletDNNWeights_prompt.h    : namespace caTripletDNN_prompt
// The bank is chosen at runtime (DnnBank, CADnnBank.h); caTripletDNN_eval::score<BANK>() is specialised per bank.
// The `caTripletDNN` alias is the displaced bank, for code reading kNFeat/kDefaultThreshold without a bank.

#include "CATripletDNNWeights_displaced.h"
#include "CATripletDNNWeights_prompt.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTripletDNN = caTripletDNN_displaced;  // default bank alias

  static_assert(caTripletDNN_prompt::kNFeat == caTripletDNN_displaced::kNFeat &&
                    caTripletDNN_prompt::kNH1 == caTripletDNN_displaced::kNH1 &&
                    caTripletDNN_prompt::kNH2 == caTripletDNN_displaced::kNH2,
                "prompt and displaced triplet-DNN banks must share the network shape");
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
