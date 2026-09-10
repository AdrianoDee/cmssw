#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CADnnBank_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CADnnBank_h

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Selects which compiled-in weight bank an in-kernel DNN evaluates. The CA
  // plugin is compiled once per topology and shared by both reco iterations, so
  // the bank is a runtime kernel argument (promptHighPt -> kPrompt, else kDisplaced).
  enum class DnnBank : int { kDisplaced = 0, kPrompt = 1 };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CADnnBank_h
