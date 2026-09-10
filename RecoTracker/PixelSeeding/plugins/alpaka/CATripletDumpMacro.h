#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h

// Toggle for the built-triplet training-dataset dump (one row per built triplet carrying the
// in-kernel triplet feature vector). Off in production: every dump-related branch is #ifdef'd on
// it. Kept in a minimal header so producer, generator and kernels see the toggle without pulling
// in CATripletCuts.h.
// #define CA_TRIPLET_DUMP

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h
