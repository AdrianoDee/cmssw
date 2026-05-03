#ifndef RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h

#include "HeterogeneousCore/AlpakaMath/interface/float16_t.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(PixelTrackFeaturesSoALayout,
                    SOA_COLUMN(cms::float16_t, chi2),
                    SOA_COLUMN(cms::float16_t, dzError),
                    SOA_COLUMN(cms::float16_t, dxyError),
                    SOA_COLUMN(cms::float16_t, eta),
                    SOA_COLUMN(cms::float16_t, nHits),
                    SOA_COLUMN(cms::float16_t, phi),
                    SOA_COLUMN(cms::float16_t, phiError),
                    SOA_COLUMN(cms::float16_t, pt),
                    SOA_COLUMN(cms::float16_t, qOverPtError),
                    SOA_COLUMN(cms::float16_t, dzBS),
                    SOA_COLUMN(cms::float16_t, dxyBS),
                    SOA_COLUMN(cms::float16_t, nLayers),
                    SOA_COLUMN(cms::float16_t, cotThetaError),
                    SOA_COLUMN(cms::float16_t, covCotThetaDz),
                    SOA_COLUMN(cms::float16_t, covDxyQOverPt),
                    SOA_COLUMN(cms::float16_t, covPhiDxy),
                    SOA_COLUMN(cms::float16_t, covPhiQOverPt));

using PixelTrackFeaturesSoA = PixelTrackFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(PixelTrackScoresSoALayout, SOA_COLUMN(cms::float16_t, score))

using PixelTrackScoresSoA = PixelTrackScoresSoALayout<>;

#endif
