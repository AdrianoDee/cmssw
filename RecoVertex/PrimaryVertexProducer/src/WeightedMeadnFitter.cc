// #include "RecoVertex/PrimaryVertexProducer/interface/WeightedMeanFitter.h"
// #include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
// #include "RecoVertex/VertexPrimitives/interface/VertexException.h"
//
// using namespace std;
//
// namespace {
//
//   bool recTrackLessZ1(const DAClusterizerInZ::track_t& tk1, const DAClusterizerInZ::track_t& tk2) {
//     return tk1.z < tk2.z;
//   }
// }  // namespace
//
// TransientVertex weightedMeanOutlierRejectionVarianceAsError(const std::vector<std::pair<GlobalPoint, GlobalPoint>>& points, std::vector<std::vector<reco::TransientTrack>>::const_iterator iclus){
//      float x=0, y=0, z=0, s_wx=0, s_wy=0, s_wz=0, s2_wx=0, s2_wy=0, s2_wz=0, wx=0, wy=0, wz=0, chi2=0;
//      float ndof_x = 0, ndof_y = 0, ndof_z = 0;
//      float precision = 1e-10;
//      AlgebraicSymMatrix33 err;
//      err(0,0) = 2 * 2;
//      err(1,1) = 2 * 2;
//      err(2,2) = 20 * 20; // error is 20 cm, so cov -> is 20 ^ 2
//      for (const auto& p : points){
//             wx = p.second.x();
//             wx = wx <=  precision ? 1. / std::pow(precision,2) : 1. / std::pow(wx,2);
//
//             wz = p.second.z();
//             wz = wz <=  precision ? 1. / std::pow(precision,2) : 1. / std::pow(wz,2);
//
//             x += p.first.x() * wx;
//             y += p.first.y() * wx;
//             z += p.first.z() * wz;
//
//             s_wx += wx;
//             s_wz += wz;
//      }
//
//      if ( s_wx == 0. || s_wz == 0. ){
//         std::cout << "Vertex fitting failed at beginning" << std::endl;
//         return TransientVertex(GlobalPoint(0,0,0), err, *iclus, 0, 0);
//      }
//
//      x /= s_wx;
//      y /= s_wx;
//      z /= s_wz;
//
//     float old_x, old_y, old_z;
//     float xpull, ypull, zpull;
//     int niter = 0;
//     float mu = 3.;
//     float err_x, err_y, err_z;
//     err_x = 1. / std::sqrt(s_wx);
//     err_y = 1. / std::sqrt(s_wx);
//     err_z = 1. / std::sqrt(s_wz);
//     float s_err_x = 0, s_err_y = 0, s_err_z = 0;
//     while ((niter++) < 50){
//         old_x = x;
//         old_y = y;
//         old_z = z;
//         s_wx = 0; s_wy = 0; s_wz = 0; s2_wx = 0; s2_wy = 0; s2_wz = 0;
//         x = 0; y = 0; z = 0;
//         s_err_x = 0.; s_err_y = 0.; s_err_z = 0;
//
//         int xout = 0, zout = 0;
//         for (const auto& p : points){
//             wx = p.second.x();
//             wx =  wx <= precision ? precision : wx;
//
//             wy = wx*wx + err_y*err_y;
//             wx = wx*wx + err_x*err_x;
//
//             wz = p.second.z();
//             wz =  wz <= precision ? precision : wz;
//             wz = wz*wz + err_z*err_z;
//
//             xpull  = std::pow((p.first.x() - old_x), 2) / wx;
//             xpull += std::pow((p.first.y() - old_y), 2) / wy;
//             xpull += std::pow((p.first.z() - old_z), 2) / wz;
//             xpull = 1. / (1. + std::exp(-0.5 * ((mu * mu) - xpull)));
//             ndof_x += xpull;
//
//             /*
//             if (niter==4){
//                 std::cout << "Begin cluster" << std::endl;
//
//             }
//             */
// //            if (std::abs(zpull)> mu ) zout++;
// //            if (std::abs(xpull)> mu ) xout++;
//
// //            err_x += std::pow(1. / (1. + std::exp(-0.5 * ((mu  * mu ) - std::pow(xpull,2)))), 1) ;
// //            err_y += std::pow(1. / (1. + std::exp(-0.5 * ((mu  * mu ) - std::pow(ypull,2)))), 1) ;
// //            err_z += std::pow(1. / (1. + std::exp(-0.5 * ((mu  * mu ) - std::pow(zpull,2)))), 1) ;
//
//
//             wx = 1. / wx;
//             wy = 1. / wy;
//             wz = 1. / wz;
//
//             wx *= xpull;
//             wy *= xpull;
//             wz *= xpull;
//
// //            wx *= 1. / (1. + std::exp(-0.5 * ((mu * mu) - std::pow(xpull,2))));
// //            wy *= 1. / (1. + std::exp(-0.5 * ((mu * mu) - std::pow(xpull,2))));
// //            wz *= 1. / (1. + std::exp(-0.5 * ((mu * mu) - std::pow(xpull,2))));
//
//             x += wx * p.first.x();
//             y += wy * p.first.y();
//             z += wz * p.first.z();
//
// //�            err_x += wx * pow(p.first.x() - old_x, 2);
// //�            err_y += wy * pow(p.first.y() - old_y, 2);
// //�            err_z += wz * pow(p.first.z() - old_z, 2);
//
//             s_wx += wx;
//             s_wy += wy;
//             s_wz += wz;
//
//             s2_wx += wx * wx;
//             s2_wy += wy * wy;
//             s2_wz += wz * wz;
//
//           s_err_x += wx * pow(p.first.x() - old_x, 2);
//           s_err_y += wy * pow(p.first.y() - old_y, 2);
//           s_err_z += wz * pow(p.first.z() - old_z, 2);
//         }
//             //std::cout << "outlier % " << zout << " , " << xout << " , " << points.size() << std::endl;
//         if ( s_wx == 0. || s_wy == 0. || s_wz == 0. ){
//             std::cout << "Vertex fitting failed" << s_wx << " , " << s_wy << " , " << s_wz << std::endl;
//             return TransientVertex(GlobalPoint(0,0,0), err, *iclus, 0, 0);
//         }
//         x /= s_wx;
//         y /= s_wy;
//         z /= s_wz;
//
// //        err_x = std::sqrt(s_err_x / (s_wx - s2_wx / s_wx));
// //        err_y = std::sqrt(s_err_y / (s_wy - s2_wy / s_wy));
// //        err_z = std::sqrt(s_err_z / (s_wz - s2_wz / s_wz));
//
//         err_x = std::sqrt(s_err_x / s_wx );
//         err_y = std::sqrt(s_err_y / s_wy );
//         err_z = std::sqrt(s_err_z / s_wz );
//
//         if (std::abs(x - old_x) < (precision/1.) && std::abs(y - old_y) < (precision/1.) && std::abs(z - old_z) < (precision/1.)){
//
//             break;
//         }
//     }
//
//      err(0,0) = err_x * err_x;
//      err(1,1) = err_y * err_y;
//      err(2,2) = err_z * err_z;
//
//      float dist = 0;
//      for (const auto& p : points){
//         wx = p.second.x();
//         wx =  wx <= precision ? precision : wx;
//
//         wz = p.second.z();
//         wz =  wz <= precision ? precision : wz;
//
//         dist =  std::pow(p.first.x() - x, 2) / ( std::pow(wx, 2) + std::pow( err(0,0), 2) );
//         dist += std::pow(p.first.y() - y, 2) / ( std::pow(wx, 2) + std::pow( err(1,1), 2) );
//         dist += std::pow(p.first.z() - z, 2) / ( std::pow(wz, 2) + std::pow( err(2,2), 2) );
//         chi2 += dist;
//      }
//      TransientVertex v(GlobalPoint(x,y,z), err, *iclus, chi2, (int) ndof_x);
//      return v;
// }
//
