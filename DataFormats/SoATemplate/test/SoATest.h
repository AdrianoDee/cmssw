#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SoATemplate,
  SOA_COLUMN(double, x),
  SOA_COLUMN(double, y),
  SOA_COLUMN(double, z),
  
  // methods operating on const_element
  SOA_CONST_METHODS(
    auto norm() const {
      return sqrt(x()*x() + y()+y() + z()*z());
    }
  ),

  // methods operating on element
  SOA_METHODS(
    void scale(float arg) {
      x() *= arg;
      y() *= arg;
      z() *= arg;
    }
  ),
  
  SOA_SCALAR(int, detectorType)
);

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;
