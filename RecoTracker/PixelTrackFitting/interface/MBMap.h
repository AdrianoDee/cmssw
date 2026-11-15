#include <fstream>
#include <sstream>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

  struct MBMapHost {
    std::vector<float> values;
    int iRMin, iZMin, nR, nZ;
    float cellSize;
  };

  inline MBMapHost loadMBMap(const std::string& fileName, const float cellSize = 5.f) {
    std::ifstream file(edm::FileInPath(fileName).fullPath());
    if (!file.is_open())
      throw cms::Exception("FileNotFound") << "Cannot open MB map " << fileName;

    struct Row { float r, z, value; };
    std::vector<Row> rows;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream iss(line);
      Row row;
      if (!(iss >> row.r >> row.z >> row.value))
        throw cms::Exception("BadFormat") << "Bad line in MB map: " << line;
      rows.push_back(row);
    }
    if (rows.empty())
      throw cms::Exception("BadFormat") << "MB map " << fileName << " is empty";

    // cell indices of the first and last cell
    int iRMin = INT_MAX, iRMax = INT_MIN, iZMin = INT_MAX, iZMax = INT_MIN;
    for (auto const& row : rows) {
      const int iR = std::lround(row.r / cellSize);
      const int iZ = std::lround(row.z / cellSize);
      iRMin = std::min(iRMin, iR);
      iRMax = std::max(iRMax, iR);
      iZMin = std::min(iZMin, iZ);
      iZMax = std::max(iZMax, iZ);
    }

    MBMapHost map{{}, iRMin, iZMin, iRMax - iRMin + 1, iZMax - iZMin + 1, cellSize};
    map.values.assign(map.nR * map.nZ, 0.f);  // cells missing in the file get 0
    for (auto const& row : rows)
      map.values[(std::lround(row.r / cellSize) - iRMin) * map.nZ + (std::lround(row.z / cellSize) - iZMin)] =
          row.value;
    return map;
  }