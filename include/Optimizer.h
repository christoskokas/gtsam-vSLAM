#ifndef MAP_H
#define MAP_H

#include "Camera.h"
#include <thread>
#include <string>
#include <Eigen/Dense>

namespace TII
{

  class MapPoint
  {
    public:
    MapPoint(const Eigen::Vector4d& worldPos);
    Eigen::Vector4d mWorldPos4d;
    Eigen::Vector3d mWorldPos3d;
  };

  class Map
  {
    CameraPose mCameraPose;
    std::shared_ptr<std::vector<MapPoint>> mMapPoints;
  };

} // namespace TII


#endif // MAP_H
