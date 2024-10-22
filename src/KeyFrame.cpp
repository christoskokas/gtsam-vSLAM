#include "Map.h"

namespace TII
{

MapPoint::MapPoint(const Eigen::Vector4d& worldPos)
{
  mWorldPos4d = worldPos;
  mWorldPos3d = {worldPos(0),worldPos(1),worldPos(2)};
}

} // namespace TII