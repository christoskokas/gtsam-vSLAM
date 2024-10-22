#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <thread>
#include <string>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <Eigen/Dense>

namespace TII
{

  class Visualizer
  {
    public:
    Visualizer();
    void RenderScene();
  };

} // namespace TII


#endif // VISUALIZATION_H
