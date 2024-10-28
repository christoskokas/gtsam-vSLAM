#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "Settings.h"
#include "Camera.h"
#include "Map.h"
#include <thread>
#include <string>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <Eigen/Dense>
#include <glk/pointcloud_buffer.hpp>
#include <glk/thin_lines.hpp>

namespace GTSAM_VIOSLAM
{

  class Visualizer
  {
    public:

    const float cameraWidth = 0.1575f;
    guik::LightViewer* mViewer;

    std::shared_ptr<StereoCamera> mStereoCamera{nullptr};

    std::shared_ptr<Map> mMap{nullptr};

    void GetCameraFrame(std::shared_ptr<glk::ThinLines>& lines);
    void LineFromKeyFrameToCamera();
    void DrawCamera(const Eigen::Matrix4d& cameraPose, const Eigen::Vector4f& color, const std::string& cameraName, float width);
    void DrawPoints();
    void DrawKeyFrames();

    Visualizer(std::shared_ptr<StereoCamera> stereoCamera, std::shared_ptr<Map> map);
    void RenderScene();
  };

} // namespace GTSAM_VIOSLAM


#endif // VISUALIZATION_H
