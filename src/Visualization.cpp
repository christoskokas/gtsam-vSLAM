#include "Visualization.h"

namespace TII
{


void Visualizer::GetCameraFrame(std::shared_ptr<glk::ThinLines>& lines)
{
  const float w = cameraWidth;
  const float h = w * 0.75;
  const float z = w * 0.3;

  // Define the vertices for the lines
  std::vector<Eigen::Vector3f> vertices = {
      {0.0f, 0.0f, 0.0f}, {w, h, z},
      {0.0f, 0.0f, 0.0f}, {w, -h, z},
      {0.0f, 0.0f, 0.0f}, {-w, -h, z},
      {0.0f, 0.0f, 0.0f}, {-w, h, z},
      {w, h, z}, {w, -h, z},
      {-w, h, z}, {-w, -h, z},
      {-w, h, z}, {w, h, z},
      {-w, -h, z}, {w, -h, z}
  };

  // Set whether to draw as a line strip or individual lines
  bool line_strip = false;
  // Create the lines with the given vertices
  lines = std::make_shared<glk::ThinLines>(vertices, line_strip);

  lines->set_line_width(2.0f);

  // auto transformation = mStereoCamera->mCameraPose.getPose();
  // // Set the color using guik::FlatColor
  // auto line_settings = guik::FlatColor({1.0f, 1.0f, 1.0f, 1.0f}, transformation); 

  // // Add to the viewer
  // mViewer->update_drawable("camera_frame", lines, line_settings);
  // const Eigen::Vector3f cameraPos(transformation(0,3), transformation(1,3), transformation(2,3));
  // mViewer->lookat(cameraPos);
}

void Visualizer::LineFromKeyFrameToCamera()
{

}

void Visualizer::DrawCamera(const Eigen::Matrix4d& cameraPose, const Eigen::Vector4f& color, const std::string& cameraName)
{
  std::shared_ptr<glk::ThinLines> lines;
  GetCameraFrame(lines);
  auto transformation = cameraPose;
  auto line_settings = guik::FlatColor( color, transformation); 

  // Add to the viewer
  mViewer->update_drawable(cameraName, lines, line_settings);
}

void Visualizer::DrawPoints()
{
  
  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector4d> colors;

  
  std::unordered_map<unsigned long, MapPoint*> mapMapP = mMap->mapPoints;
  std::unordered_map<unsigned long, MapPoint*>::const_iterator itw, endw(mapMapP.end());
  points.reserve(mapMapP.size());
  colors.reserve(mapMapP.size());
  for ( itw = mapMapP.begin(); itw != endw; itw ++)
  {
      if ( !(*itw).second )
          continue;
      if ( (*itw).second->GetIsOutlier() )
          continue;

      if ( (*itw).second->getActive() )
      {
        colors.emplace_back(0.0,1.0,0.0,1.0);
      }
      else
      {
        colors.emplace_back(1.0,1.0,0.0,1.0);
      }

      points.emplace_back((*itw).second->wp3d(0),(*itw).second->wp3d(1),(*itw).second->wp3d(2));
  }

  auto cloudBuffer = std::make_shared<glk::PointCloudBuffer>(points);
  cloudBuffer->add_color(colors);

  auto shaderSetting = guik::VertexColor();
  mViewer->update_drawable("points", cloudBuffer, shaderSetting);
}

void Visualizer::DrawKeyFrames()
{
  const int lastKeyFrameIdx {(int)mMap->kIdx - 1};
    if (lastKeyFrameIdx < 0)
        return;

    std::unordered_map<unsigned long, KeyFrame*> mapKeyF = mMap->keyFrames;
    std::unordered_map<unsigned long,KeyFrame*>::const_iterator it, end(mapKeyF.end());
    for ( it = mapKeyF.begin(); it != end; it ++)
    {

        if (!(*it).second->visualize)
            continue;
        Eigen::Matrix4d keyPose = (*it).second->getPose();
        DrawCamera(keyPose, {0.0f, 0.0f, 1.0f, 1.0f}, "keyframe_" + std::to_string(it->second->numb));

    }
}

Visualizer::Visualizer(std::shared_ptr<StereoCamera> stereoCamera, std::shared_ptr<Map> map) : mStereoCamera(stereoCamera), mMap(map)
{
  
}

void Visualizer::RenderScene() {
  
  mViewer = guik::LightViewer::instance();

  // float angle = 0.0f;

  mViewer->disable_xy_grid();

  mViewer->register_ui_callback("ui", [&]() {
    // In the callback, you can call ImGui commands to create your UI.
    // Here, we use "DragFloat" and "Button" to create a simple UI.
    // ImGui::DragFloat("Angle", &angle, 0.01f);

    // if (ImGui::Button("Close")) 
    // {
    //   mViewer->close();
    // }
  });
  
  // Spin the viewer until it gets closed
  while (mViewer->spin_once()) 
  {
    DrawPoints();
    DrawCamera(mStereoCamera->mCameraPose.getPose(), {1.0f, 1.0f, 0.0f, 1.0f}, "Current Camera");
    DrawKeyFrames();

    const Eigen::Vector3f cameraPos(mStereoCamera->mCameraPose.getPose().block<3,1>(0,3).cast<float>());
    mViewer->lookat(cameraPos);
  }
  
}

} // namespace TII