#include "Visualization.h"

namespace TII
{

Visualizer::Visualizer()
{
  
}

void Visualizer::RenderScene() {
  
  auto viewer = guik::LightViewer::instance();

  float angle = 0.0f;

  viewer->register_ui_callback("ui", [&]() {
    // In the callback, you can call ImGui commands to create your UI.
    // Here, we use "DragFloat" and "Button" to create a simple UI.
    ImGui::DragFloat("Angle", &angle, 0.01f);

    if (ImGui::Button("Close")) 
    {
      viewer->close();
    }
  });
  
  // Spin the viewer until it gets closed
  while (viewer->spin_once()) 
  {
    Eigen::AngleAxisf transform(angle, Eigen::Vector3f::UnitZ());
    viewer->update_drawable("sphere", glk::Primitives::sphere(), guik::Rainbow(transform));
    viewer->update_drawable("wire_sphere", glk::Primitives::wire_sphere(), guik::FlatColor({0.1f, 0.7f, 1.0f, 1.0f}, transform));
  }
  
}

} // namespace TII