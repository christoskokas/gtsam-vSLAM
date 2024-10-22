#ifndef SYSTEM_H
#define SYSTEM_H

#include "Camera.h"
#include "Visualization.h"
#include <thread>
#include <string>

namespace TII
{

  class VSlamSystem
  {

    enum class SlamMode
    {
      STEREO,
      MONOCULAR
    };

    public:
      void GetStereoCamera(std::shared_ptr<StereoCamera>& stereoCamera);
      void StartSystem();
      void TrackMonoCular();
      void TrackStereo();
      void ExitSystem();
      void SaveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition);
      VSlamSystem(const std::shared_ptr<ConfigFile> configFile, SlamMode mode = SlamMode::STEREO);
    private:

    std::thread mFeatureTrackingThread;
    std::thread mOptimizerThread;
    std::thread mVisualizationThread;

    std::shared_ptr<StereoCamera> mStereoCamera {};
    std::shared_ptr<Camera> mMonoCamera {};
    std::shared_ptr<Visualizer> mVisualizer;
    std::shared_ptr<ConfigFile> mConfigFile;
    
    SlamMode mMode;
  };

} // namespace TII


#endif // SYSTEM_H
