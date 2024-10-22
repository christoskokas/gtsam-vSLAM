#ifndef SYSTEM_H
#define SYSTEM_H

#include "Camera.h"
#include "Map.h"
#include "Visualization.h"
#include "FeatureTracker.h"
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
      VSlamSystem(const std::shared_ptr<ConfigFile> configFile, SlamMode mode = SlamMode::STEREO);
      void InitializeMonocular();
      void InitializeStereo();
      void GetStereoCamera(std::shared_ptr<StereoCamera>& stereoCamera);
      void StartSystem();
      void TrackMonoCular();
      void TrackStereo();
      void ExitSystem();
      void SaveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition);
    private:

    std::thread mFeatureTrackingThread;
    std::thread mFeatureExtractLeftThread;
    std::thread mFeatureExtractRightThread;
    std::thread mOptimizerThread;
    std::thread mVisualizationThread;

    std::shared_ptr<StereoCamera> mStereoCamera {};
    std::shared_ptr<Camera> mMonoCamera {};
    std::shared_ptr<Visualizer> mVisualizer;
    std::shared_ptr<ConfigFile> mConfigFile;
    std::shared_ptr<FeatureTracker> mFeatureTracker;
    std::shared_ptr<FeatureExtractor> mFeatureExtractorLeft;
    std::shared_ptr<FeatureExtractor> mFeatureExtractorRight;
    std::shared_ptr<Map> mMap;
    
    SlamMode mMode;
  };

} // namespace TII


#endif // SYSTEM_H
