#ifndef SYSTEM_H
#define SYSTEM_H

#include "Camera.h"
#include "Map.h"
#include "Visualization.h"
#include "FeatureTracker.h"
#include "OptimizationBA.h"
#include <thread>
#include <string>

namespace GTSAM_VIOSLAM
{

  class VSlamSystem
  {


    public:
    
      enum class SlamMode
      {
        STEREO_IMU,
        STEREO,
        MONOCULAR
      };

      VSlamSystem(std::shared_ptr<ConfigFile> configFile, SlamMode mode = SlamMode::STEREO);
      void InitializeMonocular();
      void InitializeStereo();
      void GetStereoCamera(std::shared_ptr<StereoCamera>& stereoCamera);
      void TrackMonoIMU(const cv::Mat& imLRect, const int frameNumb, const IMUData& IMUDataVal);
      void TrackStereo(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb);
      void TrackStereoIMU(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb, const IMUData& IMUData);
      void saveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition);
      void ExitSystem();
    private:

    std::thread mFeatureTrackingThread;
    std::thread mFeatureExtractLeftThread;
    std::thread mFeatureExtractRightThread;
    std::thread mOptimizerThread;
    std::thread mVisualizationThread;

    std::shared_ptr<StereoCamera> mStereoCamera {nullptr};
    std::shared_ptr<Camera> mMonoCamera {nullptr};
    std::shared_ptr<Visualizer> mVisualizer {nullptr};
    std::shared_ptr<ConfigFile> mConfigFile {nullptr};
    std::shared_ptr<FeatureTracker> mFeatureTracker {nullptr};
    std::shared_ptr<FeatureExtractor> mFeatureExtractorLeft {nullptr};
    std::shared_ptr<FeatureExtractor> mFeatureExtractorRight {nullptr};
    std::shared_ptr<FeatureMatcher> mFeatureMatcher {nullptr};
    std::shared_ptr<LocalMapper> mLocalMapper {nullptr};
    std::shared_ptr<Map> mMap {nullptr};
    
    SlamMode mMode;
  };

} // namespace GTSAM_VIOSLAM


#endif // SYSTEM_H
