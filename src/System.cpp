#include "System.h"

namespace TII
{

VSlamSystem::VSlamSystem(const std::shared_ptr<ConfigFile> configFile, SlamMode mode /* = SlamMode::STEREO*/) : mConfigFile(configFile), mMode(mode)
{

  mMap = std::make_shared<Map>();

  if (mode == SlamMode::MONOCULAR)
  {
    InitializeMonocular();
  }
  else
  {
    InitializeStereo();
  }



  
  // mFeatureTracker = std::shared_ptr<FeatureTracker>();
  mFeatureTrackingThread = {};
  mOptimizerThread = {};
  mVisualizer = std::make_shared<Visualizer>(mStereoCamera, mMap);
  mVisualizationThread = std::thread(&Visualizer::RenderScene, mVisualizer);
}

void VSlamSystem::InitializeMonocular()
{
  mMonoCamera = std::make_shared<Camera>(mConfigFile, "Camera_l");
  mFeatureExtractorLeft = std::make_shared<FeatureExtractor>();
  // mFeatureTracker = std::shared_ptr<FeatureTracker>();
  std::cout << "Monocular Camera Initialized.." << std::endl;
}

void VSlamSystem::InitializeStereo()
{
  auto cameraLeft = std::make_shared<Camera>(mConfigFile, "Camera_l");
  auto cameraRight = std::make_shared<Camera>(mConfigFile, "Camera_r");
  mStereoCamera = std::make_shared<StereoCamera>(mConfigFile, cameraLeft, cameraRight);
  mFeatureExtractorLeft = std::make_shared<FeatureExtractor>();
  mFeatureExtractorRight = std::make_shared<FeatureExtractor>();
  mFeatureTracker = std::make_shared<FeatureTracker>(mStereoCamera, mFeatureExtractorLeft, mFeatureExtractorRight, mMap);
  std::cout << "Stereo Camera Initialized.." << std::endl;
}

void VSlamSystem::GetStereoCamera(std::shared_ptr<StereoCamera>& stereoCamera)
{
  stereoCamera = mStereoCamera;
}

void VSlamSystem::StartSystem()
{
  // if (mMode == SlamMode::MONOCULAR)
  //   TrackMonoCular();
  // else
  //   TrackStereo();
    
}

void VSlamSystem::ExitSystem()
{
  mFeatureTrackingThread.join();
  mOptimizerThread.join();
  mVisualizationThread.join();
}

void VSlamSystem::TrackMonoCular()
{

}

void VSlamSystem::TrackStereo(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb)
{
  mFeatureTracker->TrackImageT(imLRect, imRRect, frameNumb);
}

void VSlamSystem::TrackStereoIMU(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb, const IMUData& IMUDataVal)
{
  mFeatureTracker->TrackImageT(imLRect, imRRect, frameNumb, std::make_shared<IMUData>(IMUDataVal));
}

void VSlamSystem::SaveTrajectoryAndPosition(const std::string& filepath, const std::string& filepathPosition)
{
    // std::vector<KeyFrame*>& allFrames = map->allFramesPoses;
    // KeyFrame* closeKF = allFrames[0];
    // std::ofstream datafile(filepath);
    // std::ofstream datafilePos(filepathPosition);
    // std::vector<KeyFrame*>::iterator it;
    // std::vector<KeyFrame*>::const_iterator end(allFrames.end());
    // for ( it = allFrames.begin(); it != end; it ++)
    // {
    //     KeyFrame* candKF = *it;
    //     Eigen::Matrix4d matT;
    //     if ( candKF->keyF )
    //     {
    //         matT = candKF->pose.pose;
    //         closeKF = candKF;
    //     }
    //     else
    //     {
    //         matT = (closeKF->pose.getPose() * candKF->pose.refPose);
    //     }
    //     Eigen::Matrix4d mat = matT.transpose();
    //     for (int32_t i{0}; i < 12; i ++)
    //     {
    //         if ( i == 0 )
    //             datafile << mat(i);
    //         else
    //             datafile << " " << mat(i);
    //         if ( i == 3 || i == 7 || i == 11 )
    //             datafilePos << mat(i) << " ";
    //     }
    //     datafile << '\n';
    //     datafilePos << '\n';
    // }
    // datafile.close();
    // datafilePos.close();

}

} // namespace TII