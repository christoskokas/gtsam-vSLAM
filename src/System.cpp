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
    mLocalMapper = std::make_shared<LocalMapper>(mMap, mStereoCamera, mFeatureMatcher);
  }

  
  // mFeatureTracker = std::shared_ptr<FeatureTracker>();
  mFeatureTrackingThread = {};
  mOptimizerThread = std::thread(&LocalMapper::beginLocalMapping, mLocalMapper);;
  mVisualizer = std::make_shared<Visualizer>(mStereoCamera, mMap);
  mVisualizationThread = std::thread(&Visualizer::RenderScene, mVisualizer);
}

void VSlamSystem::InitializeMonocular()
{
  mMonoCamera = std::make_shared<Camera>(mConfigFile, "Camera_l");
  mFeatureExtractorLeft = std::make_shared<FeatureExtractor>();
  mFeatureMatcher = std::make_shared<FeatureMatcher>(mStereoCamera, mFeatureExtractorLeft, mFeatureExtractorLeft);
  // mFeatureTracker = std::shared_ptr<FeatureTracker>();
  std::cout << "Monocular Camera Initialized.." << std::endl;
}

void VSlamSystem::InitializeStereo()
{
  int nFeatures = mConfigFile->getValue<int>("FE", "nFeatures");
  int nLevels = mConfigFile->getValue<int>("FE", "nLevels");
  float imScale = mConfigFile->getValue<float>("FE", "imScale");
  int edgeThreshold = mConfigFile->getValue<int>("FE", "edgeThreshold");
  int maxFastThreshold = mConfigFile->getValue<int>("FE", "maxFastThreshold");
  int minFastThreshold = mConfigFile->getValue<int>("FE", "minFastThreshold");
  int patchSize = mConfigFile->getValue<int>("FE", "patchSize");

  auto cameraLeft = std::make_shared<Camera>(mConfigFile, "Camera_l");
  auto cameraRight = std::make_shared<Camera>(mConfigFile, "Camera_r");


  std::vector < double > tBIMU = mConfigFile->getValue<std::vector<double>>("T_bc1", "data");

  Eigen::Matrix4d tBodyToImu;
  tBodyToImu = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(tBIMU.data());
  cameraLeft->TBodyToCam = tBodyToImu;
  mStereoCamera = std::make_shared<StereoCamera>(mConfigFile, cameraLeft, cameraRight);
  mFeatureExtractorLeft = std::make_shared<FeatureExtractor>(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);
  mFeatureExtractorRight = std::make_shared<FeatureExtractor>(nFeatures, nLevels, imScale, edgeThreshold, patchSize, maxFastThreshold, minFastThreshold);
  mFeatureMatcher = std::make_shared<FeatureMatcher>(mStereoCamera, mFeatureExtractorLeft, mFeatureExtractorRight);
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