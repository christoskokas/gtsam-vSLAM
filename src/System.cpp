#include "System.h"

namespace GTSAM_VIOSLAM
{

// VSlamSystem::VSlamSystem(std::shared_ptr<ConfigFile> configFile, SlamMode mode /* = SlamMode::STEREO*/)
// {

//   // mMap = std::make_shared<Map>();

//   // if (mode == SlamMode::MONOCULAR)
//   // {
//   //   InitializeMonocular();
//   // }
//   // else
//   // {
//   //   InitializeStereo();
//   //   mLocalMapper = std::make_shared<LocalMapper>(mMap, mStereoCamera, mFeatureMatcher);
//   //   mOptimizerThread = std::thread(&LocalMapper::beginLocalMapping, mLocalMapper);
//   // }

  
//   // mVisualizer = std::make_shared<Visualizer>(mStereoCamera, mMap);
//   // mVisualizationThread = std::thread(&Visualizer::RenderScene, mVisualizer);
// }

void VSlamSystem::InitializeMonocular()
{
  mMonoCamera = std::make_shared<Camera>(mConfigFile, "Camera_l");
  mStereoCamera = std::make_shared<StereoCamera>(mConfigFile, mMonoCamera, nullptr);
  mFeatureExtractorLeft = std::make_shared<FeatureExtractor>();
  mFeatureMatcher = std::make_shared<FeatureMatcher>(mStereoCamera, mFeatureExtractorLeft, mFeatureExtractorLeft);
  mFeatureTracker = std::make_shared<FeatureTracker>(mStereoCamera, mFeatureExtractorLeft, mFeatureExtractorRight, mMap);
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
}

void VSlamSystem::GetStereoCamera(std::shared_ptr<StereoCamera>& stereoCamera)
{
  stereoCamera = mStereoCamera;
}

void VSlamSystem::ExitSystem()
{
  mFeatureTrackingThread.join();
  mOptimizerThread.join();
  mVisualizationThread.join();
}

void VSlamSystem::TrackStereo(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb)
{
  mFeatureTracker->TrackImage(imLRect, imRRect, frameNumb);
}

void VSlamSystem::TrackStereoIMU(const cv::Mat& imLRect, const cv::Mat& imRRect, const int frameNumb, const IMUData& IMUDataVal)
{
  mFeatureTracker->TrackImage(imLRect, imRRect, frameNumb, std::make_shared<IMUData>(IMUDataVal));
}

void VSlamSystem::TrackMonoIMU(const cv::Mat& imLRect, const int frameNumb, const IMUData& IMUDataVal)
{
  mFeatureTracker->TrackImageMonoIMU(imLRect, frameNumb, std::make_shared<IMUData>(IMUDataVal));
}

} // namespace GTSAM_VIOSLAM