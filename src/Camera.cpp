#include "Camera.h"
#include "Settings.h"

namespace GTSAM_VIOSLAM
{

CameraPose::CameraPose(Eigen::Matrix4d _pose, std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp) : pose(_pose), timestamp(_timestamp)
{}

void CameraPose::setPose(const Eigen::Matrix4d& poseT)
{
    pose = poseT;
    poseInverse = poseT.inverse();
    timestamp = std::chrono::high_resolution_clock::now();
}

Eigen::Matrix4d CameraPose::getPose() const
{
    return pose;
}

Eigen::Matrix4d CameraPose::getInvPose() const
{
    return poseInverse;
}

void CameraPose::setPose(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose)
{
    refPose = _refPose;
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::changePose(const Eigen::Matrix4d& _keyPose)
{
    Eigen::Matrix4d truePose = _keyPose * refPose;
    setPose(truePose);
}

void CameraPose::setInvPose(const Eigen::Matrix4d poseT)
{
    poseInverse = poseT;
    pose = poseT.inverse();
}

StereoCamera::StereoCamera(std::shared_ptr<ConfigFile> configFile, std::shared_ptr<Camera> cameraLeft, std::shared_ptr<Camera> cameraRight) : mConfigFile(configFile), mCameraLeft(cameraLeft), mCameraRight(cameraRight)
{
  setCameraValues("Camera");
}

void StereoCamera::setCameraValues(const std::string& camPath)
{
  mWidth = mConfigFile->getValue<int>(camPath,"width");
  mHeight = mConfigFile->getValue<int>(camPath,"height");
  mFps = mConfigFile->getValue<float>(camPath,"fps");
  mBaseline = mConfigFile->getValue<float>(camPath ,"bl");
  extrinsics(0,3) = (double)mBaseline;
}

IMUData::IMUData(double gyroNoiseDensity, double gyroRandomWalk, double accelNoiseDensity, double accelRandomWalk, int hz) : mGyroNoiseDensity(gyroNoiseDensity), mGyroRandomWalk(gyroRandomWalk), mAccelNoiseDensity(accelNoiseDensity), mAccelRandomWalk(accelRandomWalk), mHz(hz)
{

}

Camera::Camera(std::shared_ptr<ConfigFile> configFile, const std::string& camPath) : mConfigFile(configFile), mIMUData(nullptr)
{
  rectified = configFile->getValue<bool>("rectified");
  if (rectified)
    setIntrinsicValuesR(camPath);
  else
    setIntrinsicValuesUnR(camPath);
}

void Camera::setIntrinsicValuesUnR(const std::string& cameraPath)
{
    fx = mConfigFile->getValue<double>(cameraPath,"fx");
    fy = mConfigFile->getValue<double>(cameraPath,"fy");
    cx = mConfigFile->getValue<double>(cameraPath,"cx");
    cy = mConfigFile->getValue<double>(cameraPath,"cy");
    k1 = mConfigFile->getValue<double>(cameraPath,"k1");
    k2 = mConfigFile->getValue<double>(cameraPath,"k2");
    p1 = mConfigFile->getValue<double>(cameraPath,"p1");
    p2 = mConfigFile->getValue<double>(cameraPath,"p2");
    k3 = mConfigFile->getValue<double>(cameraPath,"k3");
    std::vector < double > Rt = mConfigFile->getValue<std::vector<double>>(cameraPath,"R","data");
    std::vector < double > Pt = mConfigFile->getValue<std::vector<double>>(cameraPath,"P","data");
    std::vector < double > Dt = mConfigFile->getValue<std::vector<double>>(cameraPath,"D","data");
    std::vector < double > Kt = mConfigFile->getValue<std::vector<double>>(cameraPath,"K","data");

    R = (cv::Mat_<double>(3,3) << Rt[0], Rt[1], Rt[2], Rt[3], Rt[4], Rt[5], Rt[6], Rt[7], Rt[8]);
    P = (cv::Mat_<double>(3,4) << Pt[0], Pt[1], Pt[2], Pt[3], Pt[4], Pt[5], Pt[6], Pt[7], Pt[8], Pt[9], Pt[10], Pt[11]);
    K = (cv::Mat_<double>(3,3) << Kt[0], Kt[1], Kt[2], Kt[3], Kt[4], Kt[5], Kt[6], Kt[7], Kt[8]);
    D = (cv::Mat_<double>(1,5) << Dt[0], Dt[1], Dt[2], Dt[3], Dt[4]);

    intrinsics(0,0) = fx;
    intrinsics(1,1) = fy;
    intrinsics(0,2) = cx;
    intrinsics(1,2) = cy;

}

void Camera::setIntrinsicValuesR(const std::string& cameraPath)
{

    fx = mConfigFile->getValue<double>(cameraPath,"fx");
    fy = mConfigFile->getValue<double>(cameraPath,"fy");
    cx = mConfigFile->getValue<double>(cameraPath,"cx");
    cy = mConfigFile->getValue<double>(cameraPath,"cy");
    k1 = 0;
    k2 = 0;
    p1 = 0;
    p2 = 0;
    k3 = 0;

    intrinsics(0,0) = fx;
    intrinsics(1,1) = fy;
    intrinsics(0,2) = cx;
    intrinsics(1,2) = cy;
}
} // namespace GTSAM_VIOSLAM