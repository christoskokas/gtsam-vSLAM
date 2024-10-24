#ifndef CAMERA_H
#define CAMERA_H

#include "Settings.h"
#include <thread>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace TII
{

  class CameraPose
  {
      private:
      public:
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d refPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseInverse = Eigen::Matrix4d::Identity();
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

        CameraPose(Eigen::Matrix4d _pose = Eigen::Matrix4d::Identity(), std::chrono::time_point<std::chrono::high_resolution_clock> _timestamp = std::chrono::high_resolution_clock::now());

        // set camera Pose
        void setPose(const Eigen::Matrix4d& poseT);
        void setPose(Eigen::Matrix4d& _refPose, Eigen::Matrix4d& _keyPose);

        // get pose
        Eigen::Matrix4d getPose() const;
        Eigen::Matrix4d getInvPose() const;

        // change pose using reference psoe
        void changePose(const Eigen::Matrix4d& _keyPose);

        // set inv pose from local/global BA
        void setInvPose(const Eigen::Matrix4d poseT);
  };

  struct IMUData
  {
    IMUData(double gyroNoiseDensity, double gyroRandomWalk, double accelNoiseDensity, double accelRandomWalk, int hz);
    const double mGyroNoiseDensity, mGyroRandomWalk, mAccelNoiseDensity, mAccelRandomWalk;
    const int mHz;
    std::vector<Eigen::Vector3d> mAngleVelocity, mAcceleration;
    std::vector<double> mTimestamps;


  };

  class Camera
  {

    public:
    Camera(const std::shared_ptr<ConfigFile> configFile, const std::string& camPath);
    const std::shared_ptr<ConfigFile> mConfigFile;
    void setIntrinsicValuesUnR(const std::string& cameraPath);
    void setIntrinsicValuesR(const std::string& cameraPath);
    cv::Mat D = cv::Mat::zeros(1,5,CV_64F);
    cv::Mat K = cv::Mat::eye(3,3,CV_64F);
    cv::Mat R = cv::Mat::eye(3,3,CV_64F);
    cv::Mat P = cv::Mat::eye(3,4,CV_64F);

    std::shared_ptr<IMUData> mIMUData;

    double fx {},fy {},cx {}, cy {};
    Eigen::Matrix<double,3,3> intrinsics = Eigen::Matrix<double,3,3>::Identity();
    private:
    bool rectified {};
    double k1 {}, k2 {}, p1 {}, p2 {}, k3{};
    
  };

  class StereoCamera
  {
    public:
    bool addKeyFrame {false};
    bool rectified {};
    float mBaseline, mFps;
    int mWidth, mHeight;
    size_t numOfFrames {};


    const std::shared_ptr<ConfigFile> mConfigFile;
    std::shared_ptr<Camera> mCameraLeft;
    std::shared_ptr<Camera> mCameraRight;
    CameraPose mCameraPose;


    Eigen::Matrix<double,4,4> extrinsics = Eigen::Matrix<double,4,4>::Identity();
    Eigen::Matrix<double,4,4> TCamToCam = Eigen::Matrix<double,4,4>::Identity();
    Eigen::Matrix<double,4,4> TCamToCamInv = Eigen::Matrix<double,4,4>::Identity();

    void setCameraValues(const std::string& camPath);
    
    public:
    StereoCamera(const std::shared_ptr<ConfigFile> configFile, std::shared_ptr<Camera> cameraLeft, std::shared_ptr<Camera> cameraRight);
    StereoCamera() {}

  };

} // namespace TII


#endif // CAMERA_H
