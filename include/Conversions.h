#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>



namespace GTSAM_VIOSLAM
{

struct Converter
{

    static constexpr double parallaxThreshold {0.1f};
    static constexpr double baselineThreshold {0.1f};
    static constexpr double angleThreshold {5.0f};
    static constexpr double pixelParallaxThresh {10.0f};

    static Eigen::Matrix3d convertCVRotToEigen(cv::Mat& Rot)
    {
        cv::Rodrigues(Rot,Rot);
        Eigen::Matrix3d RotEig;
        cv::cv2eigen(Rot, RotEig);
        return RotEig;
    }
    static Eigen::Vector3d convertCVTraToEigen(cv::Mat& Tra)
    {
        Eigen::Vector3d traEig;
        cv::cv2eigen(Tra, traEig);
        return traEig;
    }

    static Eigen::Matrix4d convertRTtoPose(cv::Mat& Rot, cv::Mat& Tra)
    {
        Eigen::Vector3d traEig;
        cv::cv2eigen(Tra, traEig);
        cv::Rodrigues(Rot,Rot);
        Eigen::Matrix3d RotEig;
        cv::cv2eigen(Rot, RotEig);

        Eigen::Matrix4d convPose = Eigen::Matrix4d::Identity();
        convPose.block<3,3>(0,0) = RotEig;
        convPose.block<3,1>(0,3) = traEig;

        return convPose;
    }

    static void convertEigenPoseToMat(const Eigen::Matrix4d& poseToConv, cv::Mat& Rot, cv::Mat& Tra)
    {
        Eigen::Matrix3d RotEig;
        Eigen::Vector3d TraEig;
        RotEig = poseToConv.block<3,3>(0,0);
        TraEig = poseToConv.block<3,1>(0,3);

        cv::eigen2cv(RotEig,Rot);
        cv::eigen2cv(TraEig,Tra);

        cv::Rodrigues(Rot, Rot);
    }

    static Eigen::Matrix<double, 7, 1> Matrix4dToMatrix_7_1(
    const Eigen::Matrix4d& pose) 
    {
        Eigen::Matrix<double, 7, 1> Tcw_7_1;
        Eigen::Matrix3d R;
        R = pose.block<3, 3>(0, 0);
        // Eigen Quaternion coeffs output [x, y, z, w]
        Tcw_7_1.block<3, 1>(0, 0) = pose.block<3, 1>(0, 3);
        Tcw_7_1.block<4, 1>(3, 0) = Eigen::Quaterniond(R).coeffs();
        return Tcw_7_1;
    }

    static Eigen::Matrix4d Matrix_7_1_ToMatrix4d(
    const Eigen::Matrix<double, 7, 1>& Tcw_7_1) 
    {
        Eigen::Quaterniond q(Tcw_7_1[6], Tcw_7_1[3], Tcw_7_1[4], Tcw_7_1[5]);
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = R;
        pose.block<3, 1>(0, 3) = Tcw_7_1.block<3, 1>(0, 0);
        return pose;
    }

    static double calculateParallaxAngle(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2) 
    {
        Eigen::Matrix3d rotation1 = pose1.block<3,3>(0,0);
        Eigen::Matrix3d rotation2 = pose2.block<3,3>(0,0);
        
        Eigen::Matrix3d relativeRotation = rotation1.transpose() * rotation2;
        
        // Compute the angle of rotation 
        double trace = relativeRotation.trace();
        double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
        
        return abs(angle); // Angle in radians
    }

    static bool isParallaxSufficient(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2, double threshold = parallaxThreshold) 
    {
        double parallaxAngle = calculateParallaxAngle(pose1, pose2);
        return parallaxAngle > threshold;
    }

    static bool checkSufficientMovement(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2, double baseline_threshold = baselineThreshold, double angle_threshold = angleThreshold) 
    {
        Eigen::Vector3d t1 = pose1.block<3, 1>(0, 3);
        Eigen::Vector3d t2 = pose2.block<3, 1>(0, 3);

        double baseline = (t2 - t1).norm();

        Eigen::Matrix3d R1 = pose1.block<3, 3>(0, 0);
        Eigen::Matrix3d R2 = pose2.block<3, 3>(0, 0);

        Eigen::Matrix3d relative_rotation = R1.transpose() * R2;
        double cos_theta = (relative_rotation.trace() - 1) / 2.0;

        cos_theta = std::clamp(cos_theta, -1.0, 1.0);

        double rotation_angle = std::acos(cos_theta) * (180.0 / M_PI);


        if (baseline < baseline_threshold) 
            return false;

        if (rotation_angle < angle_threshold) 
            return false;

        return true;
    }

    // Function to compute pixel parallax
    static bool checkPixelParallax(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, double pixel_parallax_thresh = pixelParallaxThresh) 
    {
        double parallax =  (p2 - p1).norm();
        return parallax > pixel_parallax_thresh;
    }

};

} // namespace GTSAM_VIOSLAM


#endif // CONVERSIONS_H