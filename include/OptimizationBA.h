#ifndef OPTIMIZATIONBA_H
#define OPTIMIZATIONBA_H


#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Eigen/Dense"
#include <fstream>
#include <string>
#include <iostream>
#include <random>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/triangulation.h>

namespace GTSAM_VIOSLAM
{


class LocalMapper
{
    private:

    public:

        bool stopRequested {false};

        // the map
        std::shared_ptr<Map> map;

        // reprojection threshold to find outliers
        const float reprjThreshold {7.815f};

        const int actvKFMaxSize {10};
        const int minCount {3};

        std::shared_ptr<StereoCamera> zedPtr;

        std::shared_ptr<FeatureMatcher> fm;
        const double fx,fy,cx,cy;

        LocalMapper(std::shared_ptr<Map> _map, std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureMatcher> _fm);

        // triangulate new points from connected keyframes
        void findNewPoints(std::vector<KeyFrame *>& activeKF);
        
        // find all keypoints that have a stereo match for triangulating new points
        void calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale);

        // predict the mps position on the connected KFs
        void predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);

        // local BA optimization
        void localBA(std::vector<KeyFrame *>& actKeyF);

        // check the reprojection error between matched keypoints (from triangulation)
        bool checkReprojError(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix4d>& observationPoses, const std::vector<Eigen::Vector2d>& pointsVec);

        // set the ordering for the GTSAM Optimization
        void setOrdering(gtsam::Ordering& ordering, const std::vector<int>& localKFNumbs, const std::vector<int>& mpNumbs);
        
        // add optimized mappoints to vector for insertion to the map
        void addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos);

        // add the optimized mappoints to the map
        void addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs);

        
        // check if mappoint is outlier
        bool checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);
        // check if mappoint is outlier
        bool checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);

        // local BA check
        void beginLocalMapping();
        bool triangulateNewPoints(Eigen::Vector4d& p4d, KeyFrame* lastKF, std::vector<std::pair<KeyFrame*,std::pair<int, int>>>& matchesOfPoint);

};



} // namespace GTSAM_VIOSLAM


#endif // OPTIMIZATIONBA_H