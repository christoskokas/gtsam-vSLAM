#ifndef OPTIMIZATIONBA_H
#define OPTIMIZATIONBA_H


#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Optimizer.h"
#include "Eigen/Dense"
#include <fstream>
#include <string>
#include <iostream>
#include <random>

namespace TII
{


class LocalMapper
{
    private:

    public:

        bool stopRequested {false};


        std::shared_ptr<Map> map;

        const float reprjThreshold {7.815f};

        const int actvKFMaxSize {10};
        const int minCount {3};

        std::shared_ptr<StereoCamera> zedPtr;
        std::shared_ptr<StereoCamera> zedPtrB;

        std::shared_ptr<FeatureMatcher> fm;
        const double fx,fy,cx,cy;

        LocalMapper(std::shared_ptr<Map> _map, std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureMatcher> _fm);

        // loop closure optimization
        // void loopClosureR(std::vector<KeyFrame *>& actKeyF);

        // loop closure check
        // void beginLoopClosure();

        // insert MPs from localMapPoints for optimization
        void insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back);
        void insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back);
        
        // triangulate new points from connected keyframes
        void triangulateNewPointsR(std::vector<KeyFrame *>& activeKF);
        
        // find all keypoints that have a stereo match for triangulating new points
        void calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale);

        // predict the mps position on the connected KFs
        void predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints);

        // local BA optimization
        void localBAR(std::vector<KeyFrame *>& actKeyF);

        // check the reprojection error between matched keypoints (from triangulation)
        bool checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec);

        
        // calculate projection matrices for triangulation
        void calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF);

        // process matches to find the optimized 3D position of the mappoint
        void processMatchesR(std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points);

        // add optimized mappoints to vector for insertion to the map
        void addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos);

        // add the optimized mappoints to the map
        void addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs);

        
        // check if mappoint is outlier
        bool checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);
        bool checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh);

        // local BA check
        void beginLocalMapping();
        bool triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first, std::vector<Eigen::Matrix4d>& activePoses);

};



} // namespace TII


#endif // OPTIMIZATIONBA_H