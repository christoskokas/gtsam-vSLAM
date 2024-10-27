#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include "Optimizer.h"
#include <fstream>
#include <string>
#include <iostream>
#include <random>
#include <gtsam/navigation/ImuBias.h>

namespace TII
{

class ImageData
{
    private:

    public:
        cv::Mat im, rIm;
};

class FeatureTracker
{
    private :


        KeyFrame* latestKF = nullptr;
        Eigen::Matrix4d lastKFPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseRef = Eigen::Matrix4d::Identity();

        const int actvKFMaxSize {10};
        const int minNStereo {80};
        const int maxAddedStereo {100};
        const int minInliers {50};

        const double parallaxThreshold {0.01};

        int numOfMonoMPs {0};

        float precCheckMatches {0.9f};

        int lastKFTrackedNumb {0};

        double fxb,fyb,cxb,cyb;

        const int keyFrameCountEnd {5};
        int insertKeyFrameCount {0};
        int curFrame {0};
        int curFrameNumb {-1};

        ImageData pLIm, pRIm, lIm, rIm;
        std::shared_ptr<StereoCamera> zedPtr;
        std::shared_ptr<FeatureExtractor> feLeft;
        std::shared_ptr<FeatureExtractor> feRight;
        std::shared_ptr<Map> map;
        FeatureMatcher fm;
        const double fx,fy,cx,cy;

        std::vector<MapPoint*>& activeMapPoints;
        std::vector<KeyFrame*>& allFrames;
        std::shared_ptr<IMUData> currentIMUData;

        gtsam::imuBias::ConstantBias initialBias;

        bool monoInitialized {false};

    public :

        FeatureTracker(std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureExtractor> _feLeft, std::shared_ptr<FeatureExtractor> _feRight, std::shared_ptr<Map> _map);

        // predict next pose with IMU
        Eigen::Matrix4d PredictNextPoseIMU();

        // main tracking function
        void TrackImageT(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr = nullptr);

        // main tracking function Monocular
        void TrackImageMonoIMU(const cv::Mat& leftRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr = nullptr);

        // extract orb features
        void extractORBStereoMatchR(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft);

        // Initialize map with 3D mappoints
        void initializeMapR(TrackedKeys& keysLeft);

        // Keep the first Keyframe
        void initializeMono(TrackedKeys& keysLeft);

        // set 3D mappoints as outliers
        void setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<std::pair<int,int>>& matchesIdxs);

        // remove mappoints that are out of frame
        void removeOutOfFrameMPsR(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);
        void removeOutOfFrameMPsMono(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);

        // 3d world coords to frame coords
        bool worldToFrameRTrack(MapPoint* mp, const bool right, const Eigen::Matrix4d& predPoseInv, const Eigen::Matrix4d& tempPose);

        // pose estimation ( Ceres Solver )
        std::pair<int,int> estimatePoseCeresR(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);

        // pose estimation ( GTSAM )
        std::pair<int,int> estimatePoseGTSAM(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);

        // pose estimation for MonoCular ( GTSAM )
        std::pair<int,int> estimatePoseGTSAMMono(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);


        // check for outliers after pose estimation
        int findOutliersR(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, const std::vector<float>& weights, int& nInliers);
        int findOutliersMono(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, const std::vector<float>& weights, int& nInliers);

        // predict position of 3d mappoints with predicted camera pose
        void newPredictMPs(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers);

        // insert KF if needed
        void insertKeyFrameR(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const int nStereo, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm);
        void insertKeyFrameMono(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm);

        void addMappointsMono(std::vector<MapPoint*>& pointsToAdd, std::vector<KeyFrame *>& actKeyF, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs);
        bool calculateMPFromMono(Eigen::Vector4d& p4d, std::vector<MapPoint*> pointsToAdd, std::vector<std::pair<KeyFrame*,int>>& keys);
        void addNewMapPoints(std::vector<MapPoint*>& pointsToAdd);

        // Calculate parallax between 4d poses
        double calculateParallaxAngle(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2);
        bool isParallaxSufficient(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2, double threshold = 0.05);

        // check 2d Error
        bool check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const double weight);

        // change camera poses after either Local BA or Global BA
        void changePosesLCA(const int endIdx);

        // publish camera pose
        void publishPoseNew();

        // add frame if not KF
        void addFrame(const Eigen::Matrix4d& estimPose);

        // assign features to grids for faster matching
        void assignKeysToGrids(TrackedKeys& keysLeft, std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<std::vector<int>>>& keyGrid, const int width, const int height);

        // draw tracked keypoints ( TODO move this to the visual thread )
        void drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys);

};



} // namespace TII


#endif // FEATURETRACKER_H