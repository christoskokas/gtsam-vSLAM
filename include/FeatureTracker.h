#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Conversions.h"
#include "Settings.h"
#include <fstream>
#include <string>
#include <iostream>
#include <random>
#include <gtsam/navigation/ImuBias.h>

namespace GTSAM_VIOSLAM
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

        // estimated Pose
        Eigen::Matrix4d poseEst = Eigen::Matrix4d::Identity();
        // predicted next Pose
        Eigen::Matrix4d predNPose = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d predNPoseInv = Eigen::Matrix4d::Identity();
        // reference of the predicted next pose (pose from last KF to this one in case of local BA )
        Eigen::Matrix4d predNPoseRef = Eigen::Matrix4d::Identity();

        const int actvKFMaxSize {10};
        const int minNStereo {80};
        const int maxAddedStereo {100};
        const int minInliers {50};

        // MONO Params
        bool secondKF {false};
        int numOfMonoMPs {0};

        float precCheckMatches {0.9f};

        int lastKFTrackedNumb {0};

        double fxb,fyb,cxb,cyb;

        const int keyFrameCountEnd {5};
        int insertKeyFrameCount {0};
        int curFrame {0};

        ImageData lIm, rIm;
        std::shared_ptr<StereoCamera> zedPtr{nullptr};
        std::shared_ptr<FeatureExtractor> feLeft{nullptr};
        std::shared_ptr<FeatureExtractor> feRight{nullptr};
        std::shared_ptr<Map> map{nullptr};
        FeatureMatcher fm;
        const double fx,fy,cx,cy;

        // the active mappoints at the current frame
        std::vector<MapPoint*>& activeMapPoints;
        std::vector<KeyFrame*>& allFrames;
        std::shared_ptr<IMUData> currentIMUData;

        gtsam::imuBias::ConstantBias initialBias;
        gtsam::Vector3 predVelocity {0,0,0};

        bool monoInitialized {false};

    public :

        FeatureTracker(std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureExtractor> _feLeft, std::shared_ptr<FeatureExtractor> _feRight, std::shared_ptr<Map> _map);

        // predict next pose with IMU
        Eigen::Matrix4d PredictNextPoseIMU();

        // main tracking function
        void TrackImage(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr = nullptr);

        // main tracking function Monocular
        void TrackImageMonoIMU(const cv::Mat& leftRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr = nullptr);

        // extract orb features, stereo match them and assign them to grids
        void extractORBAndStereoMatch(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft);

        // Initialize map with 3D mappoints
        void initializeMap(TrackedKeys& keysLeft);

        // Keep the first Keyframe
        void initializeMono(TrackedKeys& keysLeft);

        // set Active 3D mappoints as outliers
        void setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<std::pair<int,int>>& matchesIdxs);

        // remove mappoints that are out of frame
        void removeOutOfFrameMPs(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);
        void removeOutOfFrameMPsMono(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints);

        // 3d world coords to frame coords
        bool worldToFrame(MapPoint* mp, const bool right, const Eigen::Matrix4d& predPoseInv, const Eigen::Matrix4d& tempPose);

        // pose estimation ( GTSAM )
        std::pair<int,int> estimatePoseGTSAM(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);

        // pose estimation for MonoCular ( GTSAM )
        std::pair<int,int> estimatePoseGTSAMMono(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first);


        // check for outliers after pose estimation using reprojection error
        int findOutliersR(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, const std::vector<float>& weights, int& nInliers);
        int findOutliersMono(const Eigen::Matrix4d& estimatedP, std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, const double thres, std::vector<bool>& MPsOutliers, const std::vector<float>& weights, int& nInliers);

        // predict position of 3d mappoints with predicted camera pose
        void PredictMPsPosition(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers);

        // insert KF if needed
        void insertKeyFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const int nStereo, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm);
        void insertKeyFrameMono(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm);


        // check 2d Error
        bool check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const double weight);

        // change camera poses after either Local BA or Global BA
        void changePosesLCA(const int endIdx);

        // publish camera pose
        void updatePoses();

        // add frame if not KF
        void addFrame(const Eigen::Matrix4d& estimPose);

        // assign features to grids for faster matching
        void assignKeysToGrids(TrackedKeys& keysLeft, std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<std::vector<int>>>& keyGrid, const int width, const int height);

        // draw tracked keypoints ( TODO move this to the visual thread )
        void drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys);

        // Monocular functions

        // Add mappoints to the map using triangulation between KFs
        void addMappointsMono(std::vector<MapPoint*>& pointsToAdd, std::vector<KeyFrame *>& actKeyF, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs);

        // calculate 3D point using gtsam::triangulatePoint3 
        bool calculateMPFromMono(Eigen::Vector4d& p4d, std::vector<MapPoint*> pointsToAdd, std::vector<std::pair<KeyFrame*,int>>& keys);
        // add the new mappoints to the map
        void addNewMapPoints(std::vector<MapPoint*>& pointsToAdd);

};



} // namespace GTSAM_VIOSLAM


#endif // FEATURETRACKER_H