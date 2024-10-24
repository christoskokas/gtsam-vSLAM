#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Settings.h"
#include "Camera.h"
#include "FeatureExtractor.h"
#include "Map.h"
#include "opencv2/core.hpp"


namespace TII
{

class Map;
class MapPoint;

class KeyFrame
{
    private:

    public:
        double fx,fy,cx,cy;
        double fxb,fyb,cxb,cyb;
        CameraPose pose;
        Eigen::Matrix4d extr;
        Eigen::Matrix4d TCamToCam;
        cv::Mat leftIm, rightIm;
        cv::Mat rLeftIm;
        std::vector<int> unMatchedF;
        std::vector<int> unMatchedFR;
        std::vector<float> scaleFactor;
        std::vector < float > sigmaFactor;
        std::vector < float > InvSigmaFactor;
        std::unordered_map<KeyFrame*, int> weightsKF;
        std::vector<std::pair<int,KeyFrame*>> sortedKFWeights;
        float logScale;
        int nScaleLev;

        int LBAID {-1};
        int LCID {-1};

        bool LCCand {false};

        std::shared_ptr<IMUData> mIMUData;


        TrackedKeys keys, keysB;
        Eigen::MatrixXd homoPoints3D;
        const unsigned long numb;
        const int frameIdx;
        int nKeysTracked {0};
        bool visualize {true};
        std::vector<MapPoint*> localMapPoints;
        std::vector<MapPoint*> localMapPointsR;
        KeyFrame* prevKF = nullptr;
        KeyFrame* nextKF = nullptr;
        bool active {true};
        bool keyF {false};
        bool LBA {false};
        bool fixed {false};

        void updatePose(const Eigen::Matrix4d& keyPose);

        void calcConnections();


        void eraseMPConnection(const int mpPos);
        void eraseMPConnection(const std::pair<int,int>& mpPos);
        void eraseMPConnectionR(const int mpPos);
        KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        KeyFrame(const std::shared_ptr<StereoCamera> _zedCam, const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx);
        Eigen::Vector4d getWorldPosition(int idx);
        void getConnectedKFs(std::vector<KeyFrame*>& activeKF, const int N);
        void getConnectedKFsLC(std::shared_ptr<Map> map, std::vector<KeyFrame*>& activeKF);

        Eigen::Matrix4d getPose();
};

} // namespace TII

#endif // KEYFRAME_H