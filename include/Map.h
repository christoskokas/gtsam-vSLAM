#ifndef MAP_H
#define MAP_H

#include "Camera.h"
#include "KeyFrame.h"
#include "FeatureMatcher.h"
#include "Settings.h"
#include <fstream>
#include <string>
#include <iostream>
#include <random>
#include <unordered_map>

namespace GTSAM_VIOSLAM
{

class KeyFrame;

class MapPoint
{
    private:

    public:

        Eigen::Vector4d wp;
        Eigen::Vector3d wp3d;
        int unMCnt {0};
        std::vector<cv::KeyPoint> obs;

        // last observation
        cv::KeyPoint lastObsL;
        cv::KeyPoint lastObsR;
        KeyFrame* lastObsKF;

        // the descriptor
        cv::Mat desc;

        // with which keyframes the mappoint has matched
        std::unordered_map<KeyFrame*, std::pair<int,int>> kFMatches;

        // if it was triangulated for Monocular 
        bool monoInitialized {false};

        // BA variables
        int LBAID {-1};
        int LCID {-1};

        float maxScaleDist, minScaleDist;

        bool isActive {true};

        bool inFrame {true};
        bool inFrameR {true};
        bool isOutlier {false};
        bool added {false};

        cv::Point2f predL, predR;
        float predAngleL, predAngleR;

        int scaleLevel {0};
        int prdScaleLevel {0};
        int scaleLevelL {0};
        int scaleLevelR {0};

        int keyFrameNb {0};
        const unsigned long kdx;
        const unsigned long idx;

        void update(KeyFrame* kF);
        void update(KeyFrame* kF, const bool back);
        int predictScale(float dist);

        // Connections with keyframe
        void addConnection(KeyFrame* kF, const std::pair<int,int>& keyPos);
        void addConnectionMono(KeyFrame* kF, const std::pair<int,int>& keyPos);
        void eraseKFConnection(KeyFrame* kF);

        void setActive(bool act);
        void SetInFrame(bool infr);
        void SetIsOutlier(bool isOut);
        bool getActive() const;
        bool GetIsOutlier() const;
        bool GetInFrame() const;

        // calculate a robust descriptor. Get all matched descriptors and choose the closest between them (median)
        void calcDescriptor();
        MapPoint(const Eigen::Vector4d& p, const cv::Mat& _desc, const cv::KeyPoint& obsK, const unsigned long _kdx, const unsigned long _idx);

        Eigen::Vector4d getWordPose4d() const;
        Eigen::Vector3d getWordPose3d() const;
        void updatePos(const Eigen::Vector3d& newPos, const std::shared_ptr<StereoCamera> zedPtr);
        void setWordPose4d(const Eigen::Vector4d& p);
        void setWordPose3d(const Eigen::Vector3d& p);
};

class Map
{
    private:

    public:


        bool endOfFrames {false};

        std::unordered_map<unsigned long, KeyFrame*> keyFrames;
        std::unordered_map<unsigned long, MapPoint*> mapPoints;
        std::vector<MapPoint*> activeMapPoints;
        std::vector<KeyFrame*> allFramesPoses;

        // keyframe index
        unsigned long kIdx {0};
        // mappoint index
        unsigned long pIdx {0};
        
        bool keyFrameAdded {false};
        bool keyFrameAddedMain {false};
        bool frameAdded {false};
        bool LBADone {false};
        int endLBAIdx {0};

        
        Eigen::Matrix4d LCPose = Eigen::Matrix4d::Identity();
        bool LCDone {false};
        bool LCStart {false};
        int LCCandIdx {-1};
        int endLCIdx {0};


        Map(){};
        void addMapPoint(MapPoint* mp);
        void addKeyFrame(KeyFrame* kF);
        void removeKeyFrame(int idx);

        // map mutex to ensure multithread safety
        mutable std::mutex mapMutex;

    protected:
};

} // namespace GTSAM_VIOSLAM

#endif // MAP_H