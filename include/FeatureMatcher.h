#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include "Settings.h"
#include "Camera.h"
#include "Map.h"
#include "FeatureExtractor.h"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

namespace GTSAM_VIOSLAM
{

class Map;
class MapPoint;
class KeyFrame;

class FeatureMatcher
{
    private:
        const int thDist {75};
        const int matchDist {50};
        const int matchDistProj {100};
        const float ratioProj {0.8};
        const int matchDistLBA {50};
        const float ratioLBA {0.6};


        std::shared_ptr<StereoCamera> zedptr{nullptr};

    public:
        const int closeNumber {40};
        std::shared_ptr<FeatureExtractor> feLeft{nullptr}, feRight{nullptr};
        const int imageHeight;

        FeatureMatcher(std::shared_ptr<StereoCamera> _zed, std::shared_ptr<FeatureExtractor> _feLeft, std::shared_ptr<FeatureExtractor> _feRight, const int _imageHeight = 360);

        int matchByProjectionRPredLBA(const KeyFrame* lastKF, KeyFrame* newKF, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, const float rad, const std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints, const std::vector<float>& maxDistsScale, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d);

        // destribute right keypoints to vector for faster stereo matching
        void destributeRightKeys(const std::vector < cv::KeyPoint >& rightKeys, std::vector<std::vector < int > >& indexes);

        // get matches inside radius
        void getMatchIdxs(const cv::Point2f& predP, std::vector<int>& idxs, const TrackedKeys& keysLeft, const int predictedScale, const float radius, bool right);

        // find descriptor distance
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        // find stereo matches
        void findStereoMatchesORB2R(const cv::Mat& lImage, const cv::Mat& rImage, const cv::Mat& rightDesc,  std::vector<cv::KeyPoint>& rightKeys, TrackedKeys& keysLeft);

        // match by projection ( frame to frame )
        int matchByProjectionRPred(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, const float rad);

        // match Features for Mono( frame to frame )
        int matchByProjectionMono(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const float rad);

        // match Features for Mono Initialization By Radius( frame to frame )
        int matchByRadius(TrackedKeys& lastKeys, TrackedKeys& actKeys, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const float rad, std::vector<std::vector<std::pair<KeyFrame*,int>>>& keyframeIdxMatchs, KeyFrame* actKeyFrame);

};

} // namespace GTSAM_VIOSLAM

#endif // FEATUREMATCHER_H