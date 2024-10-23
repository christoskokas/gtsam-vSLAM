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

namespace TII
{

class FeatureTracker
{

    public :
        std::shared_ptr<FeatureExtractor> mFeLeft;
        std::shared_ptr<FeatureExtractor> mFeRight;
        std::shared_ptr<FeatureMatcher> mFeatureMatcher;
        std::shared_ptr<StereoCamera> mStereoCamera;
        std::vector<cv::KeyPoint> mKeysLeft;
        std::vector<cv::KeyPoint> mKeysRight;

        cv::Mat mImageLeft, mImageRight;

        FeatureTracker(const cv::Mat& imageLeft, const cv::Mat& imageRight);

};



} // namespace TII


#endif // FEATURETRACKER_H