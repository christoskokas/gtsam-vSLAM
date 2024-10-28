#include "KeyFrame.h"

namespace GTSAM_VIOSLAM
{

void KeyFrame::updatePose(const Eigen::Matrix4d& keyPose)
{
    const Eigen::Matrix4d newPose = keyPose * pose.refPose;
    const Eigen::Matrix4d newPoseInv = newPose.inverse();
    const Eigen::Matrix4d& currPoseInv = pose.poseInverse;
    const Eigen::Matrix4d newPoseR = newPose * extr;
    const Eigen::Matrix4d newPoseRInv = newPoseR.inverse();

    int idx {0};
    for ( std::vector<MapPoint*>::iterator it = localMapPoints.begin(), end(localMapPoints.end()); it != end; it++, idx++)
    {
        MapPoint* mp = *it;
        if ( !mp || mp->GetIsOutlier() )
            continue;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        if ( mp->kdx == numb )
        {
            p4d = newPose * (currPoseInv * p4d);
            mp->setWordPose4d(p4d);
        }
        else if ( mp->kdx < numb )
        {
            p4d = newPoseInv * p4d;
            const cv::KeyPoint& obs = keys.keyPoints[idx];
            const int oct = obs.octave;
            const double invZ = 1.0/p4d(2);
            const double u {fx*p4d(0)*invZ + cx};
            const double v {fy*p4d(1)*invZ + cy};
            const double err1 = (double)obs.pt.x - u;
            const double err2 = (double)obs.pt.y - v;
            const double weight = (double)InvSigmaFactor[oct];
            const double err = ((err1*err1) + (err2*err2)) * weight;
            if ( err > 7.815f)
            {
                (*it) = nullptr;
                unMatchedF[idx] = -1;
                mp->eraseKFConnection(this);
            }
        }
    }
    idx = 0;
    for ( std::vector<MapPoint*>::iterator it = localMapPointsR.begin(), end(localMapPointsR.end()); it != end; it++, idx++)
    {
        MapPoint* mp = *it;
        if ( !mp || mp->GetIsOutlier() )
            continue;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        if ( mp->kdx == numb )
            continue;
        else if ( mp->kdx < numb )
        {
            p4d = newPoseRInv * p4d;
            const cv::KeyPoint& obs = keys.rightKeyPoints[idx];
            const int oct = obs.octave;
            const double invZ = 1.0/p4d(2);
            const double u {fx*p4d(0)*invZ + cx};
            const double v {fy*p4d(1)*invZ + cy};
            const double err1 = (double)obs.pt.x - u;
            const double err2 = (double)obs.pt.y - v;
            const double weight = (double)InvSigmaFactor[oct];
            const double err = ((err1*err1) + (err2*err2)) * weight;
            if ( err > 7.815f)
            {
                (*it) = nullptr;
                unMatchedFR[idx] = -1;
                mp->eraseKFConnection(this);
            }
        }
    }
    pose.changePose(keyPose);
}

void KeyFrame::getConnectedKFsLC(std::shared_ptr<Map> map, std::vector<KeyFrame*>& activeKF)
{
    for ( int32_t i{map->LCCandIdx - 1}; i >= 0; i--)
    {
        KeyFrame* kFLCCand = map->keyFrames.at(i);
        activeKF.emplace_back(kFLCCand);
    }
}

void KeyFrame::getConnectedKFs(std::vector<KeyFrame*>& activeKF, const int N)
{
    int count {1};
    for ( std::vector<std::pair<int,KeyFrame*>>::const_iterator it = sortedKFWeights.begin(), end = sortedKFWeights.end(); it != end; it++)
    {
        const std::pair<int,KeyFrame*>& conn = *it;
        if ( conn.second != this )
        {
            activeKF.emplace_back(conn.second);
            count++;
        }
        if ( count >= N )
            break;
    }
}

void KeyFrame::calcConnections()
{
    std::unordered_map<KeyFrame*, int> connWeights;
    for (std::vector<MapPoint*>::const_iterator it = localMapPoints.begin(), end = localMapPoints.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatches.begin(), kfend = mp->kFMatches.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            connWeights[kfCand] ++;
        }
    }

    for (std::vector<MapPoint*>::const_iterator it = localMapPointsR.begin(), end = localMapPointsR.end(); it != end; it++)
    {
        MapPoint* mp = *it;
        if ( !mp )
            continue;
        for (std::unordered_map<KeyFrame*, std::pair<int,int>>::const_iterator kf = mp->kFMatches.begin(), kfend = mp->kFMatches.end(); kf != kfend; kf++)
        {
            KeyFrame* kfCand = kf->first;
            const std::pair<int,int>& keyPos = kf->second;
            if ( keyPos.first >= 0 || keyPos.second < 0 )
                continue;
            connWeights[kfCand] ++;
        }
    }

    const int threshold = 15;
    std::vector<std::pair<int,KeyFrame*>> orderedConn;
    orderedConn.reserve(connWeights.size());
    for (std::unordered_map<KeyFrame*, int>::const_iterator it = connWeights.begin(), end(connWeights.end()); it != end; it ++)
    {
        KeyFrame* kfCand = it->first;
        int weight = it->second;
        if ( weight >= threshold )
            orderedConn.emplace_back(weight, kfCand);
    }
    std::sort(orderedConn.rbegin(), orderedConn.rend());
    sortedKFWeights = orderedConn;
}

void KeyFrame::eraseMPConnection(const std::pair<int,int>& mpPos)
{
    if ( mpPos.first >= 0 )
        eraseMPConnection(mpPos.first);
    if ( mpPos.second >= 0 )
        eraseMPConnectionR(mpPos.second);
}

void KeyFrame::eraseMPConnection(const int mpPos)
{
    localMapPoints[mpPos] = nullptr;
    unMatchedF[mpPos] = -1;
}

void KeyFrame::eraseMPConnectionR(const int mpPos)
{
    localMapPointsR[mpPos] = nullptr;
    unMatchedFR[mpPos] = -1;
}

KeyFrame::KeyFrame(Eigen::Matrix4d _pose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx) : numb(_numb), frameIdx(_frameIdx)
{
    pose.setPose(_pose);
    leftIm = _leftIm.clone();
    rLeftIm = rLIm.clone();
}

KeyFrame::KeyFrame(const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx) : numb(_numb), frameIdx(_frameIdx)
{
    pose.refPose = _refPose;
    pose.setPose(realPose);
    leftIm = _leftIm.clone();
    rLeftIm = rLIm.clone();
}

KeyFrame::KeyFrame(std::shared_ptr<StereoCamera> _zedCam, const Eigen::Matrix4d& _refPose, const Eigen::Matrix4d& realPose, cv::Mat& _leftIm, cv::Mat& rLIm, const int _numb, const int _frameIdx) : numb(_numb), frameIdx(_frameIdx)
{
    pose.refPose = _refPose;
    pose.setPose(realPose);
    leftIm = _leftIm.clone();
    rLeftIm = rLIm.clone();
    fx = _zedCam->mCameraLeft->fx;
    fy = _zedCam->mCameraLeft->fy;
    cx = _zedCam->mCameraLeft->cx;
    cy = _zedCam->mCameraLeft->cy;
    extr = _zedCam->extrinsics;
}

Eigen::Vector4d KeyFrame::getWorldPosition(int idx)
{
    return pose.pose * homoPoints3D.row(idx).transpose();
}

Eigen::Matrix4d KeyFrame::getPose()
{
    return pose.pose;
}

} // namespace GTSAM_VIOSLAM