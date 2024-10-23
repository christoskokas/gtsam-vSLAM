#include "FeatureTracker.h"

namespace TII
{

FeatureTracker::FeatureTracker(std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureExtractor> _feLeft, std::shared_ptr<FeatureExtractor> _feRight, std::shared_ptr<Map> _map) : zedPtr(_zedPtr), feLeft(_feLeft), feRight(_feRight), map(_map), fm(zedPtr, _feLeft, _feRight, zedPtr->mHeight), fx(_zedPtr->mCameraLeft->fx), fy(_zedPtr->mCameraLeft->fy), cx(_zedPtr->mCameraLeft->cx), cy(_zedPtr->mCameraLeft->cy), activeMapPoints(_map->activeMapPoints), allFrames(_map->allFramesPoses)
{
    allFrames.reserve(zedPtr->numOfFrames);
}

void FeatureTracker::assignKeysToGrids(TrackedKeys& keysLeft, std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<std::vector<int>>>& keyGrid, const int width, const int height)
{
    const float imageRatio = (float)width/(float)height;
    keysLeft.xGrids = 64;
    keysLeft.yGrids = cvCeil((float)keysLeft.xGrids/imageRatio);
    keysLeft.xMult = (float)keysLeft.xGrids/(float)width;
    keysLeft.yMult = (float)keysLeft.yGrids/(float)height;
    keyGrid = std::vector<std::vector<std::vector<int>>>(keysLeft.yGrids, std::vector<std::vector<int>>(keysLeft.xGrids, std::vector<int>()));
    int kpCount {0};
    for ( std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin(), end(keypoints.end()); it !=end; it ++, kpCount++)
    {
        const cv::KeyPoint& kp = *it;
        int xPos = cvRound(kp.pt.x * keysLeft.xMult);
        int yPos = cvRound(kp.pt.y * keysLeft.yMult);
        if ( xPos < 0 )
            xPos = 0;
        if ( yPos < 0 )
            yPos = 0;
        if ( xPos >= keysLeft.xGrids )
            xPos = keysLeft.xGrids - 1;
        if ( yPos >= keysLeft.yGrids )
            yPos = keysLeft.yGrids - 1;
        if ( keyGrid[yPos][xPos].empty() )
            keyGrid[yPos][xPos].reserve(200);
        keyGrid[yPos][xPos].emplace_back(kpCount);
    }
}

void FeatureTracker::extractORBStereoMatchR(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft)
{
    std::thread extractLeft(&FeatureExtractor::extractKeysNew, std::ref(feLeft), std::ref(leftIm), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc));
    std::thread extractRight(&FeatureExtractor::extractKeysNew, std::ref(feRight), std::ref(rightIm), std::ref(keysLeft.rightKeyPoints),std::ref(keysLeft.rightDesc));
    extractLeft.join();
    extractRight.join();



    fm.findStereoMatchesORB2R(leftIm, rightIm, keysLeft.rightDesc, keysLeft.rightKeyPoints, keysLeft);

    assignKeysToGrids(keysLeft, keysLeft.keyPoints, keysLeft.lkeyGrid, zedPtr->mWidth, zedPtr->mHeight);
    assignKeysToGrids(keysLeft, keysLeft.rightKeyPoints, keysLeft.rkeyGrid, zedPtr->mWidth, zedPtr->mHeight);

}

void FeatureTracker::initializeMapR(TrackedKeys& keysLeft)
{
    KeyFrame* kF = new KeyFrame(zedPtr->mCameraPose.pose, lIm.im, lIm.rIm,map->kIdx, curFrame);
    kF->scaleFactor = feLeft->scalePyramid;
    kF->sigmaFactor = feLeft->sigmaFactor;
    kF->InvSigmaFactor = feLeft->InvSigmaFactor;
    kF->nScaleLev = feLeft->nLevels;
    kF->logScale = log(feLeft->imScale);
    kF->keyF = true;
    kF->fixed = true;
    kF->unMatchedF.resize(keysLeft.keyPoints.size(), -1);
    kF->unMatchedFR.resize(keysLeft.rightKeyPoints.size(), -1);
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
    kF->localMapPointsR.resize(keysLeft.rightKeyPoints.size(), nullptr);
    activeMapPoints.reserve(keysLeft.keyPoints.size());
    kF->keys.getKeys(keysLeft);
    int trckedKeys {0};
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        if ( keysLeft.estimatedDepth[i] > 0 )
        {
            const int rIdx {keysLeft.rightIdxs[i]};
            const double zp = (double)keysLeft.estimatedDepth[i];
            const double xp = (double)(((double)keysLeft.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = zedPtr->mCameraPose.pose * p;
            MapPoint* mp = new MapPoint(p, keysLeft.Desc.row(i), keysLeft.keyPoints[i], map->kIdx, map->pIdx);
            mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kF, std::pair<int,int>(i,rIdx)));
            map->addMapPoint(mp);
            mp->lastObsKF = kF;
            mp->lastObsL = keysLeft.keyPoints[i];
            mp->scaleLevelL = keysLeft.keyPoints[i].octave;
            mp->lastObsR = keysLeft.rightKeyPoints[rIdx];
            mp->scaleLevelR = keysLeft.rightKeyPoints[rIdx].octave;
            mp->update(kF);
            activeMapPoints.emplace_back(mp);
            kF->localMapPoints[i] = mp;
            kF->localMapPointsR[rIdx] = mp;
            kF->unMatchedF[i] = mp->kdx;
            kF->unMatchedFR[rIdx] = mp->kdx;
            trckedKeys++;
        }
    }
    lastKFTrackedNumb = trckedKeys;
    map->addKeyFrame(kF);
    latestKF = kF;
    allFrames.emplace_back(kF);
    Eigen::Matrix4d lastKFPose = zedPtr->mCameraPose.pose;
    lastKFPoseInv = lastKFPose.inverse();
}

bool FeatureTracker::check2dError(Eigen::Vector4d& p4d, const cv::Point2f& obs, const double thres, const double weight)
{
    if ( p4d(2) <= 0 )
        return true;
    const double invZ = 1.0f/p4d(2);

    const double u {fx*p4d(0)*invZ + cx};
    const double v {fy*p4d(1)*invZ + cy};

    const double errorU = ((double)obs.x - u);
    const double errorV = ((double)obs.y - v);

    const double error = (errorU * errorU + errorV * errorV) * weight;
    if (error > thres)
        return true;
    else
        return false;
}

std::pair<int, int> FeatureTracker::estimatePoseGTSAM(std::vector<MapPoint *> &activeMapPoints, TrackedKeys &keysLeft, std::vector<std::pair<int, int>> &matchesIdxs, Eigen::Matrix4d &estimPose, std::vector<bool> &MPsOutliers, const bool first)
{
    return std::pair<int, int>();
}

int FeatureTracker::findOutliersR(const Eigen::Matrix4d &estimPose, std::vector<MapPoint *> &activeMapPoints, TrackedKeys &keysLeft, std::vector<std::pair<int, int>> &matchesIdxs, const double thres, std::vector<bool> &MPsOutliers, const std::vector<float> &weights, int &nInliers)
{
    const Eigen::Matrix4d estimPoseInv = estimPose.inverse();
    const Eigen::Matrix4d toCameraR = (estimPoseInv * zedPtr->extrinsics).inverse();
    int nStereo = 0;
    for (size_t i {0}, end{matchesIdxs.size()}; i < end; i++)
    {
        std::pair<int,int>& keyPos = matchesIdxs[i];
        MapPoint* mp = activeMapPoints[i];
        if ( !mp )
            continue;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        int nIdx;
        cv::Point2f obs;
        bool right {false};
        if ( keyPos.first >= 0 )
        {
            if  ( !mp->inFrame )
                continue;
            p4d = estimPose * p4d;
            nIdx = keyPos.first;
            obs = keysLeft.keyPoints[nIdx].pt;
        }
        else if ( keyPos.second >= 0 )
        {
            if  ( !mp->inFrameR )
                continue;
            right = true;
            p4d = toCameraR * p4d;
            nIdx = keyPos.second;
            obs = keysLeft.rightKeyPoints[nIdx].pt;
        }
        else
            continue;
        const int octL = (right) ? keysLeft.rightKeyPoints[nIdx].octave: keysLeft.keyPoints[nIdx].octave;
        const double weight = (double)feLeft->InvSigmaFactor[octL];
        bool outlier = check2dError(p4d, obs, thres, weight);
        MPsOutliers[i] = outlier;
        if ( !outlier )
        {
            nInliers++;
            if ( p4d(2) < zedPtr->mBaseline * fm.closeNumber && keysLeft.close[nIdx] && !right )
            {
                if ( keyPos.second < 0 )
                    continue;
                Eigen::Vector4d p4dr = toCameraR*mp->getWordPose4d();
                cv::Point2f obsr = keysLeft.rightKeyPoints[keyPos.second].pt;
                const int octR = keysLeft.rightKeyPoints[keyPos.second].octave;
                const double weightR = (double)feLeft->InvSigmaFactor[octR];
                bool outlierr = check2dError(p4dr, obsr, thres, weightR);
                if ( !outlierr )
                    nStereo++;
                else
                {
                    keysLeft.estimatedDepth[nIdx] = -1;
                    keysLeft.close[nIdx] = false;
                    const int rIdx = keysLeft.rightIdxs[nIdx];
                    keysLeft.rightIdxs[nIdx] = -1;
                    keysLeft.leftIdxs[rIdx] = -1;
                    keyPos.second = -1;
                    
                }
            }
        }
    }

    return nStereo;
}

// std::pair<int,int> FeatureTracker::estimatePoseCeresR(std::vector<MapPoint*>& activeMapPoints, TrackedKeys& keysLeft, std::vector<std::pair<int,int>>& matchesIdxs, Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, const bool first)
// {

//     const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
//     const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
//     const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);

//     const size_t prevS { activeMapPoints.size()};

//     const Eigen::Matrix3d& K = zedPtr->mCameraLeft->intrinsics;

//     std::vector<float> weights;
//     weights.resize(prevS, 1.0f);
//     double thresh = 7.815f;

//     size_t maxIter {2};
//     int nIn {0}, nStereo {0};
//     for (size_t iter {0}; iter < maxIter; iter ++ )
//     {
//         ceres::Problem problem;
//         Eigen::Vector3d frame_tcw;
//         Eigen::Quaterniond frame_qcw;
//         Eigen::Matrix4d frame_pose = estimPose;
//         Eigen::Matrix3d frame_R;
//         frame_R = frame_pose.block<3, 3>(0, 0);
//         frame_tcw = frame_pose.block<3, 1>(0, 3);
//         frame_qcw = Eigen::Quaterniond(frame_R);
//         ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
//         ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(7.815f));
//         for (size_t i{0}, end{matchesIdxs.size()}; i < end; i++)
//         {
//             if ( MPsOutliers[i] )
//                 continue;
//             const std::pair<int,int>& keyPos = matchesIdxs[i];

//             MapPoint* mp = activeMapPoints[i];
//             if ( mp->GetIsOutlier() )
//                 continue;
//             ceres::CostFunction* costf;
//             if ( keyPos.first >= 0 )
//             {
//                 if  ( !mp->inFrame )
//                     continue;
//                 const int nIdx {keyPos.first};
//                 if (  keysLeft.close[nIdx] )
//                 {
//                     Eigen::Vector2d obs((double)keysLeft.keyPoints[nIdx].pt.x, (double)keysLeft.keyPoints[nIdx].pt.y);
//                     Eigen::Vector3d point = mp->getWordPose3d();
//                     const int octL = keysLeft.keyPoints[nIdx].octave;
//                     double weight = (double)feLeft->InvSigmaFactor[octL];
//                     costf = OptimizePose::Create(K, point, obs, weight);
//                     problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

//                     problem.SetManifold(frame_qcw.coeffs().data(),
//                                         quaternion_local_parameterization);
//                     Eigen::Vector4d depthCheck = estimPose * mp->getWordPose4d();
//                     if ( depthCheck(2) < zedPtr->mBaseline * fm.closeNumber )
//                         continue;
//                     if ( keyPos.second < 0 )
//                         continue;
//                     const int rIdx {keyPos.second};
//                     Eigen::Vector2d obsr((double)keysLeft.rightKeyPoints[rIdx].pt.x, (double)keysLeft.rightKeyPoints[rIdx].pt.y);
//                     Eigen::Vector3d pointr = mp->getWordPose3d();
//                     const int octR = keysLeft.rightKeyPoints[rIdx].octave;
//                     weight = (double)feLeft->InvSigmaFactor[octR];
//                     costf = OptimizePoseR::Create(K,tc1c2, qc1c2, pointr, obsr, weight);
//                     problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

//                     problem.SetManifold(frame_qcw.coeffs().data(),
//                                         quaternion_local_parameterization);
//                     continue;
//                 }
//                 else
//                 {
//                     Eigen::Vector2d obs((double)keysLeft.keyPoints[nIdx].pt.x, (double)keysLeft.keyPoints[nIdx].pt.y);
//                     Eigen::Vector3d point = mp->getWordPose3d();
//                     const int octL = keysLeft.keyPoints[nIdx].octave;
//                     const double weight = (double)feLeft->InvSigmaFactor[octL];
//                     costf = OptimizePose::Create(K, point, obs, weight);
//                 }
//             }
//             else if ( keyPos.second >= 0)
//             {
//                 if  ( !mp->inFrameR )
//                     continue;
//                 const int nIdx {keyPos.second};
//                 Eigen::Vector2d obs((double)keysLeft.rightKeyPoints[nIdx].pt.x, (double)keysLeft.rightKeyPoints[nIdx].pt.y);
//                 Eigen::Vector3d point = mp->getWordPose3d();
//                 const int octR = keysLeft.rightKeyPoints[nIdx].octave;
//                 const double weight = (double)feLeft->InvSigmaFactor[octR];
//                 costf = OptimizePoseR::Create(K,tc1c2, qc1c2, point, obs, weight);
//             }
//             else
//                 continue;
//             problem.AddResidualBlock(costf, loss_function /* squared loss */,frame_tcw.data(), frame_qcw.coeffs().data());

//             problem.SetManifold(frame_qcw.coeffs().data(),
//                                         quaternion_local_parameterization);
//         }
//         ceres::Solver::Options options;
//         options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//         options.max_num_iterations = 100;
//         ceres::Solver::Summary summary;
//         ceres::Solve(options, &problem, &summary);
//         Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
//         estimPose.block<3, 3>(0, 0) = R;
//         estimPose.block<3, 1>(0, 3) = frame_tcw;
//         nIn = 0;
//         nStereo = 0;
//         nStereo = findOutliersR(estimPose, activeMapPoints, keysLeft, matchesIdxs, thresh, MPsOutliers, weights, nIn);
//     }
//     return std::pair<int,int>(nIn, nStereo);
// }

bool FeatureTracker::worldToFrameRTrack(MapPoint* mp, const bool right, const Eigen::Matrix4d& predPoseInv, const Eigen::Matrix4d& tempPose)
{
    Eigen::Vector4d wPos = mp->getWordPose4d();
    Eigen::Vector4d point = predPoseInv * wPos;

    double fxc, fyc, cxc, cyc;
    
    fxc = fx;
    fyc = fy;
    cxc = cx;
    cyc = cy;


    if ( point(2) <= 0.0 )
    {
        if ( right )
            mp->inFrameR = false;
        else
            mp->inFrame = false;
        return false;
    }
    const double invZ = 1.0f/point(2);

    const double u {fxc*point(0)*invZ + cxc};
    const double v {fyc*point(1)*invZ + cyc};

    const int h {zedPtr->mHeight};
    const int w {zedPtr->mWidth};
    if ( u < 0 || v < 0 || u >= w || v >= h )
    {
        if ( right )
            mp->inFrameR = false;
        else
            mp->inFrame = false;
        return false;
    }

    Eigen::Vector3d tPoint = point.block<3,1>(0,0);
    float dist = tPoint.norm();

    int predScale = mp->predictScale(dist);

    if ( right )
    {
        mp->scaleLevelR = predScale;
        mp->inFrameR = true;
        mp->predR = cv::Point2f((float)u, (float)v);
    }
    else
    {
        mp->scaleLevelL = predScale;
        mp->inFrame = true;
        mp->predL = cv::Point2f((float)u, (float)v);
    }

    return true;
}

void FeatureTracker::insertKeyFrameR(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const int nStereo, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm)
{
    Eigen::Matrix4d referencePose = latestKF->pose.getInvPose() * estimPose;
    KeyFrame* kF = new KeyFrame(zedPtr, referencePose, estimPose, leftIm, rleftIm,map->kIdx, curFrame);
    if ( map->aprilTagDetected && !map->LCStart )
        kF->LCCand = true;
    kF->scaleFactor = feLeft->scalePyramid;
    kF->sigmaFactor = feLeft->sigmaFactor;
    kF->InvSigmaFactor = feLeft->InvSigmaFactor;
    kF->nScaleLev = feLeft->nLevels;
    kF->logScale = log(feLeft->imScale);
    kF->keyF = true;
    kF->prevKF = latestKF;
    latestKF->nextKF = kF;
    
    kF->unMatchedF.resize(keysLeft.keyPoints.size(), -1);
    kF->unMatchedFR.resize(keysLeft.rightKeyPoints.size(), -1);
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
    kF->localMapPointsR.resize(keysLeft.rightKeyPoints.size(), nullptr);
    activeMapPoints.reserve(activeMapPoints.size() + keysLeft.keyPoints.size());
    kF->keys.getKeys(keysLeft);
    std::lock_guard<std::mutex> lock(map->mapMutex);
    int trckedKeys {0};
    for ( size_t i{0}, end {matchesIdxs.size()}; i < end; i++)
    {
        MapPoint* mp = activeMapPoints[i];
        std::pair<int,int>& keyPos = matchesIdxs[i];
        if ( !mp )
            continue;
        if ( keyPos.first < 0 && keyPos.second < 0 )
            continue;
        if ( MPsOutliers[i] )
            continue;
        mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kF, keyPos));
        mp->update(kF);

        if ( keyPos.first >= 0 )
        {
            kF->localMapPoints[keyPos.first] = mp;
            kF->unMatchedF[keyPos.first] = mp->kdx;
        }
        if ( keyPos.second >= 0 )
        {
            kF->localMapPointsR[keyPos.second] = mp;
            kF->unMatchedFR[keyPos.second] = mp->kdx;
        }
        trckedKeys++;
        

    }

    if ( nStereo < minNStereo)
    {
        std::vector<std::pair<float, int>> allDepths;
        allDepths.reserve(keysLeft.keyPoints.size());
        for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
        {
            if ( keysLeft.estimatedDepth[i] > 0 && matchedIdxsL[i] < 0 ) 
                allDepths.emplace_back(keysLeft.estimatedDepth[i], i);
        }
        std::sort(allDepths.begin(), allDepths.end());
        int count {0};
        for (size_t i{0}, end{allDepths.size()}; i < end; i++)
        {
            const int lIdx {allDepths[i].second};
            const int rIdx {keysLeft.rightIdxs[lIdx]};
            if ( count >= maxAddedStereo && !keysLeft.close[lIdx] )
                break;
            count ++;
            const double zp = (double)keysLeft.estimatedDepth[lIdx];
            const double xp = (double)(((double)keysLeft.keyPoints[lIdx].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keysLeft.keyPoints[lIdx].pt.y-cy)*zp/fy);
            Eigen::Vector4d p(xp, yp, zp, 1);
            p = estimPose * p;
            MapPoint* mp = new MapPoint(p, keysLeft.Desc.row(lIdx), keysLeft.keyPoints[lIdx], map->kIdx, map->pIdx);
            mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kF, std::pair<int,int>(lIdx,rIdx)));
            mp->update(kF);
            kF->localMapPoints[lIdx] = mp;
            kF->localMapPointsR[rIdx] = mp;
            activeMapPoints.emplace_back(mp);
            map->addMapPoint(mp);
            trckedKeys ++;
        }

    }
    kF->calcConnections();
    lastKFTrackedNumb = trckedKeys;
    kF->nKeysTracked = trckedKeys;
    if ( trckedKeys > 350 )
        precCheckMatches = 0.7f;
    else
        precCheckMatches = 0.9f;
    map->addKeyFrame(kF);
    latestKF = kF;
    Eigen::Matrix4d lastKFPose = estimPose;
    lastKFPoseInv = lastKFPose.inverse();
    allFrames.emplace_back(kF);
    if ( map->aprilTagDetected && !map->LCStart )
    {
        map->LCStart = true;
        map->LCCandIdx = kF->numb;
    }
    if ( map->keyFrames.size() > 3 && !map->LCStart )
        map->keyFrameAdded = true;

}

void FeatureTracker::addFrame(const Eigen::Matrix4d& estimPose)
{
    Eigen::Matrix4d referencePose =  latestKF->pose.getInvPose() * estimPose;
    KeyFrame* kF = new KeyFrame(referencePose, estimPose, lIm.im, lIm.rIm,map->kIdx, curFrame);
    kF->prevKF = latestKF;
    kF->keyF = false;
    kF->active = false;
    kF->visualize = false;
    allFrames.emplace_back(kF);
    
}

void FeatureTracker::changePosesLCA(const int endIdx)
{
    KeyFrame* kf = map->keyFrames.at(endIdx);
    while ( true )
    {
        KeyFrame* nextKF = kf->nextKF;
        if ( nextKF )
        {
            Eigen::Matrix4d keyPose = kf->getPose();
            nextKF->updatePose(keyPose);
            kf = nextKF;
        }
        else
            break;
    }
    Eigen::Matrix4d keyPose = kf->getPose();
    zedPtr->mCameraPose.changePose(keyPose);

    Eigen::Matrix4d lastKFPose = keyPose;
    lastKFPoseInv = lastKFPose.inverse();

    predNPose = zedPtr->mCameraPose.pose * predNPoseRef;
    predNPoseInv = predNPose.inverse();

}

void FeatureTracker::removeOutOfFrameMPsR(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints)
{
    const size_t end{activeMapPoints.size()};
    Eigen::Matrix4d toRCamera = (predNPose * zedPtr->extrinsics).inverse();
    Eigen::Matrix4d toCamera = predNPose.inverse();
    int j {0};
    Eigen::Matrix4d temp = currCamPose.inverse() * predNPose;
    Eigen::Matrix4d tempR = currCamPose.inverse() * (predNPose * zedPtr->extrinsics);
    for ( size_t i {0}; i < end; i++)
    {
        MapPoint* mp = activeMapPoints[i];
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        bool c1 = worldToFrameRTrack(mp, false, toCamera, temp);
        bool c2 = worldToFrameRTrack(mp, true, toRCamera, tempR);
        if (c1 && c2 )
        {
            mp->setActive(true);
        }
        else
        {
            mp->setActive(false);
            continue;
        }
        activeMapPoints[j++] = mp;
    }
    activeMapPoints.resize(j);
}

void FeatureTracker::newPredictMPs(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers)
{
    const size_t end{activeMapPoints.size()};
    Eigen::Matrix4d toRCamera = (predNPose * zedPtr->extrinsics).inverse();
    Eigen::Matrix4d toCamera = predNPose.inverse();
    Eigen::Matrix4d temp = currCamPose.inverse() * predNPose;
    Eigen::Matrix4d tempR = currCamPose.inverse() * (predNPose * zedPtr->extrinsics);
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for ( size_t i {0}; i < end; i++)
    {
        MapPoint* mp = activeMapPoints[i];
        if ( !mp )
            continue;
        std::pair<int,int>& keyPos = matchesIdxs[i];
        if (!worldToFrameRTrack(mp, false, toCamera, temp))
        {
            if ( keyPos.first >=0 )
            {
                matchedIdxsL[keyPos.first] = -1;
                keyPos.first = -1;
            }
        }
        if (!worldToFrameRTrack(mp, true, toRCamera, tempR))
        {
            if ( keyPos.second >= 0 )
            {
                matchedIdxsR[keyPos.second] = -1;
                keyPos.second = -1;
            }
        }
        if ( MPsOutliers[i] )
        {
            MPsOutliers[i] = false;
            if ( keyPos.first >=0 )
            {
                matchedIdxsL[keyPos.first] = -1;
                keyPos.first = -1;
            }
            if ( keyPos.second >= 0 )
            {
                matchedIdxsR[keyPos.second] = -1;
                keyPos.second = -1;
            }
        }
    }
}

void FeatureTracker::setActiveOutliers(std::vector<MapPoint*>& activeMPs, std::vector<bool>& MPsOutliers, std::vector<std::pair<int,int>>& matchesIdxs)
{
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for ( size_t i{0}, end{MPsOutliers.size()}; i < end; i++)
    {
        MapPoint*& mp = activeMPs[i];
        const std::pair<int,int>& keyPos = matchesIdxs[i];
        if ( (keyPos.first >= 0 || keyPos.second >= 0) && !MPsOutliers[i] )
            mp->unMCnt = 0;
        else
            mp->unMCnt++;

        if ( !MPsOutliers[i] && mp->unMCnt < 20 )
        {
            continue;
        }
        mp->SetIsOutlier( true );
    }
}

void FeatureTracker::TrackImageT(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb)
{
    curFrame = frameNumb;
    curFrameNumb++;
    
    if ( map->LBADone || map->LCDone )
    {
        std::lock_guard<std::mutex> lock(map->mapMutex);
        const int endIdx = (map->LCDone) ? map->endLCIdx : map->endLBAIdx;
        changePosesLCA(endIdx);
        if ( map->LCDone )
            map->LCDone = false;
        if ( map->LBADone )
            map->LBADone = false;
    }

    cv::Mat realLeftIm, realRightIm;
    cv::Mat leftIm, rightIm;

    realLeftIm = leftRect;
    realRightIm = rightRect;
    

    if(realLeftIm.channels()==3)
    {
        cvtColor(realLeftIm,leftIm,cv::COLOR_BGR2GRAY);
        cvtColor(realRightIm,rightIm,cv::COLOR_BGR2GRAY);
    }
    else if(realLeftIm.channels()==4)
    {
        cvtColor(realLeftIm,leftIm,cv::COLOR_BGRA2GRAY);
        cvtColor(realRightIm,rightIm,cv::COLOR_BGRA2GRAY);
    }
    else
    {
        leftIm = realLeftIm.clone();
        rightIm = realRightIm.clone();
    }
    
    TrackedKeys keysLeft;


    if ( curFrameNumb == 0 )
    {
        extractORBStereoMatchR(leftIm, rightIm, keysLeft);

        initializeMapR(keysLeft);

        return;
    }
    std::vector<TII::MapPoint *> activeMpsTemp;
    {
    std::lock_guard<std::mutex> lock(map->mapMutex);
    removeOutOfFrameMPsR(zedPtr->mCameraPose.pose, predNPose, activeMapPoints);
    activeMpsTemp = activeMapPoints;
    }

    

    extractORBStereoMatchR(leftIm, rightIm, keysLeft);

    std::vector<int> matchedIdxsL(keysLeft.keyPoints.size(), -1);
    std::vector<int> matchedIdxsR(keysLeft.rightKeyPoints.size(), -1);
    std::vector<std::pair<int,int>> matchesIdxs(activeMpsTemp.size(), std::make_pair(-1,-1));
    
    std::vector<bool> MPsOutliers(activeMpsTemp.size(),false);

    Eigen::Matrix4d estimPose = predNPoseInv;

    float rad {10.0};

    if ( curFrameNumb == 1 )
        rad = 120;
    else
        rad = 10;

    std::pair<int,int> nIn(-1,-1);
    int prevIn = -1;
    float prevrad = rad;
    bool toBreak {false};
    int countIte {0};
    while ( nIn.first < minInliers )
    {
        countIte++;
        fm.matchByProjectionRPred(activeMpsTemp, keysLeft, matchedIdxsL, matchedIdxsR, matchesIdxs, rad);

        nIn = estimatePoseGTSAM(activeMpsTemp, keysLeft, matchesIdxs, estimPose, MPsOutliers, true);

        if ( nIn.first < minInliers  && !toBreak )
        {
            estimPose = predNPoseInv;
            matchedIdxsL = std::vector<int>(keysLeft.keyPoints.size(), -1);
            matchedIdxsR = std::vector<int>(keysLeft.rightKeyPoints.size(), -1);
            MPsOutliers = std::vector<bool>(activeMpsTemp.size(),false);
            matchesIdxs = std::vector<std::pair<int,int>>(activeMpsTemp.size(), std::make_pair(-1,-1));
            if ( nIn.first < prevIn )
            {
                rad = prevrad;
                toBreak = true;
            }
            else
            {
                prevrad = rad;
                prevIn = nIn.first;
                rad += 30.0;
            }
        }
        else
            break;
        if ( countIte > 3 && !toBreak )
            toBreak = true;

    }

    newPredictMPs(zedPtr->mCameraPose.pose, estimPose.inverse(), activeMpsTemp, matchedIdxsL, matchedIdxsR, matchesIdxs, MPsOutliers);

    rad = 4;
    fm.matchByProjectionRPred(activeMpsTemp, keysLeft, matchedIdxsL, matchedIdxsR, matchesIdxs, rad);

    std::pair<int,int> nStIn = estimatePoseGTSAM(activeMpsTemp, keysLeft, matchesIdxs, estimPose, MPsOutliers, false);

    std::vector<cv::KeyPoint> lp;
    std::vector<bool> closeL;
    lp.reserve(matchesIdxs.size());
    closeL.reserve(matchesIdxs.size());
    for ( size_t i{0}; i < matchesIdxs.size(); i++)
    {
        const std::pair<int,int>& keyPos = matchesIdxs[i];
        if ( keyPos.first >= 0 )
        {
            lp.emplace_back(keysLeft.keyPoints[keyPos.first]);
            if ( MPsOutliers[i] )
                continue;
            if ( keysLeft.close[keyPos.first] )
                closeL.emplace_back(true);
            else
                closeL.emplace_back(false);
        }
    }
    drawKeys("VSLAM : Tracked KeyPoints", realLeftIm, lp, closeL);

    poseEst = estimPose.inverse();

    insertKeyFrameCount ++;
    if ( ((nStIn.second < minNStereo || insertKeyFrameCount >= keyFrameCountEnd) && nStIn.first < precCheckMatches * lastKFTrackedNumb) || (map->aprilTagDetected && !map->LCStart) )
    {
        insertKeyFrameCount = 0;
        insertKeyFrameR(keysLeft, matchedIdxsL,matchesIdxs, nStIn.second, poseEst, MPsOutliers, leftIm, realLeftIm);
    }
    else
        addFrame(poseEst);


    publishPoseNew();

    setActiveOutliers(activeMpsTemp,MPsOutliers, matchesIdxs);
}

void FeatureTracker::drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys, std::vector<bool>& close)
{
    cv::Mat outIm = im.clone();
    int count {0};
    for (auto& key:keys)
    {
        if ( close[count] )
            cv::circle(outIm, key.pt,3,cv::Scalar(255,0,0),2);
        else
            cv::circle(outIm, key.pt,3,cv::Scalar(255,0,0),2);
        count++;
    }
    cv::imshow(com, outIm);
    cv::waitKey(1);
}

void FeatureTracker::publishPoseNew()
{
    Eigen::Matrix4d prevWPoseInv = zedPtr->mCameraPose.poseInverse;
    Eigen::Matrix4d referencePose = lastKFPoseInv * poseEst;
    zedPtr->mCameraPose.setPose(poseEst);
    zedPtr->mCameraPose.refPose = referencePose;
    predNPoseRef = prevWPoseInv * poseEst;
    predNPose = poseEst * predNPoseRef;
    predNPoseInv = predNPose.inverse();
}

} // namespace TII