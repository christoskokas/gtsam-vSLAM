#include "FeatureTracker.h"
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/inference/Symbol.h>


namespace TII
{

FeatureTracker::FeatureTracker(std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureExtractor> _feLeft, std::shared_ptr<FeatureExtractor> _feRight, std::shared_ptr<Map> _map) : zedPtr(_zedPtr), feLeft(_feLeft), feRight(_feRight), map(_map), fm(zedPtr, _feLeft, _feRight, zedPtr->mHeight), fx(_zedPtr->mCameraLeft->fx), fy(_zedPtr->mCameraLeft->fy), cx(_zedPtr->mCameraLeft->cx), cy(_zedPtr->mCameraLeft->cy), activeMapPoints(_map->activeMapPoints), allFrames(_map->allFramesPoses), currentIMUData(nullptr)
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
    std::thread extractLeft(&FeatureExtractor::extractKeysNew, feLeft, std::ref(leftIm), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc));
    std::thread extractRight(&FeatureExtractor::extractKeysNew, feRight, std::ref(rightIm), std::ref(keysLeft.rightKeyPoints),std::ref(keysLeft.rightDesc));
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
    kF->mIMUData = currentIMUData;
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
    // const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();

    const size_t prevS { activeMapPoints.size() };
    const Eigen::Matrix3d& K_eigen = zedPtr->mCameraLeft->intrinsics;

    // Convert Eigen intrinsics to GTSAM intrinsics
    auto K = boost::make_shared<gtsam::Cal3_S2>(
        K_eigen(0, 0), K_eigen(1, 1), 0, K_eigen(0, 2), K_eigen(1, 2));

    // Initializing GTSAM graph and initial values
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    Eigen::Matrix4d estimPoseInv = estimPose.inverse();
    Eigen::Matrix4d currentPoseInv = zedPtr->mCameraPose.getPose();


    gtsam::Pose3 initialPose(
        gtsam::Rot3(estimPoseInv.block<3, 3>(0, 0)),
        gtsam::Point3(estimPoseInv.block<3, 1>(0, 3))
    );

    gtsam::Pose3 startPose(
        gtsam::Rot3(currentPoseInv.block<3, 3>(0, 0)),
        gtsam::Point3(currentPoseInv.block<3, 1>(0, 3))
    );

    gtsam::Pose3 gtsamExtrinsics(
        gtsam::Rot3(zedPtr->extrinsics.block<3, 3>(0, 0)),
        gtsam::Point3(zedPtr->extrinsics.block<3, 1>(0, 3))
    );

    // Add initial guess to the graph
    // initialEstimate.insert(gtsam::Symbol('x', 1), initialPose);
    initialEstimate.insert(gtsam::Symbol('x', 0), startPose);
    graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', 0), startPose));

    gtsam::SharedNoiseModel noiseModel = nullptr;

    // Loop through matched points and add projection factors
    for (size_t i{0}; i < matchesIdxs.size(); i++) 
    {
        if (MPsOutliers[i]) 
            continue;
        const std::pair<int, int>& keyPos = matchesIdxs[i];

        MapPoint* mp = activeMapPoints[i];
        if (mp->GetIsOutlier()) 
            continue;

        Eigen::Vector3d point = mp->getWordPose3d();
        gtsam::Point3 gtsamPoint(point);

        if (keyPos.first >= 0) 
        { // Left image point
            const int nIdx = keyPos.first;
            if (!mp->inFrame)
                continue;
            Eigen::Vector2d obs(keysLeft.keyPoints[nIdx].pt.x, keysLeft.keyPoints[nIdx].pt.y);
            gtsam::Point2 observation(obs);
            const int octL = keysLeft.keyPoints[nIdx].octave;
            double sigma = 1.0/feLeft->InvSigmaFactor[octL];
            noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
            auto factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                observation, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K);

            graph.add(factor);

            // Add landmark initial estimate
            initialEstimate.insert(gtsam::Symbol('l', i), gtsamPoint);
            // Motion Only BA
            graph.add(gtsam::NonlinearEquality<gtsam::Point3>(gtsam::Symbol('l', i), gtsamPoint));
            if (keysLeft.close[nIdx])
            {
                Eigen::Vector4d depthCheck = estimPose * mp->getWordPose4d();
                if ( depthCheck(2) >= zedPtr->mBaseline * fm.closeNumber )
                {
                    if ( keyPos.second >= 0 )
                    {
                        const int rIdx = keyPos.second;
                        if (!mp->inFrameR) 
                            continue;
                        
                        Eigen::Vector2d obsR(keysLeft.rightKeyPoints[rIdx].pt.x, keysLeft.rightKeyPoints[rIdx].pt.y);
                        gtsam::Point2 observationR(obsR);

                        // Adding projection factor for the right image (stereo)
                        const int octR = keysLeft.rightKeyPoints[rIdx].octave;
                        double sigma = 1.0/feLeft->InvSigmaFactor[octR];
                        noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
                        auto factorR = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                            observationR, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K, gtsamExtrinsics);

                        graph.add(factorR);
                        continue;
                    }

                }
                
            }
            
        }
        else if (keyPos.second >= 0) { // Right image point
            const int rIdx = keyPos.second;
            if (!mp->inFrameR) 
                continue;
            
            Eigen::Vector2d obsR(keysLeft.rightKeyPoints[rIdx].pt.x, keysLeft.rightKeyPoints[rIdx].pt.y);
            gtsam::Point2 observationR(obsR);

            // Adding projection factor for the right image (stereo)
            const int octR = keysLeft.rightKeyPoints[rIdx].octave;
            double sigma = 1.0/feLeft->InvSigmaFactor[octR];
            noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
            auto factorR = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                observationR, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K, gtsamExtrinsics);

            graph.add(factorR);

            // Optionally, add initial estimate if not already inserted
            if (!initialEstimate.exists(gtsam::Symbol('l', i))) 
            {
                initialEstimate.insert(gtsam::Symbol('l', i), gtsamPoint);
                graph.add(gtsam::NonlinearEquality<gtsam::Point3>(gtsam::Symbol('l', i), gtsamPoint));
            }
        }
        else
            continue;
    }

    // IMU Preintegration 

    const double gyroNoiseDensity = currentIMUData->mGyroNoiseDensity;
    double gyroRandomWalk = currentIMUData->mGyroRandomWalk;
    double accelNoiseDensity = currentIMUData->mAccelNoiseDensity;
    double accelRandomWalk = currentIMUData->mAccelRandomWalk;
    
    // Initialize IMU preintegration parameters
    // auto preintegrationParams = gtsam::PreintegrationParams::MakeSharedU(9.81); 
    auto preintegrationParams = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
    
    preintegrationParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(gyroNoiseDensity, 2);
    preintegrationParams->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(accelNoiseDensity, 2);

    preintegrationParams->biasOmegaCovariance = gtsam::Matrix33::Identity() * pow(gyroRandomWalk, 2);
    preintegrationParams->biasAccCovariance = gtsam::Matrix33::Identity() * pow(accelRandomWalk, 2);

    preintegrationParams->integrationCovariance = gtsam::I_3x3 * 0.0001;

    const Eigen::Matrix4d& TBodyToCam = zedPtr->mCameraLeft->TBodyToCam.inverse();
    gtsam::Pose3 TBodyToCamGtsam(
        gtsam::Rot3(TBodyToCam.block<3, 3>(0, 0)),
        gtsam::Point3(TBodyToCam.block<3, 1>(0, 3))
    );

    preintegrationParams->body_P_sensor = TBodyToCamGtsam;

    gtsam::imuBias::ConstantBias initialBias; 

    gtsam::PreintegratedCombinedMeasurements preintegratedImu(preintegrationParams, initialBias);

    const size_t IMUDataSize{currentIMUData->mTimestamps.size()};
    double dt {currentIMUData->mHz/zedPtr->mFps};
    for (size_t i = 0; i < IMUDataSize; ++i)
    {
        const auto& acceleration = currentIMUData->mAcceleration[i];
        const auto& angleVelocity = currentIMUData->mAngleVelocity[i];
        gtsam::Vector3 accel(acceleration[0], acceleration[1], acceleration[2]);
        gtsam::Vector3 angleVel(angleVelocity[0], angleVelocity[1], angleVelocity[2]);

        if ((i+1) < IMUDataSize)
        {
            const auto& timestamp0 = currentIMUData->mTimestamps[i];
            const auto& timestamp1 = currentIMUData->mTimestamps[i+1];
            dt = (timestamp1 - timestamp0) / 1e9;
        }

        // Integrate the IMU measurements
        preintegratedImu.integrateMeasurement(accel, angleVel, dt);
    }

    // Insert initial velocity and bias (assumed zero for now)
    gtsam::Vector3 priorVel(zedPtr->mCameraLeft->mVelocity.data());
    initialEstimate.insert(gtsam::Symbol('v', 0), priorVel);
    initialEstimate.insert(gtsam::Symbol('b', 0), initialBias);

    graph.add(gtsam::CombinedImuFactor(gtsam::Symbol('x', 0), gtsam::Symbol('v', 0), 
                               gtsam::Symbol('x', 1), gtsam::Symbol('v', 1), 
                               gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), preintegratedImu));

    gtsam::NavState prev_state(startPose, priorVel);
    gtsam::NavState prop_state = prev_state;
    gtsam::imuBias::ConstantBias prev_bias = initialBias;
    prop_state = preintegratedImu.predict(prev_state, prev_bias);


    initialEstimate.insert(gtsam::Symbol('x',1), prop_state.pose());
    initialEstimate.insert(gtsam::Symbol('v',1), prop_state.v());
    initialEstimate.insert(gtsam::Symbol('b',1), prev_bias);

    // Add bias factor to the graph
    auto biasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // Small noise model for bias
    graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), gtsam::imuBias::ConstantBias(), biasNoise));

    // Insert pose and velocity estimates at step_1 (initial guess)
    // initialEstimate.insert(gtsam::Symbol('v', 1), gtsam::Vector3(0, 0, 0));
    // initialEstimate.insert(gtsam::Symbol('b', 1), initialBias);

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 100;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    gtsam::Values result = optimizer.optimize();

    // Extract optimized pose
    gtsam::Pose3 optimizedPose = result.at<gtsam::Pose3>(gtsam::Symbol('x', 1));

    gtsam::Vector3 camVelocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', 1));
    zedPtr->mCameraLeft->mNewVelocity = Eigen::Vector3d(camVelocity.data());

    estimPoseInv.block<3, 3>(0, 0) = optimizedPose.rotation().matrix();
    estimPoseInv.block<3, 1>(0, 3) = optimizedPose.translation();
    estimPose = estimPoseInv.inverse();
    int nIn = 0, nStereo = 0;
    nStereo = findOutliersR(estimPose, activeMapPoints, keysLeft, matchesIdxs, 7.815, MPsOutliers, std::vector<float>(prevS, 1.0f), nIn);

    return std::pair<int, int>(nIn, nStereo);
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
    kF->mIMUData = currentIMUData;
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
    kF->mIMUData = currentIMUData;
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

Eigen::Matrix4d FeatureTracker::PredictNextPoseIMU()
{
    const double gyroNoiseDensity = currentIMUData->mGyroNoiseDensity;
    double gyroRandomWalk = currentIMUData->mGyroRandomWalk;
    double accelNoiseDensity = currentIMUData->mAccelNoiseDensity;
    double accelRandomWalk = currentIMUData->mAccelRandomWalk;
    
    // Initialize IMU preintegration parameters
    // auto preintegrationParams = gtsam::PreintegrationParams::MakeSharedU(9.81); 
    auto preintegrationParams = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
    
    preintegrationParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(gyroNoiseDensity, 2);
    preintegrationParams->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(accelNoiseDensity, 2);

    preintegrationParams->biasOmegaCovariance = gtsam::Matrix33::Identity() * pow(gyroRandomWalk, 2);
    preintegrationParams->biasAccCovariance = gtsam::Matrix33::Identity() * pow(accelRandomWalk, 2);

    preintegrationParams->integrationCovariance = gtsam::I_3x3 * 0.0001;

    const Eigen::Matrix4d& TBodyToCam = zedPtr->mCameraLeft->TBodyToCam.inverse();
    gtsam::Pose3 TBodyToCamGtsam(
        gtsam::Rot3(TBodyToCam.block<3, 3>(0, 0)),
        gtsam::Point3(TBodyToCam.block<3, 1>(0, 3))
    );

    preintegrationParams->body_P_sensor = TBodyToCamGtsam;

    gtsam::imuBias::ConstantBias initialBias; 

    gtsam::PreintegratedCombinedMeasurements preintegratedImu(preintegrationParams, initialBias);

    const size_t IMUDataSize{currentIMUData->mTimestamps.size()};
    double dt {currentIMUData->mHz/zedPtr->mFps};
    for (size_t i = 0; i < IMUDataSize; ++i)
    {
        const auto& acceleration = currentIMUData->mAcceleration[i];
        const auto& angleVelocity = currentIMUData->mAngleVelocity[i];
        gtsam::Vector3 accel(acceleration[0], acceleration[1], acceleration[2]);
        gtsam::Vector3 angleVel(angleVelocity[0], angleVelocity[1], angleVelocity[2]);

        if ((i+1) < IMUDataSize)
        {
            const auto& timestamp0 = currentIMUData->mTimestamps[i];
            const auto& timestamp1 = currentIMUData->mTimestamps[i+1];
            dt = (timestamp1 - timestamp0) / 1e9;
        }

        // Integrate the IMU measurements
        preintegratedImu.integrateMeasurement(accel, angleVel, dt);
    }

    // Insert initial velocity and bias (assumed zero for now)
    gtsam::Vector3 priorVel(0,0,0);

    const Eigen::Matrix4d& currentCamPose = zedPtr->mCameraPose.getPose();

    gtsam::Pose3 startPose(
        gtsam::Rot3(currentCamPose.block<3, 3>(0, 0)),
        gtsam::Point3(currentCamPose.block<3, 1>(0, 3))
    );

    gtsam::NavState prev_state(startPose, priorVel);
    gtsam::NavState prop_state = prev_state;
    gtsam::imuBias::ConstantBias prev_bias = initialBias;
    prop_state = preintegratedImu.predict(prev_state, prev_bias);

    gtsam::Pose3 predPose = prop_state.pose();
    Eigen::Matrix4d predPoseEigen = Eigen::Matrix4d::Identity();
    predPoseEigen.block<3, 3>(0, 0) = predPose.rotation().matrix();
    predPoseEigen.block<3, 1>(0, 3) = predPose.translation();
    return predPoseEigen;

}

void FeatureTracker::TrackImageT(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr /* = nullptr*/)
{
    if (IMUDataptr)
        currentIMUData = IMUDataptr;
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

    // predNPoseInv = PredictNextPoseIMU();
    // predNPose = predNPoseInv.inverse();

    Eigen::Matrix4d estimPose = predNPoseInv;


    std::vector<MapPoint *> activeMpsTemp;
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

    // Eigen::Matrix4d estimPose = PredictNextPoseIMU();

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

    currentIMUData = nullptr;
    zedPtr->mCameraLeft->mVelocity = zedPtr->mCameraLeft->mNewVelocity;
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