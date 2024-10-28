#include "FeatureTracker.h"
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/StereoFactor.h>
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
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/inference/Symbol.h>


namespace GTSAM_VIOSLAM
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

void FeatureTracker::extractORBAndStereoMatch(cv::Mat& leftIm, cv::Mat& rightIm, TrackedKeys& keysLeft)
{
    std::thread extractLeft(&FeatureExtractor::extractKeysNew, feLeft, std::ref(leftIm), std::ref(keysLeft.keyPoints), std::ref(keysLeft.Desc));
    std::thread extractRight(&FeatureExtractor::extractKeysNew, feRight, std::ref(rightIm), std::ref(keysLeft.rightKeyPoints),std::ref(keysLeft.rightDesc));
    extractLeft.join();
    extractRight.join();



    fm.findStereoMatchesORB2R(leftIm, rightIm, keysLeft.rightDesc, keysLeft.rightKeyPoints, keysLeft);

    assignKeysToGrids(keysLeft, keysLeft.keyPoints, keysLeft.lkeyGrid, zedPtr->mWidth, zedPtr->mHeight);
    assignKeysToGrids(keysLeft, keysLeft.rightKeyPoints, keysLeft.rkeyGrid, zedPtr->mWidth, zedPtr->mHeight);

}

void FeatureTracker::initializeMap(TrackedKeys& keysLeft)
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

void FeatureTracker::initializeMono(TrackedKeys& keysLeft)
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
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
    activeMapPoints.reserve(keysLeft.keyPoints.size());
    kF->keys.getKeys(keysLeft);
    int trckedKeys {0};
    for (size_t i{0}, end{keysLeft.keyPoints.size()}; i < end; i++)
    {
        if ( keysLeft.estimatedDepth[i] > 0 )
        {
            const int rIdx {-1};
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
            mp->update(kF);
            activeMapPoints.emplace_back(mp);
            kF->localMapPoints[i] = mp;
            kF->unMatchedF[i] = mp->kdx;
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
    const size_t prevS { activeMapPoints.size() };
    const Eigen::Matrix3d& K_eigen = zedPtr->mCameraLeft->intrinsics;

    // Convert Eigen intrinsics to GTSAM intrinsics
    auto K = boost::make_shared<gtsam::Cal3_S2>(
        K_eigen(0, 0), K_eigen(1, 1), 0, K_eigen(0, 2), K_eigen(1, 2));

    auto KStereo = boost::make_shared<gtsam::Cal3_S2Stereo>(
        K_eigen(0, 0), K_eigen(1, 1), 0, K_eigen(0, 2), K_eigen(1, 2), zedPtr->mBaseline);

    // Initializing GTSAM graph and initial values
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    Eigen::Matrix4d estimPoseInv = estimPose.inverse();
    Eigen::Matrix4d currentPoseInv = zedPtr->mCameraPose.getPose();


    gtsam::Pose3 startPose(
        gtsam::Rot3(currentPoseInv.block<3, 3>(0, 0)),
        gtsam::Point3(currentPoseInv.block<3, 1>(0, 3))
    );

    gtsam::Pose3 gtsamExtrinsics(
        gtsam::Rot3(zedPtr->extrinsics.block<3, 3>(0, 0)),
        gtsam::Point3(zedPtr->extrinsics.block<3, 1>(0, 3))
    );

    if (!currentIMUData)
    {
        gtsam::Pose3 predPose(
            gtsam::Rot3(estimPoseInv.block<3, 3>(0, 0)),
            gtsam::Point3(estimPoseInv.block<3, 1>(0, 3))
        );

        initialEstimate.insert(gtsam::Symbol('x', 1), predPose);
    }
    else
    {
        initialEstimate.insert(gtsam::Symbol('x', 0), startPose);
        graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', 0), startPose));
    }



    gtsam::SharedNoiseModel noiseModel = nullptr;

    // Loop through matched points and add projection factors
    // Motion Only BA : Optimize only the camera Pose and Velocity, not the 3D Points
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

            // For the left observation
            Eigen::Vector2d obs(keysLeft.keyPoints[nIdx].pt.x, keysLeft.keyPoints[nIdx].pt.y);
            gtsam::Point2 observation(obs);
            const int octL = keysLeft.keyPoints[nIdx].octave;
            double sigma = 1.0 / feLeft->InvSigmaFactor[octL];
            noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));

            // Check if the keypoint is close, (close enough that it can be considered stereo)
            if (keysLeft.close[nIdx]) 
            {
                Eigen::Vector2d obsR(keysLeft.rightKeyPoints[keyPos.second].pt.x, keysLeft.rightKeyPoints[keyPos.second].pt.y);
                gtsam::StereoPoint2 stereoObs(obs.x(), obsR.x(), obs.y());

                noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(sigma, sigma, sigma));

                auto stereoFactor = gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>(
                    stereoObs, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), KStereo);

                // Add the stereo factor to the graph
                graph.add(stereoFactor);

            } else 
            {
                // Use a regular projection factor for non-close keypoints
                auto factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    observation, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K);

                graph.add(factor);
            }

            // Add landmark initial estimate and any prior constraints
            initialEstimate.insert(gtsam::Symbol('l', i), gtsamPoint);

            // keep 3D point Frozen, only optimize Camera Pose
            graph.add(gtsam::NonlinearEquality<gtsam::Point3>(gtsam::Symbol('l', i), gtsamPoint));
            
        }
        else if (keyPos.second >= 0) { // Right image point
            const int rIdx = keyPos.second;
            if (!mp->inFrameR) 
                continue;
            
            Eigen::Vector2d obsR(keysLeft.rightKeyPoints[rIdx].pt.x, keysLeft.rightKeyPoints[rIdx].pt.y);
            gtsam::Point2 observationR(obsR);

            // Adding projection factor for the right keypoint
            const int octR = keysLeft.rightKeyPoints[rIdx].octave;
            double sigma = 1.0/feLeft->InvSigmaFactor[octR];
            noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
            auto factorR = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                observationR, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K, gtsamExtrinsics);

            graph.add(factorR);

            // Add initial estimate if not already inserted
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
    if (currentIMUData)
    {

        const double gyroNoiseDensity = currentIMUData->mGyroNoiseDensity;
        double gyroRandomWalk = currentIMUData->mGyroRandomWalk;
        double accelNoiseDensity = currentIMUData->mAccelNoiseDensity;
        double accelRandomWalk = currentIMUData->mAccelRandomWalk;
        
        // Set the gravity Vector of the IMU to the first IMU measurement
        const Eigen::Vector3d gravity = zedPtr->mCameraLeft->mIMUGravity;

        auto preintegrationParams = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(gravity.data())));
        // auto preintegrationParams = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(0, -9.81, 0)));
        
        // IMU Noise
        preintegrationParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(gyroNoiseDensity, 2);
        preintegrationParams->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(accelNoiseDensity, 2);
        preintegrationParams->biasOmegaCovariance = gtsam::Matrix33::Identity() * pow(gyroRandomWalk, 2);
        preintegrationParams->biasAccCovariance = gtsam::Matrix33::Identity() * pow(accelRandomWalk, 2);

        preintegrationParams->integrationCovariance = gtsam::I_3x3 * 1e-5;
        
        // preintegrationParams->biasAccOmegaInt = gtsam::Matrix::Identity(6,6)*1e-5;

        // Get Extrinsics between IMU and Camera Pose
        const Eigen::Matrix4d& TBodyToCam = zedPtr->mCameraLeft->TBodyToCam;
        gtsam::Pose3 TBodyToCamGtsam(
            gtsam::Rot3(TBodyToCam.block<3, 3>(0, 0)),
            gtsam::Point3(TBodyToCam.block<3, 1>(0, 3))
        );

        preintegrationParams->body_P_sensor = TBodyToCamGtsam;

        gtsam::PreintegratedCombinedMeasurements preintegratedImu(preintegrationParams, initialBias);

        const size_t IMUDataSize{currentIMUData->mTimestamps.size()};
        double dt {1.0/currentIMUData->mHz};
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

            preintegratedImu.integrateMeasurement(accel, angleVel, dt);
        }

        // Add velocity and bias and freeze them
        gtsam::Vector3 priorVel(zedPtr->mCameraLeft->mVelocity.data());
        initialEstimate.insert(gtsam::Symbol('v', 0), priorVel);
        graph.add(gtsam::NonlinearEquality<gtsam::Vector3>(gtsam::Symbol('v', 0), priorVel));
        initialEstimate.insert(gtsam::Symbol('b', 0), initialBias);
        graph.add(gtsam::NonlinearEquality<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', 0), initialBias));

        // Predict next Pose using IMU Measurements
        gtsam::NavState prev_state(startPose, priorVel);
        gtsam::NavState prop_state = prev_state;
        gtsam::imuBias::ConstantBias prev_bias = initialBias;
        prop_state = preintegratedImu.predict(prev_state, prev_bias);

        // Add the IMU Preintegration
        graph.add(gtsam::CombinedImuFactor(gtsam::Symbol('x', 0), gtsam::Symbol('v', 0), 
                                gtsam::Symbol('x', 1), gtsam::Symbol('v', 1), 
                                gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), preintegratedImu));


        initialEstimate.insert(gtsam::Symbol('x',1), prop_state.pose());
        initialEstimate.insert(gtsam::Symbol('v',1), prop_state.v());
        initialEstimate.insert(gtsam::Symbol('b',1), prev_bias);

        // Add bias factor to the graph
        auto biasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // Small noise model for bias
        graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), gtsam::imuBias::ConstantBias(), biasNoise));

        graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 1),prop_state.pose()));

        graph.add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', 1),prop_state.v()));

    }

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 100;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    gtsam::Values result = optimizer.optimize();

    // Extract optimized pose
    gtsam::Pose3 optimizedPose = result.at<gtsam::Pose3>(gtsam::Symbol('x', 1));
    if (currentIMUData)
    {
        gtsam::Vector3 camVelocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', 1));
        zedPtr->mCameraLeft->mNewVelocity = Eigen::Vector3d(camVelocity.data());

        initialBias = result.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', 1));
    }

    estimPoseInv.block<3, 3>(0, 0) = optimizedPose.rotation().matrix();
    estimPoseInv.block<3, 1>(0, 3) = optimizedPose.translation();
    estimPose = estimPoseInv.inverse();
    int nIn = 0, nStereo = 0;
    nStereo = findOutliersR(estimPose, activeMapPoints, keysLeft, matchesIdxs, 7.815, MPsOutliers, std::vector<float>(prevS, 1.0f), nIn);

    return std::pair<int, int>(nIn, nStereo);
}

std::pair<int, int> FeatureTracker::estimatePoseGTSAMMono(std::vector<MapPoint *> &activeMapPoints, TrackedKeys &keysLeft, std::vector<std::pair<int, int>> &matchesIdxs, Eigen::Matrix4d &estimPose, std::vector<bool> &MPsOutliers, const bool first)
{
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


    gtsam::Pose3 startPose(
        gtsam::Rot3(currentPoseInv.block<3, 3>(0, 0)),
        gtsam::Point3(currentPoseInv.block<3, 1>(0, 3))
    );

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
        if (!mp)
            continue;
        if (mp->GetIsOutlier()) 
            continue;

        Eigen::Vector3d point = mp->getWordPose3d();
        gtsam::Point3 gtsamPoint(point);

        if (keyPos.first >= 0) 
        { // Left image point
            const int nIdx = keyPos.first;
            if (!mp->inFrame)
                continue;

            // For the left observation
            Eigen::Vector2d obs(keysLeft.keyPoints[nIdx].pt.x, keysLeft.keyPoints[nIdx].pt.y);
            gtsam::Point2 observation(obs);
            const int octL = keysLeft.keyPoints[nIdx].octave;
            double sigma = 1.0 / feLeft->InvSigmaFactor[octL];
            noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));

            auto factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                observation, noiseModel, gtsam::Symbol('x', 1), gtsam::Symbol('l', i), K);

            graph.add(factor);

            // Add landmark initial estimate and any prior constraints
            initialEstimate.insert(gtsam::Symbol('l', i), gtsamPoint);
            graph.add(gtsam::NonlinearEquality<gtsam::Point3>(gtsam::Symbol('l', i), gtsamPoint));
            
        }
        else
            continue;
    }

    // IMU Preintegration 

    const double gyroNoiseDensity = currentIMUData->mGyroNoiseDensity;
    double gyroRandomWalk = currentIMUData->mGyroRandomWalk;
    double accelNoiseDensity = currentIMUData->mAccelNoiseDensity;
    double accelRandomWalk = currentIMUData->mAccelRandomWalk;
    
    const Eigen::Vector3d gravity = zedPtr->mCameraLeft->mIMUGravity;

    auto preintegrationParams = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(gravity.data())));
    
    preintegrationParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(gyroNoiseDensity, 2);
    preintegrationParams->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(accelNoiseDensity, 2);
    preintegrationParams->biasOmegaCovariance = gtsam::Matrix33::Identity() * pow(gyroRandomWalk, 2);
    preintegrationParams->biasAccCovariance = gtsam::Matrix33::Identity() * pow(accelRandomWalk, 2);

    preintegrationParams->integrationCovariance = gtsam::I_3x3 * 1e-5;
    
    const Eigen::Matrix4d& TBodyToCam = zedPtr->mCameraLeft->TBodyToCam;
    gtsam::Pose3 TBodyToCamGtsam(
        gtsam::Rot3(TBodyToCam.block<3, 3>(0, 0)),
        gtsam::Point3(TBodyToCam.block<3, 1>(0, 3))
    );

    preintegrationParams->body_P_sensor = TBodyToCamGtsam;


    gtsam::PreintegratedCombinedMeasurements preintegratedImu(preintegrationParams, initialBias);

    const size_t IMUDataSize{currentIMUData->mTimestamps.size()};
    double dt {1.0/currentIMUData->mHz};
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
        preintegratedImu.integrateMeasurement(accel, angleVel, dt);
    }

    gtsam::Vector3 priorVel(zedPtr->mCameraLeft->mVelocity.data());
    initialEstimate.insert(gtsam::Symbol('v', 0), priorVel);
    graph.add(gtsam::NonlinearEquality<gtsam::Vector3>(gtsam::Symbol('v', 0), priorVel));
    initialEstimate.insert(gtsam::Symbol('b', 0), initialBias);
    graph.add(gtsam::NonlinearEquality<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', 0), initialBias));


    gtsam::NavState prev_state(startPose, priorVel);
    gtsam::NavState prop_state = prev_state;
    gtsam::imuBias::ConstantBias prev_bias = initialBias;
    prop_state = preintegratedImu.predict(prev_state, prev_bias);

    graph.add(gtsam::CombinedImuFactor(gtsam::Symbol('x', 0), gtsam::Symbol('v', 0), 
                               gtsam::Symbol('x', 1), gtsam::Symbol('v', 1), 
                               gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), preintegratedImu));


    initialEstimate.insert(gtsam::Symbol('x',1), prop_state.pose());
    initialEstimate.insert(gtsam::Symbol('v',1), prop_state.v());
    initialEstimate.insert(gtsam::Symbol('b',1), prev_bias);

    auto biasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); 
    graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        gtsam::Symbol('b', 0), gtsam::Symbol('b', 1), gtsam::imuBias::ConstantBias(), biasNoise));

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 1),prop_state.pose()));

    graph.add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', 1),prop_state.v()));

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 100;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    gtsam::Values result = optimizer.optimize();

    // Extract optimized pose
    gtsam::Pose3 optimizedPose = result.at<gtsam::Pose3>(gtsam::Symbol('x', 1));

    gtsam::Vector3 camVelocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', 1));
    zedPtr->mCameraLeft->mNewVelocity = Eigen::Vector3d(camVelocity.data());

    initialBias = result.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', 1));

    estimPoseInv.block<3, 3>(0, 0) = optimizedPose.rotation().matrix();
    estimPoseInv.block<3, 1>(0, 3) = optimizedPose.translation();
    estimPose = estimPoseInv.inverse();
    int nIn = 0, nStereo = 0;
    nStereo = findOutliersMono(estimPose, activeMapPoints, keysLeft, matchesIdxs, 7.815, MPsOutliers, std::vector<float>(prevS, 1.0f), nIn);

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

int FeatureTracker::findOutliersMono(const Eigen::Matrix4d &estimPose, std::vector<MapPoint *> &activeMapPoints, TrackedKeys &keysLeft, std::vector<std::pair<int, int>> &matchesIdxs, const double thres, std::vector<bool> &MPsOutliers, const std::vector<float> &weights, int &nInliers)
{
    for (size_t i {0}, end{matchesIdxs.size()}; i < end; i++)
    {
        std::pair<int,int>& keyPos = matchesIdxs[i];
        MapPoint* mp = activeMapPoints[i];
        if ( !mp )
            continue;
        Eigen::Vector4d p4d = mp->getWordPose4d();
        int nIdx;
        cv::Point2f obs;
        if ( keyPos.first >= 0 )
        {
            if  ( !mp->inFrame )
                continue;
            p4d = estimPose * p4d;
            nIdx = keyPos.first;
            obs = keysLeft.keyPoints[nIdx].pt;
        }
        else
            continue;
        const int octL = keysLeft.keyPoints[nIdx].octave;
        const double weight = (double)feLeft->InvSigmaFactor[octL];
        bool outlier = check2dError(p4d, obs, thres, weight);
        MPsOutliers[i] = outlier;
        if ( !outlier )
        {
            nInliers++;
        }
    }

    return nInliers;
}

bool FeatureTracker::worldToFrame(MapPoint* mp, const bool right, const Eigen::Matrix4d& predPoseInv, const Eigen::Matrix4d& tempPose)
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

void FeatureTracker::insertKeyFrame(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const int nStereo, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm)
{
    Eigen::Matrix4d referencePose = latestKF->pose.getInvPose() * estimPose;
    KeyFrame* kF = new KeyFrame(zedPtr, referencePose, estimPose, leftIm, rleftIm,map->kIdx, curFrame);
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
    if ( map->keyFrames.size() > 3 && !map->LCStart )
        map->keyFrameAdded = true;

}

void FeatureTracker::insertKeyFrameMono(TrackedKeys& keysLeft, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs, const Eigen::Matrix4d& estimPose, std::vector<bool>& MPsOutliers, cv::Mat& leftIm, cv::Mat& rleftIm)
{
    Eigen::Matrix4d referencePose = latestKF->pose.getInvPose() * estimPose;
    KeyFrame* kF = new KeyFrame(zedPtr, referencePose, estimPose, leftIm, rleftIm,map->kIdx, curFrame);
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
    kF->localMapPoints.resize(keysLeft.keyPoints.size(), nullptr);
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
        trckedKeys++;
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
}

double FeatureTracker::calculateParallaxAngle(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2) 
{
    Eigen::Matrix3d rotation1 = pose1.block<3,3>(0,0);
    Eigen::Matrix3d rotation2 = pose2.block<3,3>(0,0);
    
    Eigen::Matrix3d relativeRotation = rotation1.transpose() * rotation2;
    
    // Compute the angle of rotation 
    double trace = relativeRotation.trace();
    double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
    
    return abs(angle); // Angle in radians
}

bool FeatureTracker::isParallaxSufficient(const Eigen::Matrix4d& pose1, const Eigen::Matrix4d& pose2, double threshold) 
{
    double parallaxAngle = calculateParallaxAngle(pose1, pose2);
    return parallaxAngle > threshold;
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

void FeatureTracker::removeOutOfFrameMPs(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints)
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
        bool c1 = worldToFrame(mp, false, toCamera, temp);
        bool c2 = worldToFrame(mp, true, toRCamera, tempR);
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

void FeatureTracker::removeOutOfFrameMPsMono(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints)
{
    const size_t end{activeMapPoints.size()};
    Eigen::Matrix4d toCamera = predNPose.inverse();
    int j {0};
    Eigen::Matrix4d temp = currCamPose.inverse() * predNPose;
    for ( size_t i {0}; i < end; i++)
    {
        MapPoint* mp = activeMapPoints[i];
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        bool c1 = worldToFrame(mp, false, toCamera, temp);
        if (c1)
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

void FeatureTracker::PredictMPsPosition(const Eigen::Matrix4d& currCamPose, const Eigen::Matrix4d& predNPose, std::vector<MapPoint*>& activeMapPoints, std::vector<int>& matchedIdxsL, std::vector<int>& matchedIdxsR, std::vector<std::pair<int,int>>& matchesIdxs, std::vector<bool> &MPsOutliers)
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
        if (!worldToFrame(mp, false, toCamera, temp))
        {
            if ( keyPos.first >=0 )
            {
                matchedIdxsL[keyPos.first] = -1;
                keyPos.first = -1;
            }
        }
        if (!worldToFrame(mp, true, toRCamera, tempR))
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
    
    const Eigen::Vector3d gravity = zedPtr->mCameraLeft->mIMUGravity;

    auto preintegrationParams = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(gravity.data())));

    preintegrationParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(gyroNoiseDensity, 2);
    preintegrationParams->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(accelNoiseDensity, 2);
    preintegrationParams->biasOmegaCovariance = gtsam::Matrix33::Identity() * pow(gyroRandomWalk, 2);
    preintegrationParams->biasAccCovariance = gtsam::Matrix33::Identity() * pow(accelRandomWalk, 2);

    preintegrationParams->integrationCovariance = gtsam::I_3x3 * 1e-5;

    // preintegrationParams->biasAccOmegaInt = gtsam::Matrix::Identity(6,6)*1e-5;

    const Eigen::Matrix4d& TBodyToCam = zedPtr->mCameraLeft->TBodyToCam;
    gtsam::Pose3 TBodyToCamGtsam(
        gtsam::Rot3(TBodyToCam.block<3, 3>(0, 0)),
        gtsam::Point3(TBodyToCam.block<3, 1>(0, 3))
    );

    preintegrationParams->body_P_sensor = TBodyToCamGtsam;

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

        preintegratedImu.integrateMeasurement(accel, angleVel, dt);
    }

    gtsam::Vector3 priorVel(predVelocity);

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
    predVelocity = prop_state.v();
    Eigen::Matrix4d predPoseEigen = Eigen::Matrix4d::Identity();
    predPoseEigen.block<3, 3>(0, 0) = predPose.rotation().matrix();
    predPoseEigen.block<3, 1>(0, 3) = predPose.translation();
    return predPoseEigen;

}

void FeatureTracker::TrackImage(const cv::Mat& leftRect, const cv::Mat& rightRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr /* = nullptr*/)
{
    if (IMUDataptr)
        currentIMUData = IMUDataptr;
    curFrame = frameNumb;
    
    // Change Poses created after Bundle Adjustment if BA has finished
    if ( map->LBADone )
    {
        std::lock_guard<std::mutex> lock(map->mapMutex);
        const int endIdx = map->endLBAIdx;
        changePosesLCA(endIdx);
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

    // first frame initialize map with stereo matches
    if ( curFrame == 0 )
    {
        extractORBAndStereoMatch(leftIm, rightIm, keysLeft);

        initializeMap(keysLeft);

        return;
    }

    // predNPoseInv = PredictNextPoseIMU();
    // predNPose = predNPoseInv.inverse();

    Eigen::Matrix4d estimPose = predNPoseInv;

    // remove out of frame MPs according to prediction
    std::vector<MapPoint *> activeMpsTemp;
    {
    std::lock_guard<std::mutex> lock(map->mapMutex);
    removeOutOfFrameMPs(zedPtr->mCameraPose.pose, predNPose, activeMapPoints);
    activeMpsTemp = activeMapPoints;
    }

    

    extractORBAndStereoMatch(leftIm, rightIm, keysLeft);

    std::vector<int> matchedIdxsL(keysLeft.keyPoints.size(), -1);
    std::vector<int> matchedIdxsR(keysLeft.rightKeyPoints.size(), -1);
    std::vector<std::pair<int,int>> matchesIdxs(activeMpsTemp.size(), std::make_pair(-1,-1));
    
    std::vector<bool> MPsOutliers(activeMpsTemp.size(),false);

    // Eigen::Matrix4d estimPose = PredictNextPoseIMU();

    float rad {10.0};


    // Match by projection and radius
    // the feature is assigned a key according its position in the frame
    // the 3D point is projected to the frame (using its predicted next pose)
    // then for each feature search around a radius for potential matches and select the best one
    if ( curFrame == 1 )
        rad = 120;
    else
        rad = 10;

    std::pair<int,int> nIn(-1,-1);
    int prevIn = -1;
    float prevrad = rad;
    bool toBreak {false};
    int countIte {0};
    // repeat until we have a lot of inliers (until 3 times)
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

    // repeat with the estimated pose
    PredictMPsPosition(zedPtr->mCameraPose.pose, estimPose.inverse(), activeMpsTemp, matchedIdxsL, matchedIdxsR, matchesIdxs, MPsOutliers);

    rad = 4;
    fm.matchByProjectionRPred(activeMpsTemp, keysLeft, matchedIdxsL, matchedIdxsR, matchesIdxs, rad);

    std::pair<int,int> nStIn = estimatePoseGTSAM(activeMpsTemp, keysLeft, matchesIdxs, estimPose, MPsOutliers, false);


    // draw the tracked Keypoints
    std::vector<cv::KeyPoint> lp;
    lp.reserve(matchesIdxs.size());
    for ( size_t i{0}; i < matchesIdxs.size(); i++)
    {
        const std::pair<int,int>& keyPos = matchesIdxs[i];
        if ( keyPos.first >= 0 )
        {
            lp.emplace_back(keysLeft.keyPoints[keyPos.first]);
        }
    }
    drawKeys("VSLAM : Tracked KeyPoints", realLeftIm, lp);

    poseEst = estimPose.inverse();

    // insert Keyframe if needed
    // adding new 3d points
    insertKeyFrameCount ++;
    if ( ((nStIn.second < minNStereo || insertKeyFrameCount >= keyFrameCountEnd) && nStIn.first < precCheckMatches * lastKFTrackedNumb))
    {
        insertKeyFrameCount = 0;
        insertKeyFrame(keysLeft, matchedIdxsL,matchesIdxs, nStIn.second, poseEst, MPsOutliers, leftIm, realLeftIm);
        // if keyframe is added local BA begins
    }
    else
        addFrame(poseEst);


    updatePoses();

    setActiveOutliers(activeMpsTemp,MPsOutliers, matchesIdxs);

    currentIMUData = nullptr;
    zedPtr->mCameraLeft->mVelocity = zedPtr->mCameraLeft->mNewVelocity;
}

void FeatureTracker::TrackImageMonoIMU(const cv::Mat& leftRect, const int frameNumb, std::shared_ptr<IMUData> IMUDataptr)
{
    if (IMUDataptr)
        currentIMUData = IMUDataptr;
    curFrame = frameNumb;

    cv::Mat realLeftIm;
    cv::Mat leftIm;

    realLeftIm = leftRect;
    

    if(realLeftIm.channels()==3)
    {
        cvtColor(realLeftIm,leftIm,cv::COLOR_BGR2GRAY);
    }
    else if(realLeftIm.channels()==4)
    {
        cvtColor(realLeftIm,leftIm,cv::COLOR_BGRA2GRAY);
    }
    else
    {
        leftIm = realLeftIm.clone();
    }
    
    TrackedKeys keysLeft;


    if ( curFrame == 0 )
    {
        feLeft->extractKeysNew(leftIm, keysLeft.keyPoints, keysLeft.Desc);

        assignKeysToGrids(keysLeft, keysLeft.keyPoints, keysLeft.lkeyGrid, zedPtr->mWidth, zedPtr->mHeight);

        keysLeft.estimatedDepth.resize(keysLeft.keyPoints.size(), -1.0f);

        initializeMono(keysLeft);
        return;
    }

    predNPoseInv = PredictNextPoseIMU();
    predNPose = predNPoseInv.inverse();

    Eigen::Matrix4d estimPose = predNPoseInv;

    if(!isParallaxSufficient(zedPtr->mCameraPose.pose, predNPose, parallaxThreshold) && !monoInitialized)
        return;
    else if (!secondKF)
    {
        feLeft->extractKeysNew(leftIm, keysLeft.keyPoints, keysLeft.Desc);

        assignKeysToGrids(keysLeft, keysLeft.keyPoints, keysLeft.lkeyGrid, zedPtr->mWidth, zedPtr->mHeight);

        keysLeft.estimatedDepth.resize(keysLeft.keyPoints.size(), -1.0f);

        initializeMono(keysLeft);

        poseEst = predNPose;
        updatePoses();
        secondKF = true;
        return;
    }

    feLeft->extractKeysNew(leftIm, keysLeft.keyPoints, keysLeft.Desc);

    assignKeysToGrids(keysLeft, keysLeft.keyPoints, keysLeft.lkeyGrid, zedPtr->mWidth, zedPtr->mHeight);

    std::vector<MapPoint *> activeMpsTemp;
    std::vector<int> matchedIdxsL(keysLeft.keyPoints.size(), -1);
    std::vector<std::pair<int,int>> matchesIdxs(activeMapPoints.size(), std::make_pair(-1,-1));
    
    std::vector<bool> MPsOutliers(activeMapPoints.size(),false);

    if (!monoInitialized)
    {
        insertKeyFrameMono(keysLeft, matchedIdxsL,matchesIdxs, predNPose, MPsOutliers, leftIm, realLeftIm);

        std::vector<KeyFrame *> actKeyF;
        KeyFrame* lastKF = map->keyFrames.at(map->kIdx - 1);
        actKeyF.reserve(20);
        actKeyF.emplace_back(lastKF);
        actKeyF = map->allFramesPoses;
        std::vector<MapPoint*> pointsToAdd(lastKF->keys.keyPoints.size(), nullptr);
        addMappointsMono(pointsToAdd, actKeyF, matchedIdxsL, matchesIdxs);
        addNewMapPoints(pointsToAdd);
        monoInitialized = true;

        poseEst = predNPose;
        updatePoses();
        return;
    }
    else
    {
    std::lock_guard<std::mutex> lock(map->mapMutex);
    // removeOutOfFrameMPsMono(zedPtr->mCameraPose.pose, predNPose, activeMapPoints);
    activeMpsTemp = activeMapPoints;

    matchesIdxs = std::vector<std::pair<int,int>>(activeMapPoints.size(), std::make_pair(-1,-1));
    MPsOutliers = std::vector<bool>(activeMapPoints.size(),false);

    }

    // std::vector<int> matchedIdxsL(keysLeft.keyPoints.size(), -1);
    
    // std::vector<bool> MPsOutliers(activeMpsTemp.size(),false);

    float rad {10.0};

    if ( !monoInitialized )
        rad = 120;
    else
        rad = 10;

    rad = 120;

    std::pair<int,int> nIn(-1,-1);
    int prevIn = -1;
    float prevrad = rad;
    bool toBreak {false};
    int countIte {0};
    if (true) // monoInitialized
    {
        while ( nIn.first < minInliers )
        {
            countIte++;
            fm.matchByProjectionMono(activeMpsTemp, keysLeft, matchedIdxsL, matchesIdxs, rad);

            nIn = estimatePoseGTSAMMono(activeMpsTemp, keysLeft, matchesIdxs, estimPose, MPsOutliers, true);

            if ( nIn.first < minInliers  && !toBreak )
            {
                estimPose = predNPoseInv;
                matchedIdxsL = std::vector<int>(keysLeft.keyPoints.size(), -1);
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
    }
    else
    {
        fm.matchByProjectionMono(activeMpsTemp, keysLeft, matchedIdxsL, matchesIdxs, rad);
    }

    // newPredictMPs(zedPtr->mCameraPose.pose, estimPose.inverse(), activeMpsTemp, matchedIdxsL, matchedIdxsR, matchesIdxs, MPsOutliers);

    // rad = 4;
    // fm.matchByProjectionRPred(activeMpsTemp, keysLeft, matchedIdxsL, matchedIdxsR, matchesIdxs, rad);

    // std::pair<int,int> nStIn = estimatePoseGTSAM(activeMpsTemp, keysLeft, matchesIdxs, estimPose, MPsOutliers, false);

    std::vector<cv::KeyPoint> lp;
    lp.reserve(matchesIdxs.size());
    for ( size_t i{0}; i < matchesIdxs.size(); i++)
    {
        const std::pair<int,int>& keyPos = matchesIdxs[i];
        if ( keyPos.first >= 0 )
        {
            lp.emplace_back(keysLeft.keyPoints[keyPos.first]);
        }
    }
    drawKeys("VSLAM : Tracked KeyPoints", realLeftIm, lp);
    cv::waitKey(0);

    poseEst = estimPose.inverse();

    insertKeyFrameCount ++;
    if ( ((insertKeyFrameCount >= keyFrameCountEnd) && nIn.first < precCheckMatches * lastKFTrackedNumb) || numOfMonoMPs < minNStereo)
    {
        insertKeyFrameCount = 0;
        insertKeyFrameMono(keysLeft, matchedIdxsL,matchesIdxs, poseEst, MPsOutliers, leftIm, realLeftIm);

        std::vector<KeyFrame *> actKeyF;
        KeyFrame* lastKF = map->keyFrames.at(map->kIdx - 1);
        actKeyF.reserve(20);
        actKeyF.emplace_back(lastKF);
        lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
        std::vector<MapPoint*> pointsToAdd(lastKF->keys.keyPoints.size(), nullptr);
        addMappointsMono(pointsToAdd, actKeyF, matchedIdxsL, matchesIdxs);
        addNewMapPoints(pointsToAdd);
    }
    else
        addFrame(poseEst);


    updatePoses();

    setActiveOutliers(activeMpsTemp,MPsOutliers, matchesIdxs);

    currentIMUData = nullptr;
    zedPtr->mCameraLeft->mVelocity = zedPtr->mCameraLeft->mNewVelocity;
}

void FeatureTracker::addMappointsMono(std::vector<MapPoint*>& pointsToAdd, std::vector<KeyFrame *>& actKeyF, std::vector<int>& matchedIdxsL, std::vector<std::pair<int,int>>& matchesIdxs)
{
    auto& lastKF = actKeyF.front();

    using namespace gtsam;

    const size_t keyPointsSize = lastKF->keys.keyPoints.size();

    std::vector<std::vector<std::pair<KeyFrame*,int>>> keyframeIdxMatchs(keyPointsSize);
    for (size_t i = 0; i < keyPointsSize; ++i)
    {
        keyframeIdxMatchs[i].reserve(actKeyF.size());
        keyframeIdxMatchs[i].emplace_back(std::make_pair(lastKF, i));
    }

    for(auto& keyF : actKeyF)
    {
        if(keyF->numb == lastKF->numb)
            continue;
        // if (!isParallaxSufficient(lastKF->pose.pose, keyF->pose.pose, parallaxThreshold))
        //     continue;
        auto& lastKeys = lastKF->keys;
        auto& actKeys = keyF->keys;
        fm.matchByRadius(lastKeys, actKeys, matchedIdxsL, matchesIdxs, 120, keyframeIdxMatchs, keyF);
    }


    unsigned long mpIdx {map->pIdx};
    for (size_t i = 0; i < keyPointsSize; ++i)
    {
        auto& keys = keyframeIdxMatchs[i];
        if (keys.size() <= 1)
            continue;
        Eigen::Vector4d p4d;
        if (!calculateMPFromMono(p4d,pointsToAdd, keys))
            continue;
        std::cout << "Point : " << p4d << std::endl;
        const TrackedKeys& temp = lastKF->keys; 
        const int idx = keys[0].second;

        MapPoint* mp = new MapPoint(p4d, temp.Desc.row(idx),temp.keyPoints[idx], lastKF->numb, mpIdx);
        mpIdx++;

        for (size_t j {0}, end{keys.size()}; j < end; j++)
        {
            KeyFrame* kFCand = keys[j].first;
            const std::pair<int,int>& keyPos = std::make_pair(keys[j].second,-1);
            mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
        }
        mp->update(lastKF);
        mp->monoInitialized = true;
        pointsToAdd[i] = mp;
        
    }
    

}

void FeatureTracker::addNewMapPoints(std::vector<MapPoint*>& pointsToAdd)
{
    int newMapPointsCount {0};
    std::lock_guard<std::mutex> lock(map->mapMutex);
    for (size_t i{0}, end{pointsToAdd.size()}; i < end; i++ )
    {
        MapPoint* newMp = pointsToAdd[i];
        if (!newMp )
            continue;
        std::unordered_map<KeyFrame *, std::pair<int, int>>::iterator it, endMp(newMp->kFMatches.end());
        for ( it = newMp->kFMatches.begin(); it != endMp; it++)
        {
            KeyFrame* kFCand = it->first;
            std::pair<int,int>& keyPos = it->second;
            newMp->addConnectionMono(kFCand, keyPos);
        }
        map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
        newMapPointsCount ++;
    }

}

bool FeatureTracker::calculateMPFromMono(Eigen::Vector4d& p4d, std::vector<MapPoint*> pointsToAdd, std::vector<std::pair<KeyFrame*,int>>& keys)
{
    using namespace gtsam;

    std::vector<gtsam::Pose3> cameraPoses;
    gtsam::Point2Vector observations;
    auto K = boost::make_shared<Cal3_S2>(fx, fy, 0, cx, cy);  
    for (const auto& pair : keys)
    {
        const auto& KF = pair.first;
        const auto& idx = pair.second;
        const auto& key = KF->keys.keyPoints[idx];

        Point2 measured(key.pt.x, key.pt.y);
        observations.emplace_back(measured);
        cameraPoses.emplace_back(KF->pose.poseInverse);
    }

    std::optional<gtsam::Point3> resu;
    try
    {
        resu = gtsam::triangulatePoint3<Cal3_S2>(cameraPoses, K,observations);
    }
    catch(const gtsam::TriangulationCheiralityException& e)
    {
        return false;
    }
    catch(const gtsam::TriangulationUnderconstrainedException& e)
    {
        return false;
    }

    if (resu.has_value())
    {
        gtsam::Point3 p3 = resu.value();
        p4d = Eigen::Vector4d(p3.x(), p3.y(), p3.z(), 1.0);
        return true;
    }
    return false;

}

void FeatureTracker::drawKeys(const char* com, cv::Mat& im, std::vector<cv::KeyPoint>& keys)
{
    cv::Mat outIm = im.clone();
    int count {0};
    for (auto& key:keys)
    {
        cv::circle(outIm, key.pt,3,cv::Scalar(255,0,0),2);
        count++;
    }
    cv::imshow(com, outIm);
    cv::waitKey(1);
}

void FeatureTracker::updatePoses()
{
    Eigen::Matrix4d prevWPoseInv = zedPtr->mCameraPose.poseInverse;
    Eigen::Matrix4d referencePose = lastKFPoseInv * poseEst;
    zedPtr->mCameraPose.setPose(poseEst);
    zedPtr->mCameraPose.refPose = referencePose;
    predNPoseRef = prevWPoseInv * poseEst;
    predNPose = poseEst * predNPoseRef;
    predNPoseInv = predNPose.inverse();
}

} // namespace GTSAM_VIOSLAM