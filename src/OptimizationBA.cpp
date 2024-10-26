#include "OptimizationBA.h"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/triangulation.h>
#include <optional>

namespace TII
{

LocalMapper::LocalMapper(std::shared_ptr<Map> _map, std::shared_ptr<StereoCamera> _zedPtr, std::shared_ptr<FeatureMatcher> _fm) : map(_map), zedPtr(_zedPtr), fm(_fm), fx(_zedPtr->mCameraLeft->fx), fy(_zedPtr->mCameraLeft->fy), cx(_zedPtr->mCameraLeft->cx), cy(_zedPtr->mCameraLeft->cy)
{

}

void LocalMapper::calcProjMatricesR(std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& projMatrices, std::vector<KeyFrame*>& actKeyF)
{

    Eigen::Matrix<double,3,3>& K = zedPtr->mCameraLeft->intrinsics;
    Eigen::Matrix4d projL = Eigen::Matrix4d::Identity();
    projL.block<3,3>(0,0) = K;
    Eigen::Matrix4d projR = Eigen::Matrix4d::Identity();
    projR.block<3,3>(0,0) = K;
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        Eigen::Matrix<double,4,4> extr2 = (*it)->pose.poseInverse;
        extr2 = projL * extr2;
        Eigen::Matrix<double,3,4> extr = extr2.block<3,4>(0,0);
        Eigen::Matrix<double,4,4> extrRight = ((*it)->pose.pose * zedPtr->extrinsics).inverse();
        extrRight =  projR * extrRight;
        Eigen::Matrix<double,3,4> extrR = extrRight.block<3,4>(0,0);
        projMatrices.emplace(*it, std::make_pair(extr, extrR));
    }
}

void LocalMapper::processMatchesR(std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>>& allProjMatrices, std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& points)
{
    proj_matrices.reserve(matchesOfPoint.size());
    points.reserve(matchesOfPoint.size());
    std::vector<std::pair<KeyFrame *, std::pair<int, int>>>::const_iterator it, end(matchesOfPoint.end());
    for ( it = matchesOfPoint.begin(); it != end; it++)
    {
        KeyFrame* kF = it->first;
        const TrackedKeys& keys = kF->keys;
        const std::pair<int,int>& keyPos = it->second;
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.keyPoints[keyPos.first].pt.x, (double)keys.keyPoints[keyPos.first].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).first);
        }

        if ( keyPos.second >= 0 )
        {
            Eigen::Vector2d vec2d((double)keys.rightKeyPoints[keyPos.second].pt.x, (double)keys.rightKeyPoints[keyPos.second].pt.y);
            points.emplace_back(vec2d);
            proj_matrices.emplace_back(allProjMatrices.at(kF).second);
        }
    }
}

bool LocalMapper::checkReprojErrNewR(KeyFrame* lastKF, Eigen::Vector4d& calcVec, std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, std::vector<Eigen::Vector2d>& pointsVec)
{
    int count {0};
    bool correctKF {false};
    const unsigned long lastKFNumb = lastKF->numb;
    int projCount {0};
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        std::pair<KeyFrame *, std::pair<int, int>>& match = matchesOfPoint[i];
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const TrackedKeys& keys = kFCand->keys;
        
        std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        const unsigned long kFCandNumb {kFCand->numb};
        bool cor {false};
        if ( keyPos.first >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.keyPoints[keyPos.first].octave;
            const double weight = (double)kFCand->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.first = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( keyPos.second >= 0 )
        {
            Eigen::Vector3d p3dnew = proj_matrices[projCount] * calcVec;
            p3dnew = p3dnew/p3dnew(2);
            double err1,err2;
            err1 = pointsVec[projCount](0) - p3dnew(0);
            err2 = pointsVec[projCount](1) - p3dnew(1);
            const int oct = keys.rightKeyPoints[keyPos.second].octave;
            const double weight = lastKF->sigmaFactor[oct];
            float err = err1*err1 + err2*err2;
            projCount ++;

            if ( err > reprjThreshold * weight )
            {
                keyPos.second = -1;
            }
            else
            {
                matchesOfPoint[count] = match;
                cor = true;
                if ( kFCandNumb == lastKFNumb )
                    correctKF = true;
            }
        }
        if ( cor )
            count++;
    }
    matchesOfPoint.resize(count);
    if ( count >= minCount  && correctKF )
        return true;
    else
        return false;
}

void LocalMapper::addMultiViewMapPointsR(const Eigen::Vector4d& posW, const std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint, std::vector<MapPoint*>& pointsToAdd, KeyFrame* lastKF, const size_t& mpPos)
{
    const TrackedKeys& temp = lastKF->keys; 
    static unsigned long mpIdx {map->pIdx};
    const unsigned long lastKFNumb {lastKF->numb};
    MapPoint* mp = nullptr;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        if ( kFCand->numb == lastKFNumb )
        {
            if ( keyPos.first >= 0 )
            {
                mp = new MapPoint(posW, temp.Desc.row(keyPos.first),temp.keyPoints[keyPos.first], lastKF->numb, mpIdx);
            }
            else if ( keyPos.second >= 0 )
            {
                mp = new MapPoint(posW, temp.rightDesc.row(keyPos.second),temp.rightKeyPoints[keyPos.second], lastKF->numb, mpIdx);
            }
            break;
        }
    }
    if ( !mp )
        return;

    mpIdx++;
    for (size_t i {0}, end{matchesOfPoint.size()}; i < end; i++)
    {
        KeyFrame* kFCand = matchesOfPoint[i].first;
        const std::pair<int,int>& keyPos = matchesOfPoint[i].second;
        mp->kFMatches.insert(std::pair<KeyFrame*, std::pair<int,int>>(kFCand, keyPos));
    }
    mp->update(lastKF);
    pointsToAdd[mpPos] = mp;
}

bool LocalMapper::triangulateCeresNew(Eigen::Vector3d& p3d, const std::vector<Eigen::Matrix<double, 3, 4>>& proj_matrices, const std::vector<Eigen::Vector2d>& obs, const Eigen::Matrix4d& lastKFPose, bool first, std::vector<Eigen::Matrix4d>& activePoses)
{
    using namespace gtsam;

    // Convert Eigen matrix to GTSAM Pose3 (camera pose)
    Pose3 camPose = Pose3(Rot3(lastKFPose.block<3, 3>(0, 0)), Point3(lastKFPose.block<3, 1>(0, 3)));

    // Create factor graph
    NonlinearFactorGraph graph;
    std::vector<gtsam::Pose3>projectionMatrices;
    std::vector<gtsam::PinholeCamera<Cal3_S2>>projectionMatricesCams;
    gtsam::Point2Vector observations;
    projectionMatrices.reserve(obs.size());
    observations.reserve(obs.size());
    auto K = boost::make_shared<Cal3_S2>(fx, fy, 0, cx, cy);  


    Values initialEstimate;
    // Add reprojection factors to the graph
    SharedNoiseModel loss_function = nullptr;
    if (first)
        loss_function = noiseModel::Isotropic::Sigma(2,1.0);
    for (size_t i = 0; i < obs.size(); ++i) 
    {
        // Extract projection matrix and observation
        const Eigen::Matrix<double, 3, 4>& proj_matrix = proj_matrices[i];
        const Eigen::Vector2d& observation = obs[i];

        const auto projMatW = proj_matrix * lastKFPose;
        Eigen::Matrix4d projMat4d = Eigen::Matrix4d::Identity();
        projMat4d.block<3,4>(0,0) = projMatW;
        const Eigen::Matrix4d projMat4dInv = activePoses[i];

        gtsam::Pose3 projMat(
            gtsam::Rot3(projMat4dInv.block<3, 3>(0, 0)),
            gtsam::Point3(projMat4dInv.block<3, 1>(0, 3))
        );

        projectionMatrices.emplace_back(projMat);

        // gtsam::PinholeCamera<Cal3_S2> cam1(projMat,K);
        // projectionMatricesCams.emplace_back(cam1);
        // Add the projection factor to the graph
        Point2 measured(observation[0], observation[1]);
        observations.emplace_back(measured);
        graph.add(boost::make_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
            measured, loss_function, Symbol('x', i), Symbol('l', 0), K));
    }

    // Initial estimate for 3D point and poses
    initialEstimate.insert(Symbol('l', 0), Point3(p3d[0], p3d[1], p3d[2]));

    for (size_t i = 0; i < obs.size(); ++i) 
    {
        initialEstimate.insert(Symbol('x', i), projectionMatrices[i]);
        graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', i), projectionMatrices[i]));
    }

    // Optimize the graph
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 20;
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    Values result = optimizer.optimize();

    // Extract optimized 3D point
    Point3 optimized_point = result.at<Point3>(Symbol('l', 0));
    p3d[0] = optimized_point.x();
    p3d[1] = optimized_point.y();
    p3d[2] = optimized_point.z();

    std::optional<gtsam::Point3> resu;
    try
    {
        resu = gtsam::triangulatePoint3<Cal3_S2>(projectionMatrices, K,observations);
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
        Eigen::Vector4d p4d2(p3(0), p3(1), p3(2), 1.0);
        p3d(0) = p4d2(0);
        p3d(1) = p4d2(1);
        p3d(2) = p4d2(2);
        return true;
        // std::cout << ":point : " << p4d2 << std::endl;
    }

    Eigen::Vector4d p4d(p3d(0), p3d(1), p3d(2), 1.0);
    p4d = lastKFPose * p4d;
    p3d(0) = p4d(0);
    p3d(1) = p4d(1);
    p3d(2) = p4d(2);
    return true;
}

void LocalMapper::addNewMapPoints(KeyFrame* lastKF, std::vector<MapPoint*>& pointsToAdd, std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs)
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
            newMp->addConnection(kFCand, keyPos);
        }
        map->activeMapPoints.emplace_back(newMp);
        map->addMapPoint(newMp);
        newMapPointsCount ++;
    }

}

void LocalMapper::calcAllMpsOfKFROnlyEst(std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>& matchedIdxs, KeyFrame* lastKF, const int kFsize, std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<float>& maxDistsScale)
{
    const size_t keysSize {lastKF->keys.keyPoints.size()};
    const size_t RkeysSize {lastKF->keys.rightKeyPoints.size()};
    const TrackedKeys& keys = lastKF->keys;
    p4d.reserve(keysSize + RkeysSize);
    maxDistsScale.reserve(keysSize + RkeysSize);
    for ( size_t i{0}; i < keysSize; i++)
    {
        MapPoint* mp = lastKF->localMapPoints[i];
        if ( !mp )
        {
            double zp;
            int rIdx {-1};
            if ( keys.estimatedDepth[i] > 0 )
            {
                rIdx = keys.rightIdxs[i];
                zp = (double)keys.estimatedDepth[i];
            }
            else
                continue;
            const double xp = (double)(((double)keys.keyPoints[i].pt.x-cx)*zp/fx);
            const double yp = (double)(((double)keys.keyPoints[i].pt.y-cy)*zp/fy);
            Eigen::Vector4d p4dcam(xp, yp, zp, 1);
            p4dcam = lastKF->pose.pose * p4dcam;
            p4d.emplace_back(p4dcam, std::make_pair((int)i, rIdx));
            Eigen::Vector3d pos = p4dcam.block<3,1>(0,0);
            pos = pos - lastKF->pose.pose.block<3,1>(0,3);
            float dist = pos.norm();
            int level = keys.keyPoints[i].octave;
            dist *= lastKF->scaleFactor[level];
            maxDistsScale.emplace_back(dist);
            continue;
        }
        if ( lastKF->unMatchedF[i] >= 0 )
            continue;
        const int rIdx {keys.rightIdxs[i]};
        p4d.emplace_back(mp->getWordPose4d(), std::make_pair((int)i, rIdx));
        Eigen::Vector3d pos = mp->getWordPose4d().block<3,1>(0,0);
        pos = pos - lastKF->pose.pose.block<3,1>(0,3);
        float dist = pos.norm();
        int level = keys.keyPoints[i].octave;
        dist *= lastKF->scaleFactor[level];
        maxDistsScale.emplace_back(dist);
    }
    const size_t allp4dsize {p4d.size()};
    matchedIdxs = std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>>(allp4dsize,std::vector<std::pair<KeyFrame*,std::pair<int, int>>>());
    for ( size_t i {0}; i < allp4dsize; i++)
    {
        matchedIdxs[i].reserve(10);
        std::pair<int,int> keyPos = p4d[i].second;
        matchedIdxs[i].emplace_back(lastKF,keyPos);
    }
}

void LocalMapper::predictKeysPosR(const TrackedKeys& keys, const Eigen::Matrix4d& camPose, const Eigen::Matrix4d& camPoseInv, const std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>>& p4d, std::vector<std::pair<cv::Point2f, cv::Point2f>>& predPoints)
{
    const Eigen::Matrix4d camPoseInvR = (camPose * zedPtr->extrinsics).inverse();

    const double fxr {zedPtr->mCameraRight->fx};
    const double fyr {zedPtr->mCameraRight->fy};
    const double cxr {zedPtr->mCameraRight->cx};
    const double cyr {zedPtr->mCameraRight->cy};
    
    const cv::Point2f noPoint(-1.-1);
    for ( size_t i {0}, end{p4d.size()}; i < end; i ++)
    {
        const Eigen::Vector4d& wp = p4d[i].first;

        Eigen::Vector4d p = camPoseInv * wp;
        Eigen::Vector4d pR = camPoseInvR * wp;

        if ( p(2) <= 0.0 || pR(2) <= 0.0)
        {
            predPoints.emplace_back(noPoint, noPoint);
            continue;
        }

        const double invZ = 1.0f/p(2);
        const double invZR = 1.0f/pR(2);

        double u {fx*p(0)*invZ + cx};
        double v {fy*p(1)*invZ + cy};

        double uR {fxr*pR(0)*invZR + cxr};
        double vR {fyr*pR(1)*invZR + cyr};

        const int w {zedPtr->mWidth};
        const int h {zedPtr->mHeight};

        cv::Point2f predL((float)u, (float)v), predR((float)uR, (float)vR);

        if ( u < 15 || v < 15 || u >= w - 15 || v >= h - 15 )
        {
            predL = noPoint;
        }
        if ( uR < 15 || vR < 15 || uR >= w - 15 || vR >= h - 15 )
        {
            predR = noPoint;
        }

        predPoints.emplace_back(predL, predR);

    }
}

void LocalMapper::triangulateNewPointsR(std::vector<KeyFrame *>& activeKF)
{
    const int kFsize {actvKFMaxSize};
    std::vector<KeyFrame *> actKeyF;
    actKeyF.reserve(kFsize);
    actKeyF = activeKF;
    KeyFrame* lastKF = actKeyF.front();
    const unsigned long lastKFIdx = lastKF->numb;
    std::vector<std::vector<std::pair<KeyFrame*,std::pair<int, int>>>> matchedIdxs;

    std::vector<std::pair<Eigen::Vector4d,std::pair<int,int>>> p4d;
    std::vector<float> maxDistsScale;
    calcAllMpsOfKFROnlyEst(matchedIdxs, lastKF, kFsize, p4d,maxDistsScale);
    {
    std::vector<KeyFrame*>::const_iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->numb == lastKFIdx)
            continue;
        std::vector<std::pair<cv::Point2f, cv::Point2f>> predPoints;
        // predict keys for both right and left camera
        predictKeysPosR(lastKF->keys, (*it)->pose.pose, (*it)->pose.poseInverse, p4d, predPoints);
        fm->matchByProjectionRPredLBA(lastKF, (*it), matchedIdxs, 4, predPoints, maxDistsScale, p4d);
        
    }
    }

    std::unordered_map<KeyFrame*, std::pair<Eigen::Matrix<double,3,4>,Eigen::Matrix<double,3,4>>> allProjMatrices;
    allProjMatrices.reserve(2 * actKeyF.size());
    calcProjMatricesR(allProjMatrices, actKeyF);

    std::vector<Eigen::Matrix4d> activePoses;
    for (const auto& kf : actKeyF)
    {
        activePoses.emplace_back(kf->getPose());
    }
    
    std::vector<MapPoint*> pointsToAdd;
    const size_t mpCandSize {matchedIdxs.size()};
    pointsToAdd.resize(mpCandSize,nullptr);
    int newMaPoints {0};
    for ( size_t i{0}; i < mpCandSize; i ++)
    {
        std::vector<std::pair<KeyFrame *, std::pair<int, int>>>& matchesOfPoint = matchedIdxs[i];
        if ((int)matchesOfPoint.size() < minCount)
            continue;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mat;
        std::vector<Eigen::Vector2d> pointsVec;
        processMatchesR(matchesOfPoint, allProjMatrices, proj_mat, pointsVec);
        Eigen::Vector4d vecCalc = lastKF->pose.getInvPose() * p4d[i].first;
        Eigen::Vector3d vec3d(vecCalc(0), vecCalc(1), vecCalc(2));
        if (!triangulateCeresNew(vec3d, proj_mat, pointsVec, lastKF->pose.pose, true, activePoses))
            continue;
        vecCalc(0) = vec3d(0);
        vecCalc(1) = vec3d(1);
        vecCalc(2) = vec3d(2);

        if ( !checkReprojErrNewR(lastKF, vecCalc, matchesOfPoint, proj_mat, pointsVec) )
            continue;

        addMultiViewMapPointsR(vecCalc, matchesOfPoint, pointsToAdd, lastKF, i);
        newMaPoints++;
    }
    std::cout << "New Mappoints added : " << newMaPoints << " ..." << std::endl;

    addNewMapPoints(lastKF, pointsToAdd, matchedIdxs);
}

bool LocalMapper::checkOutlier(const Eigen::Matrix3d& K, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

bool LocalMapper::checkOutlierR(const Eigen::Matrix3d& K, const Eigen::Matrix3d& qc1c2, const Eigen::Matrix<double,3,1>& tc1c2, const Eigen::Vector2d& obs, const Eigen::Vector3d posW,const Eigen::Vector3d& tcw, const Eigen::Quaterniond& qcw, const float thresh)
{
    Eigen::Vector3d posC = qcw * posW + tcw;
    posC = qc1c2 * posC + tc1c2;
    if ( posC(2) <= 0 )
        return true;
    Eigen::Vector3d pixel_pose = K * (posC);
    double error_u = obs[0] - pixel_pose[0] / pixel_pose[2];
    double error_v = obs[1] - pixel_pose[1] / pixel_pose[2];
    double error = (error_u * error_u + error_v * error_v);
    if (error > thresh)
        return true;
    else 
        return false;
    
}

void LocalMapper::localBAR(std::vector<KeyFrame *>& actKeyF)
{
    std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
    std::unordered_map<KeyFrame*, Eigen::Matrix4d> localKFs;
    std::unordered_map<KeyFrame*, Eigen::Matrix4d> fixedKFs;
    localKFs.reserve(actKeyF.size());
    fixedKFs.reserve(actKeyF.size());
    int blocks {0};
    unsigned long lastActKF {actKeyF.front()->numb};
    bool fixedKF {false};
    std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
    for ( it = actKeyF.begin(); it != end; it++)
    {
        (*it)->LBAID = lastActKF;
        localKFs[*it] = (*it)->pose.getPose();
        
    }
    for ( it = actKeyF.begin(); it != end; it++)
    {
        if ( (*it)->fixed )
            fixedKF = true;
        std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
        for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == (int)lastActKF )
                continue;
            
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == (int)lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = kFCand->pose.getPose();
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
        std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
        for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
        {
            MapPoint* mp = *itmp;
            if ( !mp )
                continue;
            if ( mp->GetIsOutlier() )
                continue;
            if ( mp->LBAID == (int)lastActKF )
                continue;
            
            std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
            for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
            {
                KeyFrame* kFCand = kf->first;
                const std::pair<int,int>& keyPos = kf->second;
                if ( keyPos.first >= 0 || keyPos.second < 0 )
                    continue;
                if ( !kFCand->keyF || kFCand->numb > lastActKF )
                    continue;
                if (kFCand->LBAID == (int)lastActKF )
                    continue;
                if (localKFs.find(kFCand) == localKFs.end())
                {
                    fixedKFs[kFCand] = kFCand->pose.getPose();
                    kFCand->LBAID = lastActKF;
                }
                blocks++;
            }
            allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
            (*itmp)->LBAID = lastActKF;
        }
    }
    if ( fixedKFs.size() == 0 && !fixedKF )
    {
        KeyFrame* lastKF = actKeyF.back();
        localKFs.erase(lastKF);
        fixedKFs[lastKF] = lastKF->pose.getPose();
    }
    std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
    wrongMatches.reserve(blocks);
    std::vector<bool>mpOutliers;
    mpOutliers.resize(allMapPoints.size());
    // bool first = true;
    const auto& K_eigen = zedPtr->mCameraLeft->intrinsics;

    using namespace gtsam;


    // Convert Eigen intrinsics to GTSAM intrinsics
    auto K = boost::make_shared<gtsam::Cal3_S2>(
        K_eigen(0, 0), K_eigen(1, 1), 0, K_eigen(0, 2), K_eigen(1, 2));



    // Initializing GTSAM graph and initial values
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    gtsam::SharedNoiseModel noiseModel = nullptr;
    
    gtsam::Values result;
    
    gtsam::Pose3 gtsamExtrinsics(
        gtsam::Rot3(zedPtr->extrinsics.block<3, 3>(0, 0)),
        gtsam::Point3(zedPtr->extrinsics.block<3, 1>(0, 3))
    );

    const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
    const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
    const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
    // for (size_t iterations{0}; iterations < 2; iterations++)
    // {

    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        int timesIn {0};
        int kfCount {0};
        bool mpIsOut {true};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
        for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++, kfCount++)
        {
            if ( !kf->first->keyF )
                    continue;
            if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
            {
                mpOutliers[mpCount] = true;
                break;
            }
            if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
            {
                continue;
            }
            if ( itmp->first->GetIsOutlier() )
                break;
            KeyFrame* kftemp = kf->first;
            TrackedKeys& keys = kftemp->keys;
            std::pair<int,int>& keyPos = kf->second;


            if ( kf->first->numb > lastActKF )
            {
                mpIsOut = false;
                continue;
            }
            timesIn ++;
            mpIsOut = false;
            bool close {false};
            Eigen::Vector3d point = itmp->first->getWordPose3d();
            gtsam::Point3 gtsamPoint(point);
            gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> factor;
            if ( keyPos.first >= 0 )
            {
                const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                
                gtsam::Point2 observation(obs2d);
                const int oct {obs.octave};

                double sigma = 1.0/kftemp->InvSigmaFactor[oct];
                noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
                factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    observation, noiseModel, gtsam::Symbol('x', kf->first->numb), gtsam::Symbol('l', itmp->first->idx), K);

                // Add landmark initial estimate
                if (!initialEstimate.exists(gtsam::Symbol('l', itmp->first->idx))) 
                    initialEstimate.insert(gtsam::Symbol('l', itmp->first->idx), gtsamPoint);

                close = keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                
                gtsam::Point2 observation(obs2d);
                
                const int oct {obs.octave};
                double sigma = 1.0/kftemp->InvSigmaFactor[oct];
                noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
                factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    observation, noiseModel, gtsam::Symbol('x', kf->first->numb), gtsam::Symbol('l', itmp->first->idx), K, gtsamExtrinsics);

                if (!initialEstimate.exists(gtsam::Symbol('l', itmp->first->idx))) 
                    initialEstimate.insert(gtsam::Symbol('l', itmp->first->idx), gtsamPoint);
            }

            if (localKFs.find(kf->first) != localKFs.end())
            {
                const auto& kfPoseInv = localKFs[kf->first];

                gtsam::Pose3 kfInitialPose(
                    gtsam::Rot3(kfPoseInv.block<3, 3>(0, 0)),
                    gtsam::Point3(kfPoseInv.block<3, 1>(0, 3))
                );

                graph.add(factor);

                if (!initialEstimate.exists(gtsam::Symbol('x', kf->first->numb))) 
                    initialEstimate.insert(gtsam::Symbol('x', kf->first->numb), kfInitialPose);
                if ( kf->first->fixed )
                {
                    graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', kf->first->numb),kfInitialPose));
                }
            }
            else if (fixedKFs.find(kf->first) != fixedKFs.end())
            {
                const auto& kfPoseInv = fixedKFs[kf->first];

                gtsam::Pose3 kfInitialPose(
                    gtsam::Rot3(kfPoseInv.block<3, 3>(0, 0)),
                    gtsam::Point3(kfPoseInv.block<3, 1>(0, 3))
                );

                graph.add(factor);
                
                if (!initialEstimate.exists(gtsam::Symbol('x', kf->first->numb))) 
                    initialEstimate.insert(gtsam::Symbol('x', kf->first->numb), kfInitialPose);
                graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', kf->first->numb),kfInitialPose));
            }
            else
            {
                continue;
            }
            if ( close )
            {
                if ( keyPos.second < 0 )
                    continue;
                const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
                Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
                gtsam::Point2 observation(obs2d);
                
                const int oct {obs.octave};
                double sigma = 1.0/kftemp->InvSigmaFactor[oct];
                noiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(sigma, sigma));
                factor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>(
                    observation, noiseModel, gtsam::Symbol('x', kf->first->numb), gtsam::Symbol('l', itmp->first->idx), K, gtsamExtrinsics);

                graph.add(factor);

                // Add landmark initial estimate
                if (!initialEstimate.exists(gtsam::Symbol('l', itmp->first->idx))) 
                    initialEstimate.insert(gtsam::Symbol('l', itmp->first->idx), gtsamPoint);

                if (localKFs.find(kf->first) != localKFs.end())
                {
                    const auto& kfPoseInv = localKFs[kf->first];

                    gtsam::Pose3 kfInitialPose(
                        gtsam::Rot3(kfPoseInv.block<3, 3>(0, 0)),
                        gtsam::Point3(kfPoseInv.block<3, 1>(0, 3))
                    );
                    if (!initialEstimate.exists(gtsam::Symbol('x', kf->first->numb))) 
                        initialEstimate.insert(gtsam::Symbol('x', kf->first->numb), kfInitialPose);
                    
                    if ( kf->first->fixed )
                    {
                        graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', kf->first->numb),kfInitialPose));
                    }
                }
                else if (fixedKFs.find(kf->first) != fixedKFs.end())
                {
                    const auto& kfPoseInv = fixedKFs[kf->first];

                    gtsam::Pose3 kfInitialPose(
                        gtsam::Rot3(kfPoseInv.block<3, 3>(0, 0)),
                        gtsam::Point3(kfPoseInv.block<3, 1>(0, 3))
                    );
                    
                    if (!initialEstimate.exists(gtsam::Symbol('x', kf->first->numb))) 
                        initialEstimate.insert(gtsam::Symbol('x', kf->first->numb), kfInitialPose);
                    graph.add(gtsam::NonlinearEquality<gtsam::Pose3>(gtsam::Symbol('x', kf->first->numb),kfInitialPose));
                }
            }

        }
        if ( mpIsOut )
            mpOutliers[mpCount] = true;
    }
    
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 10;
    params.relativeErrorTol = 1e-5;
    params.absoluteErrorTol = 1e-5;
    // if ( first )
    //     params.maxIterations = 5;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    try
    {
        result = optimizer.optimize();
    }
    catch(const gtsam::IndeterminantLinearSystemException& e)
    {
        std::cout << "Bundle Adjustment Failed... : IndeterminantLinearSystemException" << std::endl;
        map->keyFrameAdded = false;
        map->LBADone = true;
        return;
    }
    

    std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
    wrongMatches.swap(emptyVec);

    mpCount = 0;
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
    for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++, mpCount++)
    {
        MapPoint* mp = allmp->first;
        size_t kfCount {0};
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
        for (kf = mp->kFMatches.begin(); kf != endkf; kf++, kfCount++)
        {
            KeyFrame* kfCand = kf->first;
            std::pair<int,int>& keyPos = kf->second;
            if ( localKFs.find(kfCand) == localKFs.end() )
                continue;
            cv::KeyPoint kp;
            bool right {false};
            bool close {false};
            if ( keyPos.first >= 0 )
            {
                kp = kfCand->keys.keyPoints[keyPos.first];
                close = kfCand->keys.close[keyPos.first];
            }
            else if ( keyPos.second >= 0 )
            {
                kp = kfCand->keys.rightKeyPoints[keyPos.second];
                right = true;
            }

            // Extract optimized pose
            gtsam::Pose3 kfOptimizedPose;
            gtsam::Point3 mpOptimizedPos;
            try
            {
                kfOptimizedPose = result.at<gtsam::Pose3>(gtsam::Symbol('x', kfCand->numb));
                mpOptimizedPos = result.at<gtsam::Point3>(gtsam::Symbol('l', mp->idx));
            }
            catch(const gtsam::ValuesKeyDoesNotExist& e)
            {
                continue;
            }
            
            Eigen::Matrix4d optimizedPoseInv = Eigen::Matrix4d::Identity();
            optimizedPoseInv.block<3, 3>(0, 0) = kfOptimizedPose.rotation().matrix();
            optimizedPoseInv.block<3, 1>(0, 3) = kfOptimizedPose.translation();

            Eigen::Matrix4d optimizedPose = optimizedPoseInv.inverse();

            allmp->second = mpOptimizedPos;
            
            Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
            const int oct = kp.octave;
            const double weight = (double)kfCand->sigmaFactor[oct];
            Eigen::Vector3d tcw = optimizedPose.block<3,1>(0,3);

            Eigen::Matrix3d q_xyzw = optimizedPose.block<3, 3>(0, 0);
            
            Eigen::Quaterniond qcw(q_xyzw);
            // const auto qcw = kfOptimizedPose.rotation().toQuaternion();
            // // Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
            bool outlier {false};
            if ( right )
                outlier = checkOutlierR(K_eigen,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
            else
                outlier = checkOutlier(K_eigen, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

            if ( outlier )
                wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
            else
            {
                if ( close )
                {
                    if ( keyPos.second < 0 )
                        continue;
                    cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
                    const int octR = kpR.octave;
                    const double weightR = (double)kfCand->sigmaFactor[octR];
                    Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
                    bool outlierR = checkOutlierR(K_eigen,qc1c2, tc1c2, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
                    if ( outlierR )
                    {
                        wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
                    }
                }
            }

        }
    }
    // first = false;
    // }
    std::lock_guard<std::mutex> lock(map->mapMutex);

    if ( !wrongMatches.empty() )
    {
        for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
        {
            KeyFrame* kF = wrongMatches[wM].first;
            MapPoint* mp = wrongMatches[wM].second;
            const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
            kF->eraseMPConnection(keyPos);
            mp->eraseKFConnection(kF);
        }
    }


    size_t kfCount {0};
    std::unordered_map<KeyFrame*, Eigen::Matrix4d>::iterator localkf, endlocalkf(localKFs.end());
    for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++, kfCount++)
    {
        gtsam::Pose3 kfOptimizedPose;
        try
        {
            kfOptimizedPose = result.at<gtsam::Pose3>(gtsam::Symbol('x', localkf->first->numb));
        }
        catch(const gtsam::ValuesKeyDoesNotExist& e)
        {
            continue;
        }

        Eigen::Matrix4d localKFInvPose = Eigen::Matrix4d::Identity();
        localKFInvPose.block<3, 3>(0, 0) = kfOptimizedPose.rotation().matrix();
        localKFInvPose.block<3, 1>(0, 3) = kfOptimizedPose.translation();
        
        localkf->first->pose.setPose(localKFInvPose);
        localkf->first->LBA = true;
    }
    {
    int mpCount {0};
    std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
    for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
    {
        if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
            itmp->first->SetIsOutlier(true);
        else
        {
            gtsam::Point3 mpOptimizedPos;
            try
            {
                mpOptimizedPos = result.at<gtsam::Point3>(gtsam::Symbol('l', itmp->first->idx));
            }
            catch(const gtsam::ValuesKeyDoesNotExist& e)
            {
                continue;
            }

            itmp->second = mpOptimizedPos;
            itmp->first->updatePos(mpOptimizedPos, zedPtr);
        }
    }
    }

    std::cout << "Bundle Adjustment Completed..." << std::endl;

    map->endLBAIdx = actKeyF.front()->numb;
    map->keyFrameAdded = false;
    map->LBADone = true;
    
}

// void LocalMapper::loopClosureR(std::vector<KeyFrame *>& actKeyF)
// {
//     std::cout << "Loop Closure Detected! Starting Optimization.." << std::endl;
//     std::unordered_map<MapPoint*, Eigen::Vector3d> allMapPoints;
//     std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>> localKFs;
//     localKFs.reserve(actKeyF.size());
//     const unsigned long lastActKF {actKeyF.front()->numb};
//     KeyFrame* lCCand = actKeyF.front();
//     localKFs[lCCand] = Converter::Matrix4dToMatrix_7_1(map->LCPose.inverse());
//     lCCand->fixed = true;
//     std::vector<KeyFrame*>::iterator it, end(actKeyF.end());
//     for ( it = actKeyF.begin(); it != end; it++)
//     {
//         (*it)->LCID = lastActKF;
//         if ( (*it)->numb == lastActKF )
//             continue;
//         localKFs[*it] = Converter::Matrix4dToMatrix_7_1((*it)->pose.getInvPose());
        
//     }
//     for ( it = actKeyF.begin(); it != end; it++)
//     {
//         std::vector<MapPoint*>::iterator itmp, endmp((*it)->localMapPoints.end());
//         for ( itmp = (*it)->localMapPoints.begin(); itmp != endmp; itmp++)
//         {
//             MapPoint* mp = *itmp;
//             if ( !mp )
//                 continue;
//             if ( mp->GetIsOutlier() )
//                 continue;
//             if ( mp->LCID == (int)lastActKF )
//                 continue;
            
//             allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
//             (*itmp)->LCID = lastActKF;
//         }
//         std::vector<MapPoint*>::iterator endmpR((*it)->localMapPointsR.end());
//         for ( itmp = (*it)->localMapPointsR.begin(); itmp != endmpR; itmp++)
//         {
//             MapPoint* mp = *itmp;
//             if ( !mp )
//                 continue;
//             if ( mp->GetIsOutlier() )
//                 continue;
//             if ( mp->LCID == (int)lastActKF )
//                 continue;

//             allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
//             (*itmp)->LCID = lastActKF;
//         }
//     }

//     std::vector<std::pair<KeyFrame*, MapPoint*>> wrongMatches;
//     wrongMatches.reserve(allMapPoints.size());
//     std::vector<bool>mpOutliers;
//     mpOutliers.resize(allMapPoints.size());
//     bool first = true;
//     const Eigen::Matrix3d& K = zedPtr->mCameraLeft->intrinsics;
//     const Eigen::Matrix4d estimPoseRInv = zedPtr->extrinsics.inverse();
//     const Eigen::Matrix3d qc1c2 = estimPoseRInv.block<3,3>(0,0);
//     const Eigen::Matrix<double,3,1> tc1c2 = estimPoseRInv.block<3,1>(0,3);
//     for (size_t iterations{0}; iterations < 2; iterations++)
//     {
//     ceres::Problem problem;
//     ceres::Manifold* quaternion_local_parameterization = new ceres::EigenQuaternionManifold;
//     ceres::LossFunction* loss_function = nullptr;
//     if (first)
//         loss_function = new ceres::HuberLoss(sqrt(7.815f));
//     ceres::ParameterBlockOrdering* ordering = nullptr;
//     ordering = new ceres::ParameterBlockOrdering;
//     int mpCount {0};
//     std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
//     for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
//     {
//         int timesIn {0};
//         bool mpIsOut {true};
//         std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(itmp->first->kFMatches.end());
//         for (kf = itmp->first->kFMatches.begin(); kf != endkf; kf++)
//         {
//             if ( !kf->first->keyF )
//                     continue;
//             if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
//                 break;
//             if ( !wrongMatches.empty() && std::find(wrongMatches.begin(), wrongMatches.end(), std::make_pair(kf->first, itmp->first)) != wrongMatches.end())
//             {
//                 continue;
//             }
//             if ( itmp->first->GetIsOutlier() )
//                 break;
//             KeyFrame* kftemp = kf->first;
//             TrackedKeys& keys = kftemp->keys;
//             std::pair<int,int>& keyPos = kf->second;


//             if ( kf->first->numb > lastActKF )
//             {
//                 mpIsOut = false;
//                 continue;
//             }
//             timesIn ++;
//             mpIsOut = false;
//             ceres::CostFunction* costf;
//             bool close {false};
//             if ( keyPos.first >= 0 )
//             {
//                 const cv::KeyPoint& obs = keys.keyPoints[keyPos.first];
//                 Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
//                 const int oct {obs.octave};
//                 const double weight = (double)kftemp->InvSigmaFactor[oct];
//                 costf = LocalBundleAdjustment::Create(K, obs2d, weight);
//                 close = keys.close[keyPos.first];
//             }
//             else if ( keyPos.second >= 0 )
//             {
//                 const cv::KeyPoint& obs = keys.rightKeyPoints[keyPos.second];
//                 Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
//                 const int oct {obs.octave};
//                 const double weight = (double)kftemp->InvSigmaFactor[oct];
//                 costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);
//             }

//             ordering->AddElementToGroup(itmp->second.data(), 0);
//             if (localKFs.find(kf->first) != localKFs.end())
//             {
//                 ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
//                 ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
//                 problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
//                 problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
//                 if ( kf->first->fixed )
//                 {
//                     problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
//                     problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
//                 }
//             }
//             else
//                 continue;
//             if ( close )
//             {
//                 if ( keyPos.second < 0 )
//                     continue;
//                 const cv::KeyPoint& obs = kf->first->keys.rightKeyPoints[keyPos.second];
//                 Eigen::Vector2d obs2d((double)obs.pt.x, (double)obs.pt.y);
//                 const int oct {obs.octave};
//                 const double weight = (double)kftemp->InvSigmaFactor[oct];
//                 costf = LocalBundleAdjustmentR::Create(K,tc1c2, qc1c2, obs2d, weight);

//                 ordering->AddElementToGroup(itmp->second.data(), 0);
//                 if (localKFs.find(kf->first) != localKFs.end())
//                 {
//                     ordering->AddElementToGroup(localKFs[kf->first].block<3,1>(0,0).data(),1);
//                     ordering->AddElementToGroup(localKFs[kf->first].block<4,1>(3,0).data(),1);
//                     problem.AddResidualBlock(costf, loss_function, itmp->second.data(), localKFs[kf->first].block<3,1>(0,0).data(), localKFs[kf->first].block<4,1>(3,0).data());
//                     problem.SetManifold(localKFs[kf->first].block<4,1>(3,0).data(),quaternion_local_parameterization);
//                     if ( kf->first->fixed )
//                     {
//                         problem.SetParameterBlockConstant(localKFs[kf->first].block<3,1>(0,0).data());
//                         problem.SetParameterBlockConstant(localKFs[kf->first].block<4,1>(3,0).data());
//                     }
//                 }
//                 else
//                     continue;
//             }

//         }
//         if ( mpIsOut )
//             mpOutliers[mpCount] = true;
//     }
    
//     ceres::Solver::Options options;
//     options.linear_solver_ordering.reset(ordering);
//     options.num_threads = 8;
//     options.max_num_iterations = 45;
//     if ( first )
//         options.max_num_iterations = 5;
//     options.linear_solver_type = ceres::SPARSE_SCHUR;

//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::vector<std::pair<KeyFrame*, MapPoint*>> emptyVec;
//     wrongMatches.swap(emptyVec);
//     std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator allmp, allmpend(allMapPoints.end());
//     for (allmp = allMapPoints.begin(); allmp != allmpend; allmp ++)
//     {
//         MapPoint* mp = allmp->first;
//         std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf, endkf(mp->kFMatches.end());
//         for (kf = mp->kFMatches.begin(); kf != endkf; kf++)
//         {
//             KeyFrame* kfCand = kf->first;
//             std::pair<int,int>& keyPos = kf->second;
//             if ( localKFs.find(kfCand) == localKFs.end() )
//                 continue;
//             cv::KeyPoint kp;
//             bool right {false};
//             bool close {false};
//             if ( keyPos.first >= 0 )
//             {
//                 kp = kfCand->keys.keyPoints[keyPos.first];
//                 close = kfCand->keys.close[keyPos.first];
//             }
//             else if ( keyPos.second >= 0 )
//             {
//                 kp = kfCand->keys.rightKeyPoints[keyPos.second];
//                 right = true;
//             }
//             Eigen::Vector2d obs( (double)kp.pt.x, (double)kp.pt.y);
//             const int oct = kp.octave;
//             const double weight = (double)kfCand->sigmaFactor[oct];
//             Eigen::Vector3d tcw = localKFs[kfCand].block<3, 1>(0, 0);
//             Eigen::Vector4d q_xyzw = localKFs[kfCand].block<4, 1>(3, 0);
//             Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
//             bool outlier {false};
//             if ( right )
//                 outlier = checkOutlierR(K,qc1c2, tc1c2, obs, allmp->second, tcw, qcw, reprjThreshold * weight);
//             else
//                 outlier = checkOutlier(K, obs, allmp->second, tcw, qcw, reprjThreshold * weight);

//             if ( outlier )
//                 wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
//             else
//             {
//                 if ( close )
//                 {
//                     if ( keyPos.second < 0 )
//                         continue;
//                     cv::KeyPoint kpR = kfCand->keys.rightKeyPoints[keyPos.second];
//                     const int octR = kpR.octave;
//                     const double weightR = (double)kfCand->sigmaFactor[octR];
//                     Eigen::Vector2d obsr( (double)kpR.pt.x, (double)kpR.pt.y);
//                     bool outlierR = checkOutlierR(K,qc1c2, tc1c2, obsr, allmp->second, tcw, qcw, reprjThreshold * weightR);
//                     if ( outlierR )
//                     {
//                         wrongMatches.emplace_back(std::pair<KeyFrame*, MapPoint*>(kfCand, mp));
//                     }
//                 }
//             }

//         }
//     }
//     first = false;
//     }
//     std::lock_guard<std::mutex> lock(map->mapMutex);

//     if ( !wrongMatches.empty() )
//     {
//         for (size_t wM {0}, endwM {wrongMatches.size()}; wM < endwM; wM ++)
//         {
//             KeyFrame* kF = wrongMatches[wM].first;
//             MapPoint* mp = wrongMatches[wM].second;
//             const std::pair<int,int>& keyPos = mp->kFMatches.at(kF);
//             kF->eraseMPConnection(keyPos);
//             mp->eraseKFConnection(kF);
//         }
//     }


    
//     std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>::iterator localkf, endlocalkf(localKFs.end());
//     for ( localkf = localKFs.begin(); localkf != endlocalkf; localkf++)
//     {
//         localkf->first->pose.setInvPose(Converter::Matrix_7_1_ToMatrix4d(localkf->second));
//     }

//     int mpCount {0};
//     std::unordered_map<MapPoint*, Eigen::Vector3d>::iterator itmp, mpend(allMapPoints.end());
//     for ( itmp = allMapPoints.begin(); itmp != mpend; itmp++, mpCount ++)
//     {
//         if ( mpOutliers[mpCount] || (!itmp->first->GetInFrame() && (int)itmp->first->kFMatches.size() < minCount) )
//             itmp->first->SetIsOutlier(true);
//         else
//         {
//             itmp->first->updatePos(itmp->second, zedPtr);
//         }
//     }

    

//     map->endLCIdx = actKeyF.front()->numb;
//     map->LCDone = true;
//     map->LCStart = false;
//     map->aprilTagDetected = false;
//     std::cout << "Loop Closure Optimization Finished!" << std::endl;
// }

void LocalMapper::insertMPsForLBA(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs,std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& fixedKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LBAID == (int)lastActKF )
            continue;
        
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator kf = (back) ? mp->kFMatchesB.begin() : mp->kFMatches.begin();
        std::unordered_map<KeyFrame*, std::pair<int,int>>::iterator endkf = (back) ? mp->kFMatchesB.end() : mp->kFMatches.end();
        for (; kf != endkf; kf++)
        {
            KeyFrame* kFCand = kf->first;
            if ( !kFCand->keyF || kFCand->numb > lastActKF )
                continue;
            if (kFCand->LBAID == (int)lastActKF )
                continue;
            if (localKFs.find(kFCand) == localKFs.end())
            {
                fixedKFs[kFCand] = Converter::Matrix4dToMatrix_7_1(kFCand->pose.getInvPose());
                kFCand->LBAID = lastActKF;
            }
            blocks++;
        }
        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LBAID = lastActKF;
    }
}

void LocalMapper::insertMPsForLC(std::vector<MapPoint*>& localMapPoints, const std::unordered_map<KeyFrame*, Eigen::Matrix<double,7,1>>& localKFs, std::unordered_map<MapPoint*, Eigen::Vector3d>& allMapPoints, const unsigned long lastActKF, int& blocks, const bool back)
{
    std::vector<MapPoint*>::iterator itmp, endmp(localMapPoints.end());
    for ( itmp = localMapPoints.begin(); itmp != endmp; itmp++)
    {
        MapPoint* mp = *itmp;
        if ( !mp )
            continue;
        if ( mp->GetIsOutlier() )
            continue;
        if ( mp->LCID == (int)lastActKF )
            continue;

        allMapPoints.insert(std::pair<MapPoint*, Eigen::Vector3d>((*itmp), (*itmp)->getWordPose3d()));
        (*itmp)->LCID = lastActKF;
    }
}

void LocalMapper::beginLocalMapping()
{
    using namespace std::literals::chrono_literals;
    while ( !map->endOfFrames )
    {
        if ( map->keyFrameAdded && !map->LBADone && !map->LCStart )
        {

            std::vector<KeyFrame *> actKeyF;
            KeyFrame* lastKF = map->keyFrames.at(map->kIdx - 1);
            actKeyF.reserve(20);
            actKeyF.emplace_back(lastKF);
            lastKF->getConnectedKFs(actKeyF, actvKFMaxSize);
            
            {
            // triangulateNewPointsR(actKeyF);
            }
            {
            // localBAR(actKeyF);
            }

        }
        if ( stopRequested )
            break;
        std::this_thread::sleep_for(20ms);
    }
    std::cout << "LocalMap Thread Exited!" << std::endl;
}

// void LocalMapper::beginLoopClosure()
// {
//     using namespace std::literals::chrono_literals;
//     while ( true )
//     {
//         if ( map->LCStart )
//         {
//             std::vector<KeyFrame *> activeKF;
//             activeKF.reserve(map->LCCandIdx);
//             KeyFrame* kFLCCand = map->keyFrames.at(map->LCCandIdx);
//             activeKF.emplace_back(kFLCCand);
//             kFLCCand->getConnectedKFsLC(map, activeKF);

//             loopClosureR(activeKF);

//         }
//         if ( stopRequested )
//             break;
//         std::this_thread::sleep_for(100ms);
//     }
//     std::cout << "LoopClosure Thread Exited!" << std::endl;
// }

} // namespace TII