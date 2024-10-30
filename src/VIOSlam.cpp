#define EIGEN_MALLOC_ALREADY_ALIGNED 0
#include "System.h"
#include "Settings.h"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <signal.h>
#include <filesystem>

namespace fs = std::filesystem;

volatile sig_atomic_t flag = 0;

void signal_callback_handler(int signum) {
    flag = 1;
}

bool getAllIMUData(std::vector<Eigen::Vector3d>& angleVelocityVec, std::vector<Eigen::Vector3d>& accelerationVec, std::vector<double>& timestampsVec, const std::string& IMUFilePath)
{
    using namespace std;

    ifstream file(IMUFilePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return false;
    }

    string line;
    // Skip the header line
    getline(file, line);

    // Loop through each line of the CSV file
    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        // Variables to store parsed data
        double timestamp;
        Eigen::Vector3d angular_velocity{};
        Eigen::Vector3d acceleration{};

        // Parse the timestamp
        getline(ss, token, ',');
        timestamp = stod(token);

        // Parse angular velocity (w_RS_S_x, w_RS_S_y, w_RS_S_z)
        for (int i = 0; i < 3; ++i) {
            getline(ss, token, ',');
            angular_velocity(i) = stod(token);
        }

        // Parse acceleration (a_RS_S_x, a_RS_S_y, a_RS_S_z)
        for (int i = 0; i < 3; ++i) {
            getline(ss, token, ',');
            acceleration(i) = stod(token);
        }

        // Add the data to the vectors
        timestampsVec.push_back(timestamp);
        angleVelocityVec.push_back(angular_velocity);
        accelerationVec.push_back(acceleration);
    }

    // Close the file
    file.close();

    return true;
}

bool getImageTimestamps(std::vector<std::string>& fileNameVec, std::vector<double>& timestampsVec, const std::string& ImagesFilePath)
{
    using namespace std;

    ifstream file(ImagesFilePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return false;
    }

    string line;
    // Skip the header line
    getline(file, line);

    // Loop through each line of the CSV file
    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        // Variables to store parsed data
        double timestamp;

        // Parse the timestamp
        getline(ss, token, ',');
        timestamp = stod(token);

        getline(ss, token, ',');
        std::string fileName = token;
        if (fileName.back() == '\r')
            fileName.erase(fileName.size() - 1);
        // Add the data to the vectors
        timestampsVec.push_back(timestamp);
        fileNameVec.push_back(fileName);
    }

    // Close the file
    file.close();

    return true;
}

void getImageNames(const fs::path& directoryPath, std::vector<std::string>& imageNames) 
{
    int fileCount = 0;
    if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) 
    {
        for (const auto& entry : fs::directory_iterator(directoryPath)) 
        {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") 
            {
                ++fileCount;
            }
        }
    } else 
    {
        std::cerr << "Directory does not exist or is not a directory." << std::endl;
    }

    // Add filenames manually in the required format
    for (int i = 0; i < fileCount; ++i) {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << i << ".png"; // Format as 000000.png, 000001.png, etc.
        imageNames.push_back(oss.str());
    }
}

int main(int argc, char **argv)
{
    if ( argc < 2 )
    {
        std::cerr << "No config file given.. Usage : ./VIOSlam config_file_name (e.g. ./VIOSlam config.yaml)" << std::endl;

        return -1;
    }
    std::string file = argv[1];
    auto confFile = std::make_shared<GTSAM_VIOSLAM::ConfigFile>(file.c_str());

    const auto slamMode = static_cast<GTSAM_VIOSLAM::VSlamSystem::SlamMode>(confFile->getValue<int>("slamMode"));

    std::cout << " slam : " << static_cast<int>(slamMode) << std::endl;
    auto slamSystem = std::make_shared<GTSAM_VIOSLAM::VSlamSystem>(confFile, slamMode);

    bool hasIMU {false};
    switch (slamMode)
    {
    case GTSAM_VIOSLAM::VSlamSystem::SlamMode::STEREO_IMU:
        std::cout << "Stereo IMU Mode Selected.." << std::endl;
        hasIMU = true;
        break;
    case GTSAM_VIOSLAM::VSlamSystem::SlamMode::MONOCULAR:
        std::cout << "Monocular IMU Mode Selected.." << std::endl;
        hasIMU = true;
        break;
    default:
        std::cout << "Stereo Mode Selected.." << std::endl;
        break;
    }

    std::shared_ptr<GTSAM_VIOSLAM::StereoCamera> StereoCam{nullptr};

    slamSystem->GetStereoCamera(StereoCam);

    const std::string imagesPath = confFile->getValue<std::string>("imagesPath");


    const std::string dataset = confFile->getValue<std::string>("dataset");

    std::vector<std::string> ImageFileNamesVec;
    std::vector<double> imageTimestamps;
    std::string leftPath;
    std::string rightPath;

    if (dataset == "KITTI")
    {
        leftPath = imagesPath + "image_0/";
        rightPath = imagesPath + "image_1/";
        const std::string ImageFileNamePath = imagesPath + "image_0/";
        getImageNames(ImageFileNamePath, ImageFileNamesVec);
    }
    else if (dataset == "EuRoC")
    {
        leftPath = imagesPath + "cam0/data/";
        rightPath = imagesPath + "cam1/data/";
        const std::string ImageFileNamePath = imagesPath + "cam0/";
        getImageTimestamps(ImageFileNamesVec, imageTimestamps, ImageFileNamePath + "data.csv");
    }
    else
    {
        std::cout << "Only KITTI and EuRoC Dataset Supported.." << std::endl;
        return -1;
    }

    const size_t numberFrames {ImageFileNamesVec.size()};
    std::vector<std::string>leftImagesStr, rightImagesStr;
    leftImagesStr.reserve(numberFrames);
    rightImagesStr.reserve(numberFrames);
    for ( const auto& imageName : ImageFileNamesVec)
    {
        leftImagesStr.emplace_back(leftPath + imageName);
        rightImagesStr.emplace_back(rightPath + imageName);
    }

    StereoCam->numOfFrames = numberFrames;

    bool IMUDataValid {false};
    std::vector<GTSAM_VIOSLAM::IMUData>IMUDataPerFrame;

    if (hasIMU)
    {
        const std::string IMUDataPath = confFile->getValue<std::string>("IMU","Path");
        const int IMUHz = confFile->getValue<int>("IMU","Hz");
        const double IMUGyroNoiseDensity = confFile->getValue<double>("IMU","gyroscope_noise_density");
        const double IMUGyroRandomWalk = confFile->getValue<double>("IMU","gyroscope_random_walk");
        const double IMUAccelNoiseDensity = confFile->getValue<double>("IMU","accelerometer_noise_density");
        const double IMUAccelRandomWalk = confFile->getValue<double>("IMU","accelerometer_random_walk");

        StereoCam->mCameraLeft->mIMUData = std::make_shared<GTSAM_VIOSLAM::IMUData>(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz);

        GTSAM_VIOSLAM::IMUData allIMUData(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz);

        IMUDataValid = getAllIMUData(allIMUData.mAngleVelocity, allIMUData.mAcceleration, allIMUData.mTimestamps, IMUDataPath + "data.csv");

        IMUDataPerFrame = std::vector<GTSAM_VIOSLAM::IMUData>(numberFrames,{IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz});
        if (IMUDataValid)
        {
            const size_t IMUDataSize {allIMUData.mAngleVelocity.size()};
            const size_t IMUDataPerFrameSize {static_cast<size_t>(std::round(IMUHz / StereoCam->mFps)) + 1};
            int frameNumb {0};
            double frameTimestamp {imageTimestamps[frameNumb]};
            double nextFrameTimestamp {imageTimestamps[frameNumb + 1]};
            IMUDataPerFrame[frameNumb].mAngleVelocity.reserve(IMUDataPerFrameSize);
            IMUDataPerFrame[frameNumb].mAcceleration.reserve(IMUDataPerFrameSize);
            IMUDataPerFrame[frameNumb].mTimestamps.reserve(IMUDataPerFrameSize);
            for (size_t i = 0; i < IMUDataSize; ++i)
            {
                const auto& IMUTimestamp = allIMUData.mTimestamps[i];
                if (IMUTimestamp > frameTimestamp && IMUTimestamp > nextFrameTimestamp)
                {
                    frameNumb ++;
                    frameTimestamp = imageTimestamps[frameNumb];
                    nextFrameTimestamp = imageTimestamps[frameNumb + 1];
                    IMUDataPerFrame[frameNumb].mAngleVelocity.reserve(IMUDataPerFrameSize);
                    IMUDataPerFrame[frameNumb].mAcceleration.reserve(IMUDataPerFrameSize);
                    IMUDataPerFrame[frameNumb].mTimestamps.reserve(IMUDataPerFrameSize);
                }
                auto& IMUFrame = IMUDataPerFrame[frameNumb];
                if (IMUTimestamp > frameTimestamp && IMUTimestamp < nextFrameTimestamp)
                {
                    const auto& angleVel = allIMUData.mAngleVelocity[i];
                    const auto& accel = allIMUData.mAcceleration[i];

                    IMUFrame.mAngleVelocity.emplace_back(angleVel);
                    IMUFrame.mAcceleration.emplace_back(accel);
                    IMUFrame.mTimestamps.emplace_back(IMUTimestamp);
                }
            }

        }

        StereoCam->mCameraLeft->mIMUGravity = {IMUDataPerFrame[0].mAcceleration[0](1), -IMUDataPerFrame[0].mAcceleration[0](0),IMUDataPerFrame[0].mAcceleration[0](2)};

    }

    cv::Mat rectMap[2][2];
    const int width = StereoCam->mWidth;
    const int height = StereoCam->mHeight;

    if ( !StereoCam->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(StereoCam->mCameraLeft->K, StereoCam->mCameraLeft->D, StereoCam->mCameraLeft->R, StereoCam->mCameraLeft->P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
        cv::initUndistortRectifyMap(StereoCam->mCameraRight->K, StereoCam->mCameraRight->D, StereoCam->mCameraRight->R, StereoCam->mCameraRight->P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[1][0], rectMap[1][1]);
    }

    for ( size_t frameNumb{0}; frameNumb < numberFrames; frameNumb++)
    {

        cv::Mat imageLeft = cv::imread(leftImagesStr[frameNumb],cv::IMREAD_COLOR);
        cv::Mat imageRight = cv::imread(rightImagesStr[frameNumb],cv::IMREAD_COLOR);

        cv::Mat imLRect, imRRect;

        if ( !StereoCam->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
            cv::remap(imageRight, imRRect, rectMap[1][0], rectMap[1][1], cv::INTER_LINEAR);
        }
        else
        {
            imLRect = imageLeft.clone();
            imRRect = imageRight.clone();
        }

        if (IMUDataValid && slamMode == GTSAM_VIOSLAM::VSlamSystem::SlamMode::STEREO_IMU)
            slamSystem->TrackStereoIMU(imLRect, imRRect, frameNumb, IMUDataPerFrame[frameNumb]);
        else
            slamSystem->TrackStereo(imLRect, imRRect, frameNumb);

        if ( flag == 1 )
            break;

    }
    std::cout << "System Shutdown!" << std::endl;
    slamSystem->ExitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    slamSystem->saveTrajectoryAndPosition("single_cam_im_tra.txt", "single_cam_im_pos.txt");
    std::cout << "Trajectory Saved!" << std::endl;
    while ( flag != 1 )
    {
        usleep(1e6);
    }
    exit(SIGINT);


}
