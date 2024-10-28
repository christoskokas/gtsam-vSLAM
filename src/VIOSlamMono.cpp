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

int main(int argc, char **argv)
{
    if ( argc < 2 )
    {
        std::cerr << "No config file given.. Usage : ./VIOSlam config_file_name (e.g. ./VIOSlamMono config.yaml)" << std::endl;

        return -1;
    }
    std::string file = argv[1];
    auto confFile = std::make_shared<GTSAM_VIOSLAM::ConfigFile>(file.c_str());

    auto slamSystem = std::make_shared<GTSAM_VIOSLAM::VSlamSystem>(confFile,GTSAM_VIOSLAM::VSlamSystem::SlamMode::MONOCULAR);

    std::shared_ptr<GTSAM_VIOSLAM::StereoCamera> StereoCam;

    slamSystem->GetStereoCamera(StereoCam);

    const std::string imagesPath = confFile->getValue<std::string>("imagesPath");

    const std::string leftPath = imagesPath + "cam0/data/";
    const std::string rightPath = imagesPath + "cam1/data/";

    const std::string IMUDataPath = confFile->getValue<std::string>("IMU","Path");
    const int IMUHz = confFile->getValue<int>("IMU","Hz");
    const double IMUGyroNoiseDensity = confFile->getValue<double>("IMU","gyroscope_noise_density");
    const double IMUGyroRandomWalk = confFile->getValue<double>("IMU","gyroscope_random_walk");
    const double IMUAccelNoiseDensity = confFile->getValue<double>("IMU","accelerometer_noise_density");
    const double IMUAccelRandomWalk = confFile->getValue<double>("IMU","accelerometer_random_walk");

    StereoCam->mCameraLeft->mIMUData = std::make_shared<GTSAM_VIOSLAM::IMUData>(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz);

    GTSAM_VIOSLAM::IMUData allIMUData(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz);

    bool IMUDataValid = getAllIMUData(allIMUData.mAngleVelocity, allIMUData.mAcceleration, allIMUData.mTimestamps, IMUDataPath + "data.csv");

    const std::string ImageFileNamePath = imagesPath + "cam0/";
    std::vector<std::string> ImageFileNamesVec;
    std::vector<double> imageTimestamps;

    bool imageTimestampsValid = getImageTimestamps(ImageFileNamesVec, imageTimestamps, ImageFileNamePath + "data.csv");

    const size_t numberFrames {ImageFileNamesVec.size()};
    std::vector<std::string>leftImagesStr;
    leftImagesStr.reserve(numberFrames);
    for ( const auto& imageName : ImageFileNamesVec)
    {
        leftImagesStr.emplace_back(leftPath + imageName);
    }

    // std::vector < double > tBIMU = confFile->getValue<std::vector<double>>("T_bc1", "data");

    // Eigen::Matrix4d tBodyToImu;
    // tBodyToImu = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(tBIMU.data());

    // Eigen::Matrix4d tBodyToImuInv = tBodyToImu.inverse();

    // Get all the IMU Data between each new frame for pre Integration
    auto IMUDataPerFrame = std::vector<GTSAM_VIOSLAM::IMUData>(numberFrames,{IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk, IMUHz});
    if (IMUDataValid && imageTimestampsValid)
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

                // Eigen::Vector3d angleCamFrame, accelCamFrame;

                // transformIMUToCameraFrame2(angleVel, accel, tBodyToImu, angleCamFrame, accelCamFrame);
                // // transformIMUToCameraFrame(angleVel, accel, tBodyToImuInv.block<3,3>(0,0), tBodyToImuInv.block<3,1>(0,3), angleCamFrame, accelCamFrame);

                // Eigen::Matrix3d RImu = tBodyToImuInv.block<3,3>(0,0);
                // Eigen::Vector3d tImu = tBodyToImuInv.block<3,1>(0,3);
                // Eigen::Vector3d angleCamFrame = transformAngularVelocity(angleVel, RImu);
                // Eigen::Vector3d accelCamFrame = transformLinearAcceleration(accel, angleVel, RImu, tImu);

                IMUFrame.mAngleVelocity.emplace_back(angleVel);
                IMUFrame.mAcceleration.emplace_back(accel);
                IMUFrame.mTimestamps.emplace_back(IMUTimestamp);
            }
        }

    }

    StereoCam->mCameraLeft->mIMUGravity = {IMUDataPerFrame[0].mAcceleration[0](1), -IMUDataPerFrame[0].mAcceleration[0](0),IMUDataPerFrame[0].mAcceleration[0](2)};

    cv::Mat rectMap[2][2];
    const int width = StereoCam->mWidth;
    const int height = StereoCam->mHeight;

    if ( !StereoCam->rectified )
    {
        cv::Mat R1,R2;
        cv::initUndistortRectifyMap(StereoCam->mCameraLeft->K, StereoCam->mCameraLeft->D, StereoCam->mCameraLeft->R, StereoCam->mCameraLeft->P.rowRange(0,3).colRange(0,3), cv::Size(width, height), CV_32F, rectMap[0][0], rectMap[0][1]);
    }

    // double timeBetFrames = 1.0/StereoCam->mFps;

    for ( size_t frameNumb{0}; frameNumb < numberFrames; frameNumb++)
    {
        // auto start = std::chrono::high_resolution_clock::now();

        cv::Mat imageLeft = cv::imread(leftImagesStr[frameNumb],cv::IMREAD_COLOR);

        cv::Mat imLRect;

        if ( !StereoCam->rectified )
        {
            cv::remap(imageLeft, imLRect, rectMap[0][0], rectMap[0][1], cv::INTER_LINEAR);
        }
        else
        {
            imLRect = imageLeft.clone();
        }

        if (IMUDataValid)
            slamSystem->TrackMonoIMU(imLRect, frameNumb, IMUDataPerFrame[frameNumb]);
        else
        {
            std::cout << "Monocular requires IMU" << std::endl;
            return -1;
        }


        // auto end = std::chrono::high_resolution_clock::now();
        // double duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

        // if ( duration < timeBetFrames )
        //     usleep((timeBetFrames-duration)*1e6);

        if ( flag == 1 )
            break;

    }
    while ( flag != 1 )
    {
        usleep(1e6);
    }
    std::cout << "System Shutdown!" << std::endl;
    slamSystem->ExitSystem();
    exit(SIGINT);


}
