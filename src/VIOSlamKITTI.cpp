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
        const std::string fileName = token;
        
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
        std::cerr << "No config file given.. Usage : ./VIOSlam config_file_name (e.g. ./VIOSlam config.yaml)" << std::endl;

        return -1;
    }
    std::string file = argv[1];
    auto confFile = std::make_shared<TII::ConfigFile>(file.c_str());
    auto slamSystem = std::make_shared<TII::VSlamSystem>(confFile);

    std::shared_ptr<TII::StereoCamera> StereoCam;

    slamSystem->GetStereoCamera(StereoCam);

    const size_t nFrames {StereoCam->numOfFrames};
    std::vector<std::string>leftImagesStr, rightImagesStr;
    leftImagesStr.reserve(nFrames);
    rightImagesStr.reserve(nFrames);

    const std::string imagesPath = confFile->getValue<std::string>("imagesPath");

    const std::string leftPath = imagesPath + "left/";
    const std::string rightPath = imagesPath + "right/";
    const std::string fileExt = confFile->getValue<std::string>("fileExtension");

    const std::string IMUDataPath = confFile->getValue<std::string>("IMU","Path");
    const int IMUHz = confFile->getValue<int>("IMU","Hz");
    const double IMUGyroNoiseDensity = confFile->getValue<double>("IMU","gyroscope_noise_density");
    const double IMUGyroRandomWalk = confFile->getValue<double>("IMU","gyroscope_random_walk");
    const double IMUAccelNoiseDensity = confFile->getValue<double>("IMU","accelerometer_noise_density");
    const double IMUAccelRandomWalk = confFile->getValue<double>("IMU","accelerometer_random_walk");

    StereoCam->mCameraLeft->mIMUData = std::make_shared<TII::IMUData>(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk);

    TII::IMUData allIMUData(IMUGyroNoiseDensity, IMUGyroRandomWalk, IMUAccelNoiseDensity, IMUAccelRandomWalk);

    bool IMUDataValid = getAllIMUData(allIMUData.mAngleVelocity, allIMUData.mAcceleration, allIMUData.mTimestamps, IMUDataPath + "data.csv");

    const std::string ImageFileNamePath = confFile->getValue<std::string>("ImageFileNamePath");
    std::vector<std::string> ImageFileNamesVec;
    std::vector<double> imageTimestamps;

    bool imageTimestampsValid = getImageTimestamps(ImageFileNamesVec, imageTimestamps, ImageFileNamePath + "data.csv");

    const size_t imageNumbLength = 6;

    for ( size_t i {0}; i < nFrames; i++)
    {
        std::string frameNumb = std::to_string(i);
        std::string frameStr = std::string(imageNumbLength - std::min(imageNumbLength, frameNumb.length()), '0') + frameNumb;
        leftImagesStr.emplace_back(leftPath + frameStr + fileExt);
        rightImagesStr.emplace_back(rightPath + frameStr + fileExt);
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

    double timeBetFrames = 1.0/StereoCam->mFps;

    for ( size_t frameNumb{0}; frameNumb < nFrames; frameNumb++)
    {
        auto start = std::chrono::high_resolution_clock::now();

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

        slamSystem->TrackStereo(imLRect, imRRect, frameNumb);


        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

        if ( duration < timeBetFrames )
            usleep((timeBetFrames-duration)*1e6);

        if ( flag == 1 )
            break;

    }
    while ( flag != 1 )
    {
        usleep(1e6);
    }
    std::cout << "System Shutdown!" << std::endl;
    slamSystem->ExitSystem();
    std::cout << "Saving Trajectory.." << std::endl;
    slamSystem->SaveTrajectoryAndPosition("single_cam_im_tra.txt", "single_cam_im_pos.txt");
    std::cout << "Trajectory Saved!" << std::endl;
    exit(SIGINT);


}
