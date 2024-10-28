# gtsam-vSLAM

Stereo Inertial, Stereo or Monocular Inertial visual SLAM using GTSAM library for optimization and Iridescence for visualization.


To build the project run :
```
./build_project.sh
```

From the config files the mode can be changed :

```
# 0 Stereo IMU
# 1 Stereo 
# 2 Monocular IMU
slamMode: 1
```

To run the project : 

```
./VIOSlam config.yaml
```