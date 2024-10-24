echo "Downloading all apt-get packages needed ..."

# sudo apt-get install cmake

# sudo apt-get install libgoogle-glog-dev libgflags-dev

# sudo apt-get install libatlas-base-dev

# sudo apt-get install libeigen3-dev

# sudo apt-get install libsuitesparse-dev

# sudo apt-get install libyaml-cpp-dev

# sudo apt install libopencv-dev python3-opencv

# sudo apt-get install libboost-all-dev

sudo apt-get install -y libglm-dev libglfw3-dev libpng-dev libjpeg-dev libboost-all-dev libopencv-dev python3-opencv libyaml-cpp-dev libsuitesparse-dev libeigen3-dev libatlas-base-dev cmake libgoogle-glog-dev libgflags-dev

echo "Downloading and Building GTSAM v4.2 ..."

mkdir packages
cd packages

git clone --recursive --branch 4.2 --depth 1 https://github.com/borglab/gtsam.git

#!bash
cd gtsam
mkdir build
cd build
cmake ..
make check -j4
make install


cd ../../../

mkdir build
cd build
cmake ..
make -j4
