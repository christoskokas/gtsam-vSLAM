echo "Downloading all apt-get packages needed ..."

sudo apt-get install -y libglm-dev libglfw3-dev libpng-dev libjpeg-dev libboost-all-dev libopencv-dev python3-opencv libyaml-cpp-dev libsuitesparse-dev libeigen3-dev libatlas-base-dev cmake libgoogle-glog-dev libgflags-dev

curl -s --compressed "https://koide3.github.io/ppa/ubuntu2004/KEY.gpg" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/koide3_ppa.gpg >/dev/null
echo "deb [signed-by=/etc/apt/trusted.gpg.d/koide3_ppa.gpg] https://koide3.github.io/ppa/ubuntu2004 ./" | sudo tee /etc/apt/sources.list.d/koide3_ppa.list

sudo apt update && sudo apt install -y libiridescence-dev

mkdir packages
cd packages

echo "Downloading and Building GTSAM ..."

git clone --branch release/4.2 https://github.com/borglab/gtsam.git


#!bash
cd gtsam
mkdir build
cd build
sudo cmake .. -DGTSAM_USE_SYSTEM_EIGEN=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DCMAKE_CXX_FLAGS:STRING=-std=c++17
sudo make install -j8


cd ../../../

mkdir build
cd build
sudo cmake ..
sudo make -j8
