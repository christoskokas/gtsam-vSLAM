# gtsam-vSLAM

Stereo or Monocular visual SLAM using GTSAM library for optimization and Iridescence for visualization.

```
sudo apt-get install -y libglm-dev libglfw3-dev libpng-dev libjpeg-dev libeigen3-dev
```

Install Iridescence : 

```
curl -s --compressed "https://koide3.github.io/ppa/ubuntu2004/KEY.gpg" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/koide3_ppa.gpg >/dev/null
echo "deb [signed-by=/etc/apt/trusted.gpg.d/koide3_ppa.gpg] https://koide3.github.io/ppa/ubuntu2004 ./" | sudo tee /etc/apt/sources.list.d/koide3_ppa.list

sudo apt update && sudo apt install -y libiridescence-dev
```

Install GTSAM : 

```
sudo apt-get install libboost-all-dev
sudo apt-get install cmake

git clone --recursive --branch 4.2 --depth 1 https://github.com/borglab/gtsam.git
```

Inside the library folder :

```
#!bash
mkdir build
cd build
cmake ..
make check
sudo make install
```