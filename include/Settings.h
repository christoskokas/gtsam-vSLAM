#ifndef SETTINGS_H
#define SETTINGS_H

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <chrono>

namespace GTSAM_VIOSLAM
{

class ConfigFile
{
    private:

    public:
        bool badFile = false;
        YAML::Node configNode;
        ConfigFile(const std::string& config = "config.yaml") 
        { 
            std::cout << "Constructor: " << this << std::endl; 
            std::string file_path = __FILE__;
            std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
            std::cout << dir_path + "/../config/" + config << std::endl;
            try
            {
                configNode = YAML::LoadFile(dir_path + "/../config/" + config);
            }
            catch(const YAML::BadFile& e)
            {
                std::cerr << "No config file named " << config << " in path " << dir_path + "/../config/" + config << std::endl;
                badFile = true;
            }
        }
        ~ConfigFile() { std::cout << "Destructor: " << this << std::endl; }
        template<typename T> 
        T getValue(const std::string& first = "", const std::string& second = "", const std::string& third = "") const
        {
            if (!third.empty())
                return configNode[first][second][third].as<T>();
            else if (!second.empty())
                return configNode[first][second].as<T>();
            else
                return configNode[first].as<T>();
        }
};

} // namespace GTSAM_VIOSLAM


#endif // SETTINGS_H