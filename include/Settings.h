#ifndef SETTINGS_H
#define SETTINGS_H

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <chrono>

namespace TII
{

class ConfigFile
{
    private:
        const char* configPath;

    public:
        bool badFile = false;
        YAML::Node configNode;
        ConfigFile(const char* config = "config.yaml");
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

} // namespace TII


#endif // SETTINGS_H