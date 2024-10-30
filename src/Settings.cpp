#include "Settings.h"

namespace GTSAM_VIOSLAM
{

ConfigFile::ConfigFile(const std::string& config /*= "config.yaml"*/)
{
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.find_last_of("/\\"));
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


} // namespace GTSAM_VIOSLAM