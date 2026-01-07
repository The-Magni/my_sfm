#include "config.h"
#include <string>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>

void YAMLConfig::Init(const std::string &file)
{
    config_.reset(new YAML::Node);
    *config_ = YAML::LoadFile(file);
}
