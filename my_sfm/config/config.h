#pragma once

#include <glog/logging.h>
#include <memory>
#include <string>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

class YAMLConfig {
    private:
        std::unique_ptr<YAML::Node> config_{nullptr};

    public:
        void Init(const std::string &file);

        const YAML::Node &config() const {
            if (!config_) LOG(ERROR) << "Config is not initialized";
            return *config_;
        }

        template<typename T>
        const T Get(const std::string &key) const {
            if (!config_) LOG(ERROR) << "Config is not initialized";
            return (*config_)[key].as<T>();
        }
};
