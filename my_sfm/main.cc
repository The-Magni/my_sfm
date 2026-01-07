#include "config/config.h"
#include "data_association.h"
#include "keypoints.h"
#include "descriptors.h"
#include "outlier_filtering.h"
#include "reconstruction.h"
#include "two_view_geometries.h"
#include "visualization/visualization.h"
#include <cassert>
#include <cstring>
#include <glog/logging.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <string>

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    if (argc != 3) {
        LOG(ERROR) << "Must provide the specific step in sfm and the config file path";
        google::ShutdownGoogleLogging();
        return 1;
    }
    YAMLConfig yaml_config;
    yaml_config.Init(argv[2]);
    std::string database_path = yaml_config.Get<std::string>("database_path");
    std::string data_path = yaml_config.Get<std::string>("data_path");
    std::shared_ptr<Images> imgs(new Images(yaml_config.Get<std::string>("images_path")));
    std::shared_ptr<KeyPointsDB> keypoints_db(new KeyPointsDB(database_path));
    std::shared_ptr<DescriptorsDB> descriptors_db(new DescriptorsDB(database_path));
    std::shared_ptr<TwoViewGeometriesDB> two_view_db(new TwoViewGeometriesDB(database_path));

    double max_reproj_error = 4.0, min_tri_angle = 1.5;
    if (yaml_config.config()["outlier_filtering"]) {
        max_reproj_error = yaml_config.config()["outlier_filtering"]["max_reproj_error"].as<double>();
        min_tri_angle = yaml_config.config()["outlier_filtering"]["min_tri_angle"].as<double>();
    }
    std::shared_ptr<OutlierFiltering> outlier_filtering(new OutlierFiltering(keypoints_db, max_reproj_error, min_tri_angle));
    DataAssociation da;
    Reconstruction recon;
    Visualization viz;

    da.Init(imgs, keypoints_db, descriptors_db, two_view_db);
    viz.Init(data_path);


    if (strcmp(argv[1], "data_association") == 0)
        da.Process();
    else if(strcmp(argv[1], "reconstruction") == 0) {
        recon.Init(two_view_db, imgs, keypoints_db, outlier_filtering);
        recon.IncrementalReconstruction();
        recon.Write(data_path);
    } else if (strcmp(argv[1], "viz") == 0)
        viz.Process();
    else if (strcmp(argv[1], "full_reconstruction") == 0) {
        da.Process();
        recon.Init(two_view_db, imgs, keypoints_db, outlier_filtering);
        recon.IncrementalReconstruction();
        recon.Write(data_path);
        viz.Process();
    } else {
        LOG(ERROR) << "Options not available" << '\n';
    }

    google::ShutdownGoogleLogging();
    return 0;
}
