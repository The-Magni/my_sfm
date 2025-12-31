#include "data_association.h"
#include "keypoints.h"
#include "descriptors.h"
#include "point_cloud.h"
#include "reconstruction.h"
#include "two_view_geometries.h"
#include "opencv2/viz/viz3d.hpp"
#include <glog/logging.h>
#include <memory>
#include <opencv2/core/types.hpp>

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    std::shared_ptr<Images> imgs(new Images("/home/dinh/my_sfm/data/images"));
    std::shared_ptr<KeyPointsDB> db(new KeyPointsDB("/home/dinh/my_sfm/data/database.db"));
    std::shared_ptr<DescriptorsDB> db1(new DescriptorsDB("/home/dinh/my_sfm/data/database.db"));
    std::shared_ptr<TwoViewGeometriesDB> db2(new TwoViewGeometriesDB("/home/dinh/my_sfm/data/database.db"));
    DataAssociation da;
    // da.Init(imgs, db, db1, db2);
    // da.Process();
    Reconstruction recon;
    recon.Init(db2, imgs, db);
    recon.IncrementalReconstruction();
    const PointCloud &pointcloud = recon.getPointCloud();
    cv::viz::Viz3d viz_window("Incremental Pointcloud");
    viz_window.setBackgroundColor(cv::viz::Color::white());
    viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(1.0));
    cv::Mat cloud(pointcloud.points.size(), 1, CV_64FC3);
    cv::Mat colors(pointcloud.points.size(), 1, CV_8UC3);
    cv::viz::WCloud cloud_widget(cloud, colors);
    viz_window.showWidget("Point Cloud", cloud_widget);
    viz_window.spin();

    google::ShutdownGoogleLogging();
    return 0;
}
