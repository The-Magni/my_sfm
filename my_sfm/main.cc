#include "data_association.h"
#include "keypoints.h"
#include "descriptors.h"
#include "reconstruction.h"
#include "two_view_geometries.h"
#include "visualization/visualization.h"
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
    // DataAssociation da;
    // da.Init(imgs, db, db1, db2);
    // da.Process();
    // Reconstruction recon;
    // recon.Init(db2, imgs, db);
    // recon.IncrementalReconstruction();
    // recon.Write("/home/dinh/my_sfm/data");

    const std::string dir_path = "/home/dinh/my_sfm/data";
    Visualization viz(dir_path);
    viz.Process();

    google::ShutdownGoogleLogging();
    return 0;
}
