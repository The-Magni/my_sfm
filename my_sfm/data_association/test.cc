#include "data_association.h"
#include "keypoints.h"
#include "descriptors.h"
#include "two_view_geometries.h"
#include <memory>
#include <opencv2/core/types.hpp>

int main()
{
    std::shared_ptr<Images> imgs(new Images("/home/dinh/my_sfm/data/images"));
    std::shared_ptr<KeyPointsDB> db(new KeyPointsDB("/home/dinh/my_sfm/data/database.db"));
    std::shared_ptr<DescriptorsDB> db1(new DescriptorsDB("/home/dinh/my_sfm/data/database.db"));
    std::shared_ptr<TwoViewGeometriesDB> db2(new TwoViewGeometriesDB("/home/dinh/my_sfm/data/database.db"));
    DataAssociation da;
    da.Init(imgs, db, db1, db2);
    da.Process();
    // std::vector<cv::KeyPoint> k1, k2;
    // cv::Mat d1, d2;
    // da.featureExtraction(0, k1, d1);
    // da.featureExtraction(1, k2, d2);


    // // insert to the db
    // // db->Insert(0, k1);
    // // db->Insert(1, k2);
    // // db1->Insert(0, d1);
    // // db1->Insert(1, d2);
    // std::vector<cv::DMatch> good_matches;

    // // db1->Retrieve(0, d1);
    // // db1->Retrieve(1, d2);
    // da.featureMatching(d1, d2, good_matches);
    // cv::Mat img_matches;

    // // db->Retrieve(0, k1);
    // // db->Retrieve(1, k2);
    // cv::drawMatches( imgs->loadImages(0), k1, imgs->loadImages(1), k2, good_matches, img_matches, cv::Scalar::all(-1),
    //              cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // //-- Show detected matches
    // std::string window_name = "good_matches";
    //  cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    //    // 2. Set the window property to full screen.
    //    // WND_PROP_FULLSCREEN is the window property to change.
    //    // WINDOW_FULLSCREEN is the value to set for the property.
    // cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    // cv::imshow(window_name, img_matches );
    // cv::waitKey(0);
    // // cv::Mat F;
    // // std::cout << da.geometricVerification(k1, k2, good_matches, F);
    // // std::cout << F;
}
