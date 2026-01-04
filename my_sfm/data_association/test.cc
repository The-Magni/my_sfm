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
    // da.Init(imgs, db, db1, db2);
    // da.Process();

    unsigned int i = 8, j = 10;
    std::vector<Match> inlier_correspondences;
    cv::Mat F;
    // db2->RetrieveBestTwoView(i, j, inlier_correspondences, F);
    db2->Retrieve(i, j, inlier_correspondences, F);
    std::vector<cv::KeyPoint> keypoints1;
    db->Retrieve(i, keypoints1);
    std::vector<cv::KeyPoint> keypoints2;
    db->Retrieve(j, keypoints2);
    std::vector<cv::DMatch> matches;
    for (const Match &match : inlier_correspondences) {
        matches.emplace_back(match.idx1, match.idx2, 0);
        cv::Point2f point1 = keypoints1[match.idx1].pt;
        cv::Point2f point2 = keypoints2[match.idx2].pt;
        cv::Matx31d point1_homo(point1.x, point1.y, 1);
        cv::Matx13d point2_homo(point2.x, point2.y, 1);
        // std::cout << point2_homo * F * point1_homo << '\n';
    }
    cv::Mat img_matches;
    cv::drawMatches( imgs->loadImages(i), keypoints1, imgs->loadImages(j), keypoints2, matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    std::string window_name = "good_matches";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

        // 2. Set the window property to full screen.
        // WND_PROP_FULLSCREEN is the window property to change.
        // WINDOW_FULLSCREEN is the value to set for the property.
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::imshow(window_name, img_matches);
    cv::waitKey(0);
    return 0;
}
