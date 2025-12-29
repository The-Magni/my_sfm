#include "data_association.h"
#include "descriptors.h"
#include "keypoints.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "thread_pool.h"
#include "two_view_geometries.h"
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>

bool DataAssociation::Init(
    std::shared_ptr<Images> img,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    std::shared_ptr<DescriptorsDB> descriptors_db,
    std::shared_ptr<TwoViewGeometriesDB> two_view_db
) {
    images_ = img;
    keypoints_db_ = keypoints_db;
    descriptors_db_ = descriptors_db;
    two_view_db_ = two_view_db;
    return true;
}

void DataAssociation::featureExtraction(unsigned int image_id, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Mat img = images_->loadImages(image_id);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

void DataAssociation::featureMatching(const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &good_matches)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    const float ratio_thresh = 0.7f; // value from opencv tutorial
    // 2 means finding the 2 closet neighbors
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    for (unsigned int i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);
    }
}

/*
 * reject homography case
 * only care abput funcdamental matrix
 */
bool DataAssociation::geometricVerification(
    const std::vector<cv::KeyPoint> &k1,
    const std::vector<cv::KeyPoint> &k2,
    const std::vector<cv::DMatch> &good_matches,
    cv::Mat &F,
    std::vector<Match> &inlier_correspondences
) {
    std::vector<cv::Point2f> points1, points2;
    cv::Mat inlier_mask, H_inlier_mask;
    unsigned int num_inliers = 0, num_H_inliers = 0;
    for (const cv::DMatch &match : good_matches) {
        points1.push_back(k1[match.queryIdx].pt);
        points2.push_back(k2[match.trainIdx].pt);
    }

    // check if this is homography case
    cv::findHomography(points1, points2, cv::RANSAC, 1, H_inlier_mask);
    F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1, 0.99, inlier_mask);
    for (unsigned int i = 0; i < good_matches.size(); i++) {
        if (H_inlier_mask.at<uchar>(i) != 0)
            num_H_inliers++;
        if (inlier_mask.at<uchar>(i) != 0) {
            num_inliers++;
            if (good_matches[i].queryIdx < 0 || good_matches[i].trainIdx < 0)
                continue;
            Match match{static_cast<unsigned int>(good_matches[i].queryIdx), static_cast<unsigned int>(good_matches[i].trainIdx)};
            inlier_correspondences.push_back(match);
        }
    }
    return (num_inliers >= MIN_NUM_INLIERS)
    && ((double) num_inliers / good_matches.size() >= MIN_INLIER_RATIO)
    && ((double) num_H_inliers / good_matches.size() <= MAX_H_INLIER_RATIO);
}

// definitely should apply reader-writer problems
// reader: Retrieve call, writer: Insert call
void parellarizedMatchingVerification(
    DataAssociation *da,
    unsigned int img1_id,
    unsigned int img2_id,
    std::mutex &db_mutex,
    std::mutex &read_count_mutex,
    unsigned int &read_count
) {
    std::cout << "Matching image " << img1_id << ", " << img2_id << '\n';
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> good_matches;
    cv::Mat F;
    std::vector<Match> inlier_correspondences;
    bool is_verified;

    read_count_mutex.lock();
    read_count++;
    if (read_count == 1) db_mutex.lock();
    read_count_mutex.unlock();
    // critical setion: reader
    da->descriptors_db_->Retrieve(img1_id, descriptors1);
    da->keypoints_db_->Retrieve(img1_id, keypoints1);
    da->descriptors_db_->Retrieve(img2_id, descriptors2);
    da->keypoints_db_->Retrieve(img2_id, keypoints2);
    read_count_mutex.lock();
    read_count--;
    if (read_count == 0) db_mutex.unlock();
    read_count_mutex.unlock();

    da->featureMatching(descriptors1, descriptors2, good_matches);
    is_verified = da->geometricVerification(keypoints1, keypoints2, good_matches, F, inlier_correspondences);
    if (!is_verified) return;

    db_mutex.lock();
    // critical section: writer
    da->two_view_db_->Insert(img1_id, img2_id, inlier_correspondences, F);
    db_mutex.unlock();
}

void DataAssociation::Process()
{
    //step 1: feature extraction with SIFT
    for (unsigned int i = 0; i < images_->getNumImgs(); i++) {
        std::vector<cv::KeyPoint> k;
        cv::Mat d;
        featureExtraction(i, k, d);
        // save to the db
        keypoints_db_->Insert(i, k);
        descriptors_db_->Insert(i, d);
    }
    // step 2: feature matching and step 3: geometric verification
    // using  exhaustive search here due to laziness
    ThreadPool threads;
    std::mutex db_mutex;
    std::mutex read_count_mutex;
    unsigned int read_count = 0;
    for (unsigned int i = 0; i < images_->getNumImgs(); i++) {
        for (unsigned int j = i + 1; j < images_->getNumImgs(); j++) {
            DataAssociation *da = this;
            threads.enqueue([da, i, j, &db_mutex, &read_count_mutex, &read_count] {
                parellarizedMatchingVerification(da, i, j, db_mutex, read_count_mutex, read_count);
            });
        }
    }
}
