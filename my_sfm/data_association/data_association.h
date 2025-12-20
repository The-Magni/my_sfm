#pragma once

#include "images.h"
#include "opencv2/core/types.hpp"
#include "keypoints.h"
#include "descriptors.h"
#include "two_view_geometries.h"
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

#define MIN_NUM_INLIERS 15
#define MIN_INLIER_RATIO 0.5
#define MAX_H_INLIER_RATIO 0.8

class DataAssociation {
    private:
        std::shared_ptr<Images> images_{nullptr};
        std::shared_ptr<KeyPointsDB> keypoints_db_{nullptr};
        std::shared_ptr<DescriptorsDB> descriptors_db_{nullptr};
        std::shared_ptr<TwoViewGeometriesDB> two_view_db_{nullptr};

    public:
        bool Init(
            std::shared_ptr<Images> imgs,
            std::shared_ptr<KeyPointsDB> keypoints_db,
            std::shared_ptr<DescriptorsDB> descriptors_db,
            std::shared_ptr<TwoViewGeometriesDB> two_view_db
        );

        void featureExtraction(unsigned int img_id, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

        void featureMatching(const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &good_matches);

        friend void parellarizedMatchingVerification(
            DataAssociation *da,
            unsigned int img1_id,
            unsigned int img2_id,
            std::mutex &db_mutex,
            std::mutex &read_count_mutex,
            unsigned int &read_count
        );

        bool geometricVerification(
            const std::vector<cv::KeyPoint> &k1,
            const std::vector<cv::KeyPoint> &k2,
            const std::vector<cv::DMatch> &good_matches,
            cv::Mat &F,
            std::vector<Match> &inlier_correspondences
        );

        void Process();
};
