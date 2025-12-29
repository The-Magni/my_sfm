#pragma once

#include "opencv2/core/mat.hpp"
#include <sqlite3.h>
#include <string>
#include <vector>

#define MAX_INT 2147483647

struct Match {
    unsigned int idx1;
    unsigned int idx2;

    bool operator==(const Match &other) const {
        return other.idx1 == idx1 && other.idx2 == idx2;
    }
};

class TwoViewGeometriesDB {
    private:
        std::string table_name;
        std::string db_name;
        const int cols = 2;

        unsigned long long imgIdsToPairId(unsigned int img1_id, unsigned int img2_id);

        void pairIdToImgIds(unsigned long long pair_id, unsigned int &img1_id, unsigned int &img2_id);

    public:
        TwoViewGeometriesDB(std::string db_name);

        bool Insert(
            unsigned int img1_id,
            unsigned int img2_id,
            const std::vector<Match> &inlier_correspondences,
            const cv::Mat &F
        );

        bool Retrieve(
            unsigned int img1_id,
            unsigned int img2_id,
            std::vector<Match> &inlier_correspondences,
            cv::Mat &F
        );

        bool RetrieveBestTwoView(
            unsigned int &img1_id,
            unsigned int &img2_id,
            std::vector<Match> &inlier_correspondences,
            cv::Mat &F
        );
};
