#include "two_view_geometries.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

unsigned long long TwoViewGeometriesDB::imgIdsToPairId(unsigned int img1_id, unsigned int img2_id)
{
    if (img1_id > img2_id)
        return static_cast<unsigned long long>(MAX_INT) * img2_id + img1_id;
    else
        return static_cast<unsigned long long>(MAX_INT) * img1_id + img2_id;
}

void TwoViewGeometriesDB::pairIdToImgIds(unsigned long long pair_id, unsigned int &img1_id, unsigned int &img2_id)
{
    img2_id = pair_id % MAX_INT;
    img1_id = (pair_id - img2_id) / MAX_INT;
}

TwoViewGeometriesDB::TwoViewGeometriesDB(std::string db_name)
{
    sqlite3 *db = nullptr;
    std::stringstream ss;
    int rc;

    table_name = "two_view_geometries";
    this->db_name = db_name;

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);


    char *error_msg = nullptr;
    ss.clear();
    ss << "CREATE TABLE IF NOT EXISTS " << table_name << "( "
    << "pair_id INTEGER PRIMARY KEY, "
    << "rows INTEGER NOT NULL, "
    << "cols INTEGER NOT NULL, "
    << "matches BLOB, "
    << "geometry BLOB "
    << " );";
    sqlite3_exec(db, ss.str().c_str(), nullptr, nullptr, &error_msg);
}


bool TwoViewGeometriesDB::Insert(
    unsigned int img1_id,
    unsigned int img2_id,
    const std::vector<Match> &inlier_correspondences,
    const cv::Mat &F
) {
    thread_local sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    int rc;
    std::stringstream ss;
    unsigned long long pair_id = imgIdsToPairId(img1_id, img2_id);

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss <<  "INSERT INTO " << table_name << " (pair_id, rows, cols, matches, geometry) "
    << "VALUES (?, ?, ?, ?, ?) "
    << "ON CONFLICT(pair_id) DO UPDATE SET "
    << "rows = excluded.rows, cols = excluded.cols, matches = excluded.matches, geometry = excluded.geometry;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);

    CV_Assert(F.type() == CV_64F);
    CV_Assert(F.rows == 3);
    CV_Assert(F.cols == 3);
    CV_Assert(F.isContinuous());

    sqlite3_bind_int64(stmt, 1, pair_id);
    sqlite3_bind_int(stmt, 2, inlier_correspondences.size());
    sqlite3_bind_int(stmt, 3, cols);
    sqlite3_bind_blob(stmt, 4, inlier_correspondences.data(), inlier_correspondences.size() * sizeof(Match), SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 5, F.data, F.rows * F.cols * F.elemSize(), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cout << "Fail to insert two_view " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);
    return true;
}

bool TwoViewGeometriesDB::Retrieve(
    unsigned int img1_id,
    unsigned int img2_id,
    std::vector<Match> &inlier_correspondences,
    cv::Mat &F
) {
    thread_local sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    int rc;
    std::stringstream ss;
    unsigned long long pair_id = imgIdsToPairId(img1_id, img2_id);

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss << "SELECT rows, cols, matches, geometry FROM " << table_name << " WHERE pair_id=?;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, pair_id);
    rc = sqlite3_step(stmt);

    // no data is found (no edge between 2 images in the scene graph)
    if (rc == SQLITE_DONE) {
        std::cout << "No edge between image id " << img1_id << " and " << img2_id << '\n';
        sqlite3_finalize(stmt);
        return false;
    }
    if (rc != SQLITE_ROW) {
        std::cout << "Fail to retrieve two_view " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }

    int rows = sqlite3_column_int(stmt, 0);
    int cols = sqlite3_column_int(stmt, 1);
    const void *matches_blob = sqlite3_column_blob(stmt, 2);
    int matches_bytes = sqlite3_column_bytes(stmt, 2);
    const void *geometry_blob = sqlite3_column_blob(stmt, 3);
    int geometry_bytes = sqlite3_column_bytes(stmt, 3);

    assert(cols == 2);
    assert(matches_bytes == rows * sizeof(Match));
    assert(geometry_bytes == 9 * sizeof(double));

    inlier_correspondences.resize(rows);
    std::memcpy(inlier_correspondences.data(), matches_blob, matches_bytes);
    F = cv::Mat(3, 3, CV_64F);
    std::memcpy(F.data, geometry_blob, geometry_bytes);

    sqlite3_finalize(stmt);
    return true;
}

bool TwoViewGeometriesDB::RetrieveBestTwoView(
    unsigned int &img1_id,
    unsigned int &img2_id,
    std::vector<Match> &inlier_correspondences,
    cv::Mat &F
) {
    thread_local sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    int rc;
    std::stringstream ss;

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss << "SELECT pair_id, rows, cols, matches, geometry FROM " << table_name
    << " WHERE rows = (SELECT MAX(rows) FROM " << table_name << " ) LIMIT 1;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);

    rc = sqlite3_step(stmt);

    // no data is found (no edge between 2 images in the scene graph)
    if (rc != SQLITE_ROW) {
        std::cout << "Fail to retrieve best two_view " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }

    unsigned long long pair_id = sqlite3_column_int64(stmt, 0);
    int rows = sqlite3_column_int(stmt, 1);
    int cols = sqlite3_column_int(stmt, 2);
    const void *matches_blob = sqlite3_column_blob(stmt, 3);
    int matches_bytes = sqlite3_column_bytes(stmt, 3);
    const void *geometry_blob = sqlite3_column_blob(stmt, 4);
    int geometry_bytes = sqlite3_column_bytes(stmt, 4);

    assert(cols == 2);
    assert(matches_bytes == rows * sizeof(Match));
    assert(geometry_bytes == 9 * sizeof(double));

    pairIdToImgIds(pair_id, img1_id, img2_id);
    inlier_correspondences.resize(rows);
    std::memcpy(inlier_correspondences.data(), matches_blob, matches_bytes);
    F = cv::Mat(3, 3, CV_64F);
    std::memcpy(F.data, geometry_blob, geometry_bytes);

    sqlite3_finalize(stmt);
    return true;
}
