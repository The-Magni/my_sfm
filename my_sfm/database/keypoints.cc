#include "keypoints.h"
#include "opencv2/core/types.hpp"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <sqlite3.h>
#include <stdexcept>

void KeyPointsDB::serialize(const std::vector<cv::KeyPoint> &keypoints, std::vector<SerializedKeyPoint> &points_flatten)
{
    points_flatten.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++) {
        points_flatten.at(i) = {
            keypoints.at(i).pt.x,
            keypoints.at(i).pt.y,
            keypoints.at(i).size,
            keypoints.at(i).angle
        };
    }
}

void KeyPointsDB::deserialize(const std::vector<SerializedKeyPoint> &points_flatten, std::vector<cv::KeyPoint> &keypoints)
{
    keypoints.resize(points_flatten.size());
    for (size_t i = 0; i < points_flatten.size(); i++) {
        keypoints.at(i) = cv::KeyPoint(
            points_flatten.at(i).x,
            points_flatten.at(i).y,
            points_flatten.at(i).size,
            points_flatten.at(i).angle
        );
    }
}

KeyPointsDB::KeyPointsDB(const std::string &db_name)
{
    table_name = "keypoints";
    this->db_name = db_name;
    sqlite3 *db = nullptr;
    std::stringstream ss;

    if (!db)
        int rc = sqlite3_open(db_name.c_str(), &db);

    char *error_msg = nullptr;
    ss.clear();
    ss << "CREATE TABLE IF NOT EXISTS " << table_name << "( "
    << "image_id INTEGER PRIMARY KEY, "
    << "rows INTEGER NOT NULL, "
    << "cols INTEGER NOT NULL, "
    << "data BLOB "
    << " );";
    sqlite3_exec(db, ss.str().c_str(), nullptr, nullptr, &error_msg);
}

bool KeyPointsDB::Insert(unsigned int img_id, const std::vector<cv::KeyPoint> &keypoints)
{
    thread_local sqlite3 *db;
    sqlite3_stmt *stmt = nullptr;
    int rc;
    std::stringstream ss;
    std::vector<SerializedKeyPoint> points_flatten;
    serialize(keypoints, points_flatten);

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss <<  "INSERT INTO " << table_name << " (image_id, rows, cols, data) "
    << "VALUES (?, ?, ?, ?) "
    << "ON CONFLICT(image_id) DO UPDATE SET "
    << "rows = excluded.rows, cols = excluded.cols, data = excluded.data;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);

    sqlite3_bind_int(stmt, 1, img_id);
    sqlite3_bind_int(stmt, 2, keypoints.size());
    sqlite3_bind_int(stmt, 3, cols);
    sqlite3_bind_blob(stmt, 4, points_flatten.data(), keypoints.size() * sizeof(SerializedKeyPoint), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cout << "Fail to insert keypoints " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);
    return true;
}

bool KeyPointsDB::Retrieve(unsigned int img_id, std::vector<cv::KeyPoint> &keypoints)
{
    thread_local sqlite3 *db;
    sqlite3_stmt *stmt = nullptr;
    int rc;
    std::stringstream ss;
    std::vector<SerializedKeyPoint> points_flatten;

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss << "SELECT rows, cols, data FROM " << table_name << " WHERE image_id=?;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, img_id);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        std::cout << "Fail to retrieve keypoints " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }

    int rows = sqlite3_column_int(stmt, 0);
    const void *blob = sqlite3_column_blob(stmt, 2);
    int bytes = sqlite3_column_bytes(stmt, 2);

    if (bytes != rows * sizeof(SerializedKeyPoint))
        throw std::runtime_error("Size mismatch between real keypoints and stored data");

    points_flatten.resize(rows);
    std::memcpy(points_flatten.data(), blob, bytes);
    deserialize(points_flatten, keypoints);

    assert(keypoints.size() == rows);

    sqlite3_finalize(stmt);
    return true;
}
