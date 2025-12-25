#include "descriptors.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"

DescriptorsDB::DescriptorsDB(std::string db_name)
{
    sqlite3 *db = nullptr;
    table_name = "descriptors";
    this->db_name = db_name;
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

bool DescriptorsDB::Insert(unsigned int img_id, const cv::Mat &descriptors)
{
    thread_local sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    std::stringstream ss;
    int rc;

    if (!db)
        sqlite3_open(db_name.c_str(), &db);

    ss <<  "INSERT INTO " << table_name << " (image_id, rows, cols, data) "
    << "VALUES (?, ?, ?, ?) "
    << "ON CONFLICT(image_id) DO UPDATE SET "
    << "rows = excluded.rows, cols = excluded.cols, data = excluded.data;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);
    // check if data can be serialized
    CV_Assert(descriptors.type() == CV_32F);
    CV_Assert(descriptors.cols == 128);
    CV_Assert(descriptors.isContinuous());

    sqlite3_bind_int(stmt, 1, img_id);
    sqlite3_bind_int(stmt, 2, descriptors.rows);
    sqlite3_bind_int(stmt, 3, descriptors.cols);

    sqlite3_bind_blob(
        stmt,
        4,
        descriptors.data,
        descriptors.rows * descriptors.cols * descriptors.elemSize(),
        SQLITE_TRANSIENT
    );

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cout << "Fail to insert descriptors " << sqlite3_errmsg(db) << '\n';
        sqlite3_finalize(stmt);
        return false;
    }
    sqlite3_finalize(stmt);
    return true;
}

bool DescriptorsDB::Retrieve(unsigned int img_id, cv::Mat &descriptors)
{
    thread_local sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    std::stringstream ss;
    int rc;

    if (!db)
        rc = sqlite3_open(db_name.c_str(), &db);

    ss << "SELECT rows, cols, data FROM " << table_name << " WHERE image_id=?;";
    sqlite3_prepare_v2(db, ss.str().c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, img_id);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        std::cout << "Fail to retrieve descriptors " << sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        return false;
    }

    int rows = sqlite3_column_int(stmt, 0);
    int cols = sqlite3_column_int(stmt, 1);
    CV_Assert(cols == 128);

    const void *blob = sqlite3_column_blob(stmt, 2);
    int bytes = sqlite3_column_bytes(stmt, 2);

    if (bytes != rows * cols * sizeof(float))
        throw std::runtime_error("Size mismatch between real descriptors and stored data");

    descriptors = cv::Mat(rows, cols, CV_32F);
    std::memcpy(descriptors.data, blob, bytes);

    sqlite3_finalize(stmt);
    return true;
}
