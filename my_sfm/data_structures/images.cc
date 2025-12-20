#include "images.h"

Images::Images(std::string dir_name)
{
    for (const auto &entry : std::filesystem::directory_iterator(dir_name)) {
        img_paths.push_back(entry.path().string());
    }
}

cv::Mat Images::loadImages(unsigned int id)
{
    std::string path = img_paths[id];
    return cv::imread(path);
}

unsigned int Images::getNumImgs()
{
    return img_paths.size();
}
