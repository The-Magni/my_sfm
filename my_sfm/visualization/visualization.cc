#include "visualization.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/viz/types.hpp"
#include "opencv2/viz/viz3d.hpp"
#include "opencv2/viz/widgets.hpp"
#include <cassert>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/quaternion.hpp>
#include <sstream>
#include <string>
#include <vector>

void Visualization::Init(const std::string &dir_path)
{
    points3D_path = dir_path + "/points3D.txt";
    cameras_path = dir_path + "/cameras.txt";
}

bool Visualization::readPoints3D(cv::Mat &cloud, cv::Mat &colors)
{
    std::string line, word;
    std::ifstream points3D_file(points3D_path);
    unsigned int num_points = 0, i = 0;
    std::vector<std::string> words;
    words.reserve(6);
    if (!points3D_file.is_open()) return false;

    while (std::getline(points3D_file, line))
        num_points++;
    points3D_file.close();

    points3D_file.open(points3D_path);
    if (!points3D_file.is_open()) return false;
    cloud = cv::Mat(num_points, 1, CV_64FC3);
    colors = cv::Mat(num_points, 1, CV_8UC3);

    while (std::getline(points3D_file, line)) {
        std::stringstream ss(line);
        words.clear();
        while (ss >> word)
            words.push_back(word);
        assert(words.size() == 6);
        cloud.at<cv::Vec3d>(i, 0) = cv::Vec3d(std::stod(words[0]), std::stod(words[1]), std::stod(words[2]));
        colors.at<cv::Vec3b>(i, 0) = cv::Vec3b (
            static_cast<unsigned char>(std::stoul(words[3])),
            static_cast<unsigned char>(std::stoul(words[4])),
            static_cast<unsigned char>(std::stoul(words[5]))
        );
        i++;
    }
    points3D_file.close();
    return true;
}

void Visualization::Process()
{
    cv::Mat cloud, colors;
    std::vector<cv::Affine3d> poses;
    std::vector<cv::Matx33d> Ks;
    if (!readPoints3D(cloud, colors)){
        LOG(ERROR) << "Cannot open " << points3D_path << '\n';
       return;
    }
    if (!readCameras(poses, Ks)) {
        LOG(ERROR) << "Cannot open " << cameras_path << '\n';
        return;
    }
    cv::viz::Viz3d viz_window("Point Cloud and Cameras");
    cv::viz::WCloud cloud_widget(cloud, colors);
    viz_window.setBackgroundColor(cv::viz::Color::black());
    viz_window.showWidget("Coordinate", cv::viz::WCoordinateSystem(1.0));
    viz_window.showWidget("Point Cloud", cloud_widget);

    assert(poses.size() == Ks.size());
    for (unsigned int i = 0; i < poses.size(); i++) {
        cv::viz::WCameraPosition cam_widget(Ks[i], 0.3);
        viz_window.showWidget(
            "camera " + std::to_string(i),
            cam_widget,
            poses[i]
        );
        cam_widget.setColor(cv::viz::Color::red());
    }
    viz_window.spin();
}

bool Visualization::readCameras(std::vector<cv::Affine3d> &poses, std::vector<cv::Matx33d> &Ks)
{
    std::string line, word;
    std::ifstream cameras_file(cameras_path);
    std::vector<double> nums;
    double num;
    cv::Matx33d R;
    cv::Matx33d K = cv::Matx33d::eye();
    nums.reserve(11);

    if (!cameras_file.is_open()) return false;
    while(std::getline(cameras_file, line)) {
        std::stringstream ss(line);
        nums.clear();
        while (ss >> num)
            nums.push_back(num);
        assert(nums.size() == 11);
        cv::Quatd q(nums[0], nums[1], nums[2], nums[3]);
        cv::Vec3d t(nums[4], nums[5], nums[6]);
        R = q.toRotMat3x3();
        poses.push_back(cv::Affine3d(R, t));
        K(0, 0) = nums[7];
        K(1, 1) = nums[7];
        K(0, 2) = nums[8];
        K(1, 2) = nums[9];
        Ks.push_back(K);
    }
    cameras_file.close();
    return true;
}
