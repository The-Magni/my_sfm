#include "optimization.h"
#include "camera.h"
#include "opencv2/core/types.hpp"
#include "point_cloud.h"
#include <cassert>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/sphere_manifold.h>
#include <glog/logging.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>


struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const q,
                    const T* const t,
                    const T* const intrinsic,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(q, point, p);
        T x = t[0], y = t[1], z = t[2];
        p[0] += x;
        p[1] += y;
        p[2] += z;
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        const T &k = intrinsic[3];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + k * r2;

        const T &focal = intrinsic[0];
        const T &px = intrinsic[1];
        const T &py = intrinsic[2];

        T predicted_x = focal * distortion * xp + px;
        T predicted_y = focal * distortion * yp + py;
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);


        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        const double observed_x,
        const double observed_y
    ) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 4, 3>(
            new ReprojectionError(observed_x, observed_y));
    }

    double observed_x;
    double observed_y;
};

void optimize(
    PointCloud &pointcloud,
    std::vector<Camera> &cameras,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int first_img_id,
    unsigned int second_img_id,
    bool is_first_two_views
) {
    ceres::Problem problem;
    std::map<unsigned int, std::vector<cv::KeyPoint>> cache;
    // perform global BA, too lazy to code like colmap paper, will optimize if have time
    for (unsigned int i = 0; i < cameras.size(); i++) {
        Camera &cam = cameras.at(i);
        problem.AddParameterBlock(cam.getQuartenionParams(), 4, new ceres::QuaternionManifold());
        if (i == second_img_id)
            problem.AddParameterBlock(cam.getTranslationParams(), 3, new ceres::SphereManifold<3>());
        else
            problem.AddParameterBlock(cam.getTranslationParams(), 3);

        // if (is_first_two_views) {
        //     problem.AddParameterBlock(cam.getIntrinsicParams(), 5);
        //     problem.SetParameterBlockConstant(cam.getIntrinsicParams());
        // } else {
        std::vector fix_pp = {1, 2};
        problem.AddParameterBlock(cam.getIntrinsicParams(), 4, new ceres::SubsetManifold(4, fix_pp));
        // }
    }
    for (Point &point : pointcloud.points) {
        problem.AddParameterBlock(point.getPt(), 3);
        for (const Observation &observation : point.observations) {
            if (!cache.count(observation.img_id)) {
                keypoints_db->Retrieve(observation.img_id, cache[observation.img_id]);
            }
            cv::Point2f point2D = cache[observation.img_id].at(observation.point_id).pt;

            ceres::CostFunction *cost_function = ReprojectionError::Create(point2D.x, point2D.y);
            problem.AddResidualBlock(
                cost_function,
                new ceres::SoftLOneLoss(1.0),
                cameras[observation.img_id].getQuartenionParams(),
                cameras[observation.img_id].getTranslationParams(),
                cameras[observation.img_id].getIntrinsicParams(),
                point.getPt()
            );
        }
    }
    // fix initial camera pose
    problem.SetParameterBlockConstant(cameras[first_img_id].getQuartenionParams());
    problem.SetParameterBlockConstant(cameras[first_img_id].getTranslationParams());
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = std::thread::hardware_concurrency();

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport() << '\n';
}
