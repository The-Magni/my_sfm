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
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const extrinsic_params,
                    const T* const intrinsic_params,
                    const T* const pp_const,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(extrinsic_params, point, p);
        T x = extrinsic_params[3], y = extrinsic_params[4], z = extrinsic_params[5];
        p[0] += x;
        p[1] += y;
        p[2] += z;

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        const T &l1 = intrinsic_params[1];
        const T &l2 = intrinsic_params[2];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + l1 * r2 + l2 * r2 * r2;

        const T &focal = intrinsic_params[0];
        const T &px = pp_const[0];
        const T &py = pp_const[1];

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
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 2, 3>(
            new ReprojectionError(observed_x, observed_y));
    }

    double observed_x;
    double observed_y;
};

struct UnitNormError {
    UnitNormError() {}

    template <typename T>
    bool operator()(const T* const extrinsic_params, T *residuals) const {
        T x = extrinsic_params[3], y = extrinsic_params[4], z = extrinsic_params[5];
        residuals[0] = x*x + y*y + z*z - T(1.0);
        return true;
    }

    static ceres::CostFunction *Create() {
        return new ceres::AutoDiffCostFunction<UnitNormError, 1, 6>(
            new UnitNormError());
    }
};

void optimize(
    PointCloud &pointcloud,
    std::vector<Camera> &cameras,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int first_img_id,
    unsigned int second_img_id
) {
    ceres::Problem problem;
    std::map<unsigned int, std::vector<cv::KeyPoint>> cache;
    // perform global BA, too lazy to code like colmap paper, will optimize if have time
    for (Point &point : pointcloud.points) {
        for (const Observation &observation : point.observations) {
            if (!cache.count(observation.img_id)) {
                std::vector<cv::KeyPoint> keypoints;
                keypoints_db->Retrieve(observation.img_id, cache[observation.img_id]);
            }
            cv::Point2f point2D = cache[observation.img_id].at(observation.point_id).pt;
            // std::cout << point.pt << '\n' << point2D << '\n' << cameras[observation.img_id].getIntrinsicMat();
            ceres::CostFunction *cost_function = ReprojectionError::Create(point2D.x, point2D.y);
            problem.AddResidualBlock(
                cost_function,
                new ceres::HuberLoss(1.0),
                cameras[observation.img_id].getExtrinsicParams(),
                cameras[observation.img_id].getIntrinsicParams(),
                cameras[observation.img_id].getPpConst(),
                point.getPt()
            );
            problem.SetParameterBlockConstant(cameras[observation.img_id].getPpConst());
            problem.SetParameterBlockConstant(cameras[observation.img_id].getIntrinsicParams());
        }
    }
    // fix initial camera pose
    problem.SetParameterBlockConstant(cameras[first_img_id].getExtrinsicParams());
    // constraint norm of translation vector of second camera
    ceres::CostFunction *norm_cost_function = UnitNormError::Create();
    problem.AddResidualBlock(
        norm_cost_function,
        new ceres::HuberLoss(1.0),
        cameras[second_img_id].getExtrinsicParams()
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << '\n';
}
