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
        p[0] += extrinsic_params[3];
        p[1] += extrinsic_params[4];
        p[2] += extrinsic_params[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        const T &l1 = intrinsic_params[1];
        const T &l2 = intrinsic_params[2];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + l1 * r2 + l2 * r2 * r2;

        const T &focal = intrinsic_params[0];
        const T &px = pp_const[1];
        const T &py = pp_const[2];

        T predicted_x = focal * distortion * xp + px;
        T predicted_y = focal * distortion * yp + py;
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 2, 3>(
            new ReprojectionError(observed_x, observed_y));
    }

    double observed_x;
    double observed_y;
};

void optimize(
    PointCloud &pointcloud,
    std::vector<Camera> &cameras,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int fixed_img_id
) {
    ceres::Problem problem;
    // perform global BA, too lazy to code like colmap paper, will optimize if have time
    for (Point &point : pointcloud.points) {
        for (const Observation &observation : point.observations) {
            std::vector<cv::KeyPoint> keypoints;
            keypoints_db->Retrieve(observation.img_id, keypoints);
            cv::Point2f point2D = keypoints.at(observation.point_id).pt;
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
            // set bound for focal length
            problem.SetParameterBlockConstant(cameras[observation.img_id].getIntrinsicParams());
            problem.SetParameterBlockConstant(cameras[observation.img_id].getIntrinsicParams());
        }
    }
    // fix initial camera pose
    problem.SetParameterBlockConstant(cameras[fixed_img_id].getExtrinsicParams());


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << '\n';
}
