#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // L5, s7
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size()) {
    cout << "Size of estimations and ground truth data don't match." << endl;
    return rmse;
  }

  if (estimations.size() == 0) {
    cout << "No estimations." << endl;
    return rmse;
  }

  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    VectorXd residual_squared = residual.array() * residual.array();
    rmse += residual_squared;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}
