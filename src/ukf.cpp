#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3; // 3

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8; // 0.8

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.0);
  
  // Lesson 7, section 24
  
  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  R_laser = MatrixXd(2, 2);
  R_laser << std_laspx_ * std_laspx_ , 0                        ,
             0                       , std_laspy_ * std_laspy_  ;

  R_radar = MatrixXd(3, 3);
  R_radar << std_radr_ * std_radr_ , 0                         , 0                        ,
             0                     , std_radphi_ * std_radphi_ , 0                        ,
             0                     , 0                         , std_radrd_ * std_radrd_  ;

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0.0);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  if (!is_initialized_) {
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  Prediction(delta_t);

  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      UpdateLidar(measurement_pack);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      UpdateRadar(measurement_pack);
  }
  previous_timestamp_ = measurement_pack.timestamp_;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
  // Lesson 7, section 18
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);
  Xsig_aug.fill(0.0);

  // create augmented mean state
  x_aug.head(n_x_) = x_;

  // create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
 
  return Xsig_aug;
}

void UKF::PredictSigmaPoints(MatrixXd & Xsig_aug, double delta_t) {
  // Lesson 7, section 21
  Xsig_pred_.fill(0.0);

  //predict sigma points
  for (int i = 0; i < 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

double normalize_angle(double angle) {
  angle -= 2*M_PI*floor(angle/(2*M_PI));
  while (angle > M_PI) angle -= 2.*M_PI;
  while (angle < -M_PI) angle += 2.*M_PI;
  return angle;
}

void matrix_angle_norm(MatrixXd &m, int i) {
  for(int j=0, c = m.cols(); j < c; ++j){
    m(i,j) = normalize_angle(m(i,j));
  }
}

void UKF::PredictMeanAndCovariance() {
  // Lesson 7, section 24
  x_ = Xsig_pred_ * weights_;

  MatrixXd xd = Xsig_pred_.colwise() - x_;
  matrix_angle_norm(xd, 3);

  P_ = xd * weights_.asDiagonal() * xd.transpose();
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints();
  PredictSigmaPoints(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  MatrixXd Zs = Xsig_pred_.topRows(2);
  VectorXd zp = Zs*weights_;

  MatrixXd y = Zs.colwise() - zp;

  MatrixXd S = y * weights_.asDiagonal() * y.transpose() + R_laser;

  // create matrix for cross correlation Tc
  MatrixXd Tc = (Xsig_pred_.colwise() - x_) * weights_.asDiagonal() * y.transpose();

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - zp;

  x_ = x_ + (K * z_diff);
  P_ = P_ - K*S*K.transpose();

  double NIS = (z_diff.transpose() * S.inverse() * z_diff)(0);
  NIS_lidar_ = NIS;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // L7, s27

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  // L7, s30
  VectorXd zp = Zsig*weights_;
  MatrixXd y = Zsig.colwise() - zp;
  MatrixXd xd = Xsig_pred_.colwise() - x_;

  matrix_angle_norm(y, 1);
  matrix_angle_norm(xd, 3);

  MatrixXd S = y * weights_.asDiagonal() * y.transpose() + R_radar;

  // create matrix for cross correlation Tc
  MatrixXd Tc = xd * weights_.asDiagonal() * y.transpose();

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - zp;

  while (z_diff(1) >  M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

  x_ = x_ + (K * z_diff);
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = (z_diff.transpose() * S.inverse() * z_diff)(0);
}
