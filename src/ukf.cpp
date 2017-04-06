#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

static double normalizeAngle(double angle)
{
  return atan2(sin(angle), cos(angle));
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // Mark object as not initialized
  is_initialized_ = false;
  
  // Assign size of the state vector
  n_x_ = 5;
  
  // Assign size of the augmented state vector
  n_aug_ = 7;
  
  // Assign the lambda parameter for the sigma points generation
  lambda_ = 3 - n_x_;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI_2;

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
  
  // Assign initial covariance matrix
  P_ <<
    50, 0,   0,            0,              0,
    0, 50,   0,            0,              0,
    0,  0, 100,            0,              0,
    0,  0,   0, pow(M_PI, 2),              0,
    0,  0,   0,            0, pow(M_PI_2, 2);
  
  // Initialize predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // Initialize vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  
  // Set weights
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    
    //initialize the state x_ with the first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       Convert radar from polar to cartesian coordinates and initialize state.
       */
      double r = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double x = r * cos(phi);
      double y = r * sin(phi);
      
      x_ << x, y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.
       */
      
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  
  this->Prediction(dt);
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if (use_radar_) {
      this->UpdateRadar(meas_package);
    }
  } else {
    // Laser updates
    if (use_laser_) {
      this->UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Step 1: Generate sigma points
  
  // Create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  
  // Calculate square root of P
  MatrixXd A = P_.llt().matrixL();
  
  // Calculate sigma points ...
  // Set sigma points as columns of matrix Xsig
  Xsig.col(0) = x_;
  Xsig.block(0, 1, n_x_, n_x_) = (sqrt(lambda_ + n_x_) * A).colwise() + x_;
  Xsig.block(0, n_x_ + 1, n_x_, n_x_) = (-1 * sqrt(lambda_ + n_x_) * A).colwise() + x_;
  
  // Step 2: Augmentation
  
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  
  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  
  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  // Create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_) << 0., 0.;
  
  // Create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) <<
    pow(std_a_, 2), 0., 0., pow(std_yawdd_, 2);
  
  // Create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) =
    (sqrt(lambda_ + n_aug_) * A_aug).colwise() + x_aug;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) =
    (-1 * sqrt(lambda_ + n_aug_) * A_aug).colwise() + x_aug;
  
  // Step 3: Predit sigma points
  
  // Extract values for better readability
  ArrayXd p_x = Xsig_aug.row(0).array();
  ArrayXd p_y = Xsig_aug.row(1).array();
  ArrayXd v = Xsig_aug.row(2).array();
  ArrayXd yaw = Xsig_aug.row(3).array();
  ArrayXd yawd = Xsig_aug.row(4).array();
  ArrayXd nu_a = Xsig_aug.row(5).array();
  ArrayXd nu_yawdd = Xsig_aug.row(6).array();
    
  // Predicted state values
  ArrayXd px_p, py_p;
    
  // Avoid division by zero
  Array<bool, -1, -1> yawd_is_not_zero = yawd.abs() > 0.001;
  px_p = (yawd_is_not_zero).select(
    p_x + v / yawd * ((yaw + yawd * delta_t).sin() - yaw.sin()),
    p_x + v * delta_t * yaw.cos());
  py_p = (yawd_is_not_zero).select(
    p_y + v / yawd * (yaw.cos() - (yaw + yawd * delta_t).cos()),
    p_y + v * delta_t * yaw.sin());
    
  ArrayXd v_p = v;
  ArrayXd yaw_p = yaw + yawd * delta_t;
  ArrayXd yawd_p = yawd;
  
  // Add noise
  px_p = px_p + 0.5 * nu_a * delta_t * delta_t * yaw.cos();
  py_p = py_p + 0.5 * nu_a * delta_t * delta_t * yaw.sin();
  v_p = v_p + nu_a * delta_t;
    
  yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
  yawd_p = yawd_p + nu_yawdd * delta_t;
    
  // Write predicted sigma point into right column
  Xsig_pred_.row(0) = px_p;
  Xsig_pred_.row(1) = py_p;
  Xsig_pred_.row(2) = v_p;
  Xsig_pred_.row(3) = yaw_p;
  Xsig_pred_.row(4) = yawd_p;
  
  // Step 4: Predict mean and covariance
  
  // Predict state mean
  x_ = Xsig_pred_ * weights_;
  
  x_(3) = normalizeAngle(x_(3));
  
  // Calculate the difference between predicted sigma points
  // and mean
  // This is a modification suggested by:
  // https://www3.nd.edu/~lemmon/courses/ee67033/pubs/julier-ukf-tac-2000.pdf
  MatrixXd x_diff = Xsig_pred_.colwise() - Xsig_pred_.col(0);
  
  // TODO: Perform more efficient angle normalization
  for (int i = 0; i < x_diff.cols(); i++)
  {
    x_diff(3, i) = normalizeAngle(x_diff(3, i));
  }
  
  // Predict state covariance matrix
  P_ = x_diff.cwiseProduct(weights_.transpose().replicate(n_x_, 1)) *
    x_diff.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set measurement dimension, lidar can measure x and y
  const int n_z = 2;
  
  MatrixXd H_laser = MatrixXd(2, n_x_);
  H_laser <<
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;
  
  // Matrix for sigma points in measurement space
  MatrixXd Zsig = H_laser * Xsig_pred_;
  
  // Measurement covariance matrix for laser
  MatrixXd R_laser = MatrixXd(2, 2);
  R_laser <<
    pow(std_laspx_, 2),                  0,
    0,                  pow(std_laspy_, 2);
  
  // Calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;
  
  // Calculate differences between sigma points
  // and mean measurement in measurement space
  // This is a modification suggested by:
  // https://www3.nd.edu/~lemmon/courses/ee67033/pubs/julier-ukf-tac-2000.pdf
  MatrixXd z_diff = Zsig.colwise() - Zsig.col(0);
  
  // Calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S = z_diff.cwiseProduct(weights_.transpose().replicate(n_z, 1)) *
    z_diff.transpose() + R_laser;
  
  // Calculate differences between sigma points
  // and mean
  // This is a modification suggested by:
  // https://www3.nd.edu/~lemmon/courses/ee67033/pubs/julier-ukf-tac-2000.pdf
  MatrixXd x_diff = Xsig_pred_.colwise() - Xsig_pred_.col(0);
  for (int i = 0; i < x_diff.cols(); i++)
  {
    // TODO: Perform more efficient angle normalization for Xsig_pred - x (3)
    x_diff(3, i) = normalizeAngle(x_diff(3, i));
  }
  
  // Calculate cross correlation matrix
  MatrixXd Tc = x_diff.cwiseProduct(weights_.transpose().replicate(n_x_, 1)) *
    z_diff.transpose();
  
  // Calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();
  
  // Update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();
  
  x_(3) = normalizeAngle(x_(3));
  
  NIS_laser_ = (z - z_pred).transpose() * S.inverse() *
    (z - z_pred);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z = 3;
  
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  ArrayXd px = Xsig_pred_.row(0).array();
  ArrayXd py = Xsig_pred_.row(1).array();
  ArrayXd v = Xsig_pred_.row(2).array();
  ArrayXd yaw = Xsig_pred_.row(3).array();

  ArrayXd sqrt_px_2_plus_py_2 = (px.pow(2) + py.pow(2)).sqrt();
  Zsig.row(0) = sqrt_px_2_plus_py_2;
  // TODO: Replace with more efficient atan2 implementation
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 simga points
    // Extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    
    // Measurement model
    Zsig(1, i) = atan2(p_y, p_x); //phi
  }
  // Avoid division by zero by replacing zero with a small number
  Zsig.row(2) = (px * yaw.cos() * v + py * yaw.sin() * v) / sqrt_px_2_plus_py_2.max(0.0001);
  
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // Calculate mean predicted measurement
  z_pred = Zsig * weights_;
  
  MatrixXd R_radar = MatrixXd(n_z, n_z);
  R_radar <<
    pow(std_radr_, 2), 0.,                  0.,
    0.,                pow(std_radphi_, 2), 0.,
    0.,                0.,                  pow(std_radrd_, 2);
  
  
  // Calculate differences between sigma points
  // and mean measurement in measurement space
  // This is a modification suggested by:
  // https://www3.nd.edu/~lemmon/courses/ee67033/pubs/julier-ukf-tac-2000.pdf
  MatrixXd z_diff = Zsig.colwise() - Zsig.col(0);
  for (int i = 0; i < z_diff.cols(); i++)
  {
    // TODO: Perform more efficient angle normalization for Zsig - z_pred (1)
    z_diff(1, i) = normalizeAngle(z_diff(1, i));
  }
  
  // Calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S = z_diff.cwiseProduct(weights_.transpose().replicate(n_z, 1)) *
    z_diff.transpose() + R_radar;
  
  // Calculate differences between sigma points
  // and mean
  // This is a modification suggested by:
  // https://www3.nd.edu/~lemmon/courses/ee67033/pubs/julier-ukf-tac-2000.pdf
  MatrixXd x_diff = Xsig_pred_.colwise() - Xsig_pred_.col(0);
  for (int i = 0; i < x_diff.cols(); i++)
  {
    // TODO: Perform more efficient angle normalization for Xsig_pred - x (3)
    x_diff(3, i) = normalizeAngle(x_diff(3, i));
  }
  
  // Calculate cross correlation matrix
  MatrixXd Tc = x_diff.cwiseProduct(weights_.transpose().replicate(n_x_, 1)) *
    z_diff.transpose();
  
  // Calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();
  
  // Update state mean and covariance matrix
  VectorXd z = meas_package.raw_measurements_;
  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();
  
  x_(3) = normalizeAngle(x_(3));
  
  NIS_radar_ = (z - z_pred).transpose() * S.inverse() *
    (z - z_pred);
}
