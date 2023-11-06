#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void NormalizeAngleOnComponent(VectorXd& vec, int index) {
  while (vec(index)> M_PI) vec(index) -= 2. * M_PI;
  while (vec(index)<-M_PI) vec(index) += 2. * M_PI;
}

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
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0.0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3-n_aug_;

  // number of sigma points
  sig_points = 2*n_aug_+1;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, sig_points);

  // Weights of sigma points
  weights_.resize(sig_points);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for(int i=1; i < weights_.size(); i++){
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    x_.fill(0.0);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0); // p_x
      x_(1) = meas_package.raw_measurements_(1); // p_y
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
     
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = rho_dot;
    }

    P_.setIdentity();
    P_(2, 2) = 100.0;
    P_(3, 3) = 10.0;

    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);   

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else {
    printf("ERROR - sensor_type_: %d is not defined!\n", meas_package.sensor_type_);
    exit(-1);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  MatrixXd Xsig_aug(n_aug_, sig_points);
  VectorXd x_aug(n_aug_);
  MatrixXd P_aug(n_aug_, n_aug_);
  VectorXd x(n_x_);
  MatrixXd P(n_x_, n_x_);

  // Generate Sigma Points 
  // init x_aug
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_; 

  // create square root matrix
  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i < sig_points/2; i++){
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*sqrt_P_aug.col(i);
    Xsig_aug.col(sig_points/2 + i+1) = x_aug - sqrt(lambda_+n_aug_)*sqrt_P_aug.col(i);
  }

  // Predict sigma points
  for(int i=0; i < sig_points; i++){
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    // avoid division by zero
    if(fabs(yawd) > 0.001){
      px_p = p_x + v/yawd*(sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd*(-cos(yaw + yawd*delta_t) + cos(yaw));
    }
    else{
      px_p = p_x + v*cos(yaw)*delta_t;
      py_p = p_y + v*sin(yaw)*delta_t;
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p    = px_p + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
    py_p    = py_p + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
    v_p     = v_p + nu_a*delta_t;
    yaw_p   = yaw_p + 0.5*delta_t*delta_t*nu_yawdd;
    yawd_p  = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // predict state mean 
  x.fill(0.0);
  for (int i = 0; i < sig_points; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < sig_points; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    NormalizeAngleOnComponent(x_diff, 3);
    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  // update x_ and P_
  x_ = x;
  P_ = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2;
  
  MatrixXd Zsig(n_z, sig_points);
  VectorXd z_pred(n_z);
  MatrixXd S(n_z,n_z);
  MatrixXd Tc(n_x_,n_z);

  Zsig.fill(0.0);
  for (int i=0; i<sig_points; i++)
  {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < sig_points; i++) 
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }   

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < sig_points; i++) 
  {  
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  Tc.fill(0.0);
  for(int i=0; i < sig_points; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    NormalizeAngleOnComponent(x_diff, 3);
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;

  S = S + R;  

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  x_ = x_ + K*z_diff;

  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Predict measurement 
  int n_z = 3;

  MatrixXd Zsig(n_z, sig_points);
  VectorXd z_pred(n_z);
  MatrixXd S(n_z, n_z);
  MatrixXd R(3, 3);
  MatrixXd Tc(n_x_, n_z);

  for(int i=0; i < Zsig.cols(); i++){
    double px   = Xsig_pred_(0, i);
    double py   = Xsig_pred_(1, i);
    double v    = Xsig_pred_(2, i);
    double yaw  = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(px*px + py*py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px*cos(yaw)*v + py*sin(yaw)*v) / sqrt(px*px + py*py);
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < sig_points; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < sig_points; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred; 
      NormalizeAngleOnComponent(z_diff, 1);
      S = S + weights_(i)*z_diff*z_diff.transpose();
  }

 // add measurment noise corvariance matrix
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
  
  S = S + R;

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i < sig_points; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngleOnComponent(z_diff, 1);

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngleOnComponent(x_diff, 3);

    Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Update state
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  NormalizeAngleOnComponent(z_diff, 1);
  x_ = x_ + K*z_diff;

  // Update corvariance matrix
  P_ = P_ - K*S*K.transpose();
}