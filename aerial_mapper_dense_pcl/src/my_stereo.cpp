/*
 *    Filename: stereo.cpp
 *  Created on: Nov 01, 2017
 *      Author: Timo Hinzmann
 *   Institute: ETH Zurich, Autonomous Systems Lab
 */

#include "aerial-mapper-dense-pcl/my_stereo.h"
#include <DUtils/FileFunctions.h>
#include <fstream>

namespace stereo {

MyStereo::MyStereo(const std::shared_ptr<aslam::NCamera> ncameras,
               const Settings& settings,
               const BlockMatchingParameters& block_matching_params)
    : ncameras_(ncameras),
      settings_(settings),
      first_frame_(true),
      node_handle_(),
      image_transport_(image_transport::ImageTransport(node_handle_)),
      image_data_dir_("/home/mylxiaoyi/work/ros_aerial_mapper/data/ieu"){
  CHECK(ncameras_);

  cv::Size image_resolution;
  image_resolution.width = (ncameras_->getCamera(kFrameIdx).imageWidth());
  image_resolution.height = (ncameras_->getCamera(kFrameIdx).imageHeight());

  // Undistorter.
  static constexpr float undistortion_alpha = 1.0;
  static constexpr float undistortion_scale = 1.0;
  undistorter_ = aslam::createMappedUndistorter(
      ncameras_->getCamera(kFrameIdx), undistortion_alpha, undistortion_scale,
      aslam::InterpolationMethod::Linear);

  rectifier_.reset(new Rectifier(image_resolution));
  densifier_.reset(new Densifier(block_matching_params, image_resolution));

  // Set the calibration matrix K (assumed to be constant for all frames).
  aslam::PinholeCamera::ConstPtr pinhole_camera_ptr =
      std::dynamic_pointer_cast<const aslam::PinholeCamera>(
          ncameras_->getCameraShared(kFrameIdx));
  stereo_rig_params_.K = pinhole_camera_ptr->getCameraMatrix();

  // Set the camera-IMU transformation (assumed to be constant for all frames).
  T_B_C_ = ncameras_->get_T_C_B(kFrameIdx).inverse();

  // Define the point cloud message.
  point_cloud_ros_msg_.header.frame_id = "/world";
  point_cloud_ros_msg_.height = image_resolution.height;
  point_cloud_ros_msg_.width = image_resolution.width;
  point_cloud_ros_msg_.fields.resize(4);

  point_cloud_ros_msg_.fields[0].name = "x";
  point_cloud_ros_msg_.fields[0].offset = 0;
  point_cloud_ros_msg_.fields[0].count = 1;
  point_cloud_ros_msg_.fields[0].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud_ros_msg_.fields[1].name = "y";
  point_cloud_ros_msg_.fields[1].offset = 4;
  point_cloud_ros_msg_.fields[1].count = 1;
  point_cloud_ros_msg_.fields[1].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud_ros_msg_.fields[2].name = "z";
  point_cloud_ros_msg_.fields[2].offset = 8;
  point_cloud_ros_msg_.fields[2].count = 1;
  point_cloud_ros_msg_.fields[2].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud_ros_msg_.fields[3].name = "rgb";
  point_cloud_ros_msg_.fields[3].offset = 12;
  point_cloud_ros_msg_.fields[3].count = 1;
  point_cloud_ros_msg_.fields[3].datatype = sensor_msgs::PointField::UINT32;

  point_cloud_ros_msg_.point_step = 16;
  point_cloud_ros_msg_.row_step =
      point_cloud_ros_msg_.point_step * point_cloud_ros_msg_.width;
  point_cloud_ros_msg_.data.resize(point_cloud_ros_msg_.row_step *
                                   point_cloud_ros_msg_.height);
  point_cloud_ros_msg_.is_dense = false;

  pub_point_cloud_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
      "/planar_rectification/point_cloud", 100);
}

void MyStereo::addFrames(const Poses& T_G_Bs, const Images& images,
                       AlignedType<std::vector, Eigen::Vector3d>::type* point_cloud,
                       std::vector<int>* point_cloud_intensities) {
  CHECK(point_cloud);
  point_cloud->clear();
  if (point_cloud_intensities) {
    point_cloud_intensities->clear();
  }

  size_t skip = 0u;
  for (size_t i = 0u; i < images.size()-1; ++i) {
    if (++skip % settings_.use_every_nth_image == 0) {
      LOG(INFO) << "Processing image " << i << "/" << images.size();
      AlignedType<std::vector, Eigen::Vector3d>::type point_cloud_tmp;
      std::vector<int> point_cloud_intensities_tmp;

      stereo_rig_params_.t_G_C1 = T_G_Bs[i].getPosition();
      stereo_rig_params_.R_G_C1 = T_G_Bs[i].getRotationMatrix();
      image_distorted_1_ = images[i];

      stereo_rig_params_.t_G_C2 = T_G_Bs[i+1].getPosition();
      stereo_rig_params_.R_G_C2 = T_G_Bs[i+1].getRotationMatrix();
      image_distorted_2_ = images[i+1];

      RectifiedStereoPair rectified_stereo_pair;
      rectifier_->rectifyStereoPair(stereo_rig_params_, image_distorted_1_,
                                    image_distorted_2_, &rectified_stereo_pair);

      // New projection matrices [Eq. (9)]
      const Eigen::Matrix<double, 3, 4> P1_rect =
          stereo_rig_params_.K *
          (Eigen::Matrix<double, 3, 4>() << rectified_stereo_pair.R_G_C,
           rectified_stereo_pair.R_G_C * (-stereo_rig_params_.t_G_C1)).finished();
      const Eigen::Matrix<double, 3, 4> P2_rect =
          stereo_rig_params_.K *
          (Eigen::Matrix<double, 3, 4>() << rectified_stereo_pair.R_G_C,
           rectified_stereo_pair.R_G_C * (-stereo_rig_params_.t_G_C2)).finished();

      std::string rec_path = image_data_dir_ + "/rec_" + std::to_string(i);
      if (!DUtils::FileFunctions::DirExists(rec_path.c_str()))
          DUtils::FileFunctions::MkDir(rec_path.c_str());

      std::ofstream ofs(rec_path + "/ieu.cfg");
      ofs << "Camera.fx = " << stereo_rig_params_.K(0, 0) << std::endl;
      ofs << "Camera.fy = " << stereo_rig_params_.K(1, 1) << std::endl;
      ofs << "Camera.cx = " << stereo_rig_params_.K(0, 2) << std::endl;
      ofs << "Camera.cy = " << stereo_rig_params_.K(1, 2) << std::endl;
      ofs << "Camera.baseline = " << rectified_stereo_pair.baseline << std::endl;
      ofs << "minDepth = 0.8" << std::endl;
      ofs << "maxDepth = 1.1" << std::endl;
      ofs << "Pn1 = " << P1_rect(0, 0) << " " << P1_rect(0, 1) << " "
          << P1_rect(0, 2) << " " << P1_rect(0, 3) << " "
          << P1_rect(1, 0) << " " << P1_rect(1, 1) << " "
          << P1_rect(1, 2) << " " << P1_rect(1, 3) << " "
          << P1_rect(2, 0) << " " << P1_rect(2, 1) << " "
          << P1_rect(2, 2) << " " << P1_rect(2, 3) << std::endl;
      ofs << "Pn2 = " << P2_rect(0, 0) << " " << P2_rect(0, 1) << " "
          << P2_rect(0, 2) << " " << P2_rect(0, 3) << " "
          << P2_rect(1, 0) << " " << P2_rect(1, 1) << " "
          << P2_rect(1, 2) << " " << P2_rect(1, 3) << " "
          << P2_rect(2, 0) << " " << P2_rect(2, 1) << " "
          << P2_rect(2, 2) << " " << P2_rect(2, 3) << std::endl;
      ofs.close();

      cv::imwrite(rec_path + "/rec_0.jpg", rectified_stereo_pair.image_left);
      cv::imwrite(rec_path + "/rec_1.jpg", rectified_stereo_pair.image_right);
    }
  }
}

void MyStereo::addFrame(const Pose& T_G_B, const Image& image_raw,
                      AlignedType<std::vector, Eigen::Vector3d>::type* point_cloud,
                      std::vector<int>* point_cloud_intensities) {
  CHECK(point_cloud);

  if (first_frame_) {
    // Prepare the first/left frame of the stereo pair.
    stereo_rig_params_.t_G_C1 = (T_G_B * T_B_C_).getPosition();
    stereo_rig_params_.R_G_C1 = (T_G_B * T_B_C_).getRotationMatrix();
    image_distorted_1_ = image_raw;
    first_frame_ = false;
    return;
  }
  // Prepare the second/right frame of the stereo pair.
  stereo_rig_params_.t_G_C2 = (T_G_B * T_B_C_).getPosition();
  stereo_rig_params_.R_G_C2 = (T_G_B * T_B_C_).getRotationMatrix();
  image_distorted_2_ = image_raw;

  processStereoFrame(point_cloud, point_cloud_intensities);

  // Prepare next iteration: The previously second/right frame is
  // now the first/left frame.
  stereo_rig_params_.t_G_C1 = stereo_rig_params_.t_G_C2;
  stereo_rig_params_.R_G_C1 = stereo_rig_params_.R_G_C2;
  image_distorted_1_ = image_distorted_2_;
}

void MyStereo::processStereoFrame(
    AlignedType<std::vector, Eigen::Vector3d>::type* point_cloud,
    std::vector<int>* point_cloud_intensities) {
  // 1. Undistort raw images.
  cv::Mat image_undistorted_1 = image_distorted_1_;
  cv::Mat image_undistorted_2 = image_distorted_2_;
  if (settings_.images_need_undistortion) {
    undistortRawImages(image_distorted_1_, image_distorted_2_,
                       &image_undistorted_1, &image_undistorted_2);
  }

  // 2. Rectify undistorted images.
  RectifiedStereoPair rectified_stereo_pair;
  rectifier_->rectifyStereoPair(stereo_rig_params_, image_undistorted_1,
                                image_undistorted_2, &rectified_stereo_pair);

  // 3. Compute disparity map based on rectified images.
  DensifiedStereoPair densified_stereo_pair;
  densifier_->computeDisparityMap(rectified_stereo_pair,
                                  &densified_stereo_pair);

  // 4. Compute point cloud.
  point_cloud_ros_msg_.data.clear();
  point_cloud_ros_msg_.data.resize(point_cloud_ros_msg_.row_step *
                                   point_cloud_ros_msg_.height);
  ros::Time timestamp = ros::Time::now();
  point_cloud_ros_msg_.header.stamp = timestamp;
  densifier_->computePointCloud(stereo_rig_params_, rectified_stereo_pair,
                                &densified_stereo_pair, point_cloud_ros_msg_);
  *point_cloud = densified_stereo_pair.point_cloud_eigen;
  if (point_cloud_intensities) {
    *point_cloud_intensities = densified_stereo_pair.point_cloud_intensities;
  }
  
  // 5. Publish the point cloud.
  pub_point_cloud_.publish(point_cloud_ros_msg_);
  ros::spinOnce();

  // [Optional] Visualize rectification.
  if (settings_.show_rectification) {
    visualizeRectification(image_undistorted_1, image_undistorted_2,
                           rectified_stereo_pair.image_left,
                           rectified_stereo_pair.image_right);
  }
}

void MyStereo::undistortRawImages(const cv::Mat& image_distorted_1,
                                const cv::Mat& image_distorted_2,
                                cv::Mat* image_undistorted_1,
                                cv::Mat* image_undistorted_2) const {
  CHECK_NOTNULL(image_undistorted_1);
  CHECK_NOTNULL(image_undistorted_2);
  // Undistort the raw images.
  undistorter_->processImage(image_distorted_1, image_undistorted_1);
  undistorter_->processImage(image_distorted_2, image_undistorted_2);
}

void MyStereo::visualizeRectification(
    const cv::Mat& image_undistorted_1, const cv::Mat& image_undistorted_2,
    const cv::Mat& image_undistorted_rectified_1,
    const cv::Mat& image_undistorted_rectified_2) const {
  static int counter = 0;
  cv::Mat images_undistorted, images_undistorted_rectified, all_images;
  cv::hconcat(image_undistorted_1, image_undistorted_2, images_undistorted);
  cv::hconcat(image_undistorted_rectified_1, image_undistorted_rectified_2,
              images_undistorted_rectified);
  for (int i = 50; i < image_undistorted_1.rows; i = i + 50) {
    cv::line(images_undistorted_rectified, cv::Point(0, i),
             cv::Point(image_undistorted_1.cols * 2, i),
             cv::Scalar(255, 255, 255));
  }
  cv::vconcat(images_undistorted, images_undistorted_rectified, all_images);
  cv::imshow("top: undistorted, bottom: undistorted + rectified",
             all_images);
  char buf[200];
  snprintf(buf, 200, "/tmp/recimg_%d.jpg", counter++);
  cv::imwrite(std::string(buf), all_images);
  cv::waitKey(1);
}

}  // namespace stereo
