/*
 *    Filename: block-matching-bm.cpp
 *  Created on: Nov 01, 2017
 *      Author: Timo Hinzmann
 *   Institute: ETH Zurich, Autonomous Systems Lab
 */

// HEADER
#include "aerial-mapper-dense-pcl/block-matching-bm.h"

namespace stereo {
void BlockMatchingBM::computeDisparityMap(
    const RectifiedStereoPair& rectified_stereo_pair,
    DensifiedStereoPair* densified_stereo_pair) const {
  CHECK(densified_stereo_pair);

  // Compute the disparity map.
  bm_->compute(rectified_stereo_pair.image_left,
               rectified_stereo_pair.image_right,
               densified_stereo_pair->disparity_map);

  densified_stereo_pair->disparity_map.convertTo(
      densified_stereo_pair->disparity_map, CV_32F);
  cv::Mat dis_img1;
  densified_stereo_pair->disparity_map.convertTo(dis_img1, CV_8UC1);
  densified_stereo_pair->disparity_map =
      densified_stereo_pair->disparity_map / 16.0;
  cv::Mat dis_img2;
  densified_stereo_pair->disparity_map.convertTo(dis_img2, CV_8UC1);

  static int counter = 0;
  char buf[200];
  snprintf(buf, 200, "/tmp/left_%d.jpg", counter);
  cv::imwrite(std::string(buf), rectified_stereo_pair.image_left);
  snprintf(buf, 200, "/tmp/right_%d.jpg", counter);
  cv::imwrite(std::string(buf), rectified_stereo_pair.image_right);

  snprintf(buf, 200, "/tmp/dis_img1_%d.jpg", counter);
  cv::imwrite(std::string(buf), dis_img1);
  snprintf(buf, 200, "/tmp/dis_img2_%d.jpg", counter++);
  cv::imwrite(std::string(buf), dis_img2);

  // Create masked disparity map.
  cv::Mat disparity_map_masked(densified_stereo_pair->disparity_map.rows,
                               densified_stereo_pair->disparity_map.cols,
                               densified_stereo_pair->disparity_map.type());

  // Apply the rectification mask.
  disparity_map_masked.setTo(kMaxInvalidDisparity);
  densified_stereo_pair->disparity_map.copyTo(disparity_map_masked,
                                              rectified_stereo_pair.mask);
  densified_stereo_pair->disparity_map = disparity_map_masked;
}

}  // namespace stereo
