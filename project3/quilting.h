//
// Created by Yan Hang on 3/19/16.
//

#ifndef CSE555_QUILTING_H
#define CSE555_QUILTING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <random>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace quilting {

	struct QuiltingConfig {
		QuiltingConfig(int b, int o, int w, int h, int br, float a) :
				blockSize(b), overlap(o), outputWidth(w), outputHeight(h), blurR(br), alpha(a) { }

		int blockSize;
		int overlap;
		int outputWidth;
		int outputHeight;
		int blurR;
		float alpha;
	};

	void expandTexture(const cv::Mat &input, const cv::Mat& guide, cv::Mat &output, cv::Mat& mask, const QuiltingConfig &config);

	cv::Mat samplePatch(const cv::Mat &input, const cv::Mat& inputMap, const cv::Mat &output, const cv::Mat& guide, const cv::Mat &mask, const int cx, const int cy,
	                    const QuiltingConfig &config);


	//Find a horizontal seam. make sure that b1 and b2 are in 'horizontal position'.
	// To solve vertical seam, first transpose both patches, solve and then transpose back
	cv::Mat seamCut(const cv::Mat &b1, const cv::Mat &b2);

}//namespace quilting

#endif //CSE555_QUILTING_H
