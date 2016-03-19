//
// Created by Yan Hang on 3/19/16.
//

#ifndef CSE555_QUILTING_H
#define CSE555_QUILTING_H

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace quilting{

	struct QuiltingConfig{
		QuiltingConfig(int b, int o, int w, int h):
				blockSize(b), overlap(o), outputWidth(w), outputHeight(h){}
		int blockSize;
		int overlap;
		int outputWidth;
		int outputHeight;
	};

	void expandTexture(const cv::Mat& input, cv::Mat& output, const QuiltingConfig& config);

	cv::Mat samplePatch(const cv::Mat& input, const cv::Mat& output, const cv::Mat& mask, const int x, const int y, const QuiltingConfig& config);

	cv::Mat seamCut(const cv::Mat& b1, const cv::Mat& b2, bool vertical);

}//namespace quilting

#endif //CSE555_QUILTING_H
