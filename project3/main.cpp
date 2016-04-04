//
// Created by Yan Hang on 3/19/16.
//
#include "quilting.h"
#include <iostream>
#include <gflags/gflags.h>

using namespace std;
using namespace cv;
using namespace quilting;

DEFINE_string(source, "", "Path to input source texture");
DEFINE_string(guide, "", "Path to guide image");
DEFINE_int32(blockSize, 50, "block size");
DEFINE_int32(overlap, 10, "overlap");
DEFINE_int32(outWidth, 400, "width of output image");
DEFINE_int32(outHeight, 300, "height of output image");
DEFINE_double(alpha, 0.02, "weight of transfer");
DEFINE_int32(blurSize, 10, "size of blur kernel");
DEFINE_int32(max_iter, 3, "number of iterations");

int main(int argc, char** argv){

	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv,true);
	CHECK(!FLAGS_source.empty()) << "Path to input source cannot be empty!";

	Mat srcTex = imread(FLAGS_source);
	//sanity check
	CHECK(srcTex.data) << "Can not read source texture";
	CHECK_EQ(FLAGS_blockSize%2, 0) << "Block size must be multiple of 2";

	//debug for seamCut
//	Mat p1 = srcTex(cv::Rect(50,50,100,60)).clone();
//	Mat p2 = srcTex(cv::Rect(100,50,100,60)).clone();
//	Mat b1 = p1(cv::Rect(0,30,100,30)).clone();
//	Mat b2 = p2(cv::Rect(0,0,100,30)).clone();
//	Mat b3 = seamCut(b1,b2);
//
//	Mat ori;
//	vconcat(p1(Rect(0,0,100,45)).clone(), p2(Rect(0,15,100,45)).clone(), ori);
//
//	Mat seamcut4, seamcut;
//	vconcat(p1(Rect(0,0,100,30)).clone(), b3, seamcut4);
//	vconcat(seamcut4, p2(Rect(0,30,100,30)), seamcut);
//	imwrite("cut_ori.jpg", ori);
//	imwrite("cut_DP.jpg", seamcut);

	QuiltingConfig config(FLAGS_blockSize, FLAGS_overlap, FLAGS_outWidth, FLAGS_outHeight, FLAGS_blurSize, (float)FLAGS_alpha);

	Mat guide;
	if(!FLAGS_guide.empty()){
		Mat guideImg = imread(FLAGS_guide);
		cvtColor(guideImg, guide, CV_RGB2GRAY);
		guide.convertTo(guide, CV_32FC1);
		blur(guide, guide, Size(config.blurR,config.blurR));
		CHECK(guide.data) << "Guide image is empty";
		config.outputWidth = guide.cols;
		config.outputHeight = guide.rows;
	}else {
		guide = Mat(config.outputHeight, config.outputWidth, CV_32FC1, Scalar(-1.0));
		config.alpha = 0.0;
	}

	Mat output(config.outputHeight, config.outputWidth, CV_8UC3, Scalar(0,0,0));
	Mat mask(config.outputHeight, config.outputWidth, CV_8UC1, Scalar(0));

	const int& N = FLAGS_max_iter;
	for(auto i=0; i<N; ++i) {
		expandTexture(srcTex, guide, output, mask, config);
		config.blockSize *= 0.67;
		config.alpha = 0.1 * 0.8 * ((float)(i) / (float)(N-1)) + 0.1;
	}
	imwrite("result.jpg", output);

	return 0;
}

