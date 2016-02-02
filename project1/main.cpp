#include <iostream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <Eigen/Eigen>

using namespace std;
using namespace cv;
using namespace Eigen;

Vector2i computeOffset(const Mat& src, const Mat& tgt, const Vector2i init, const int radius = 10);
void myPynDown(const Mat& input, Mat& output);

int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "usage: ./project1 <path-to-image>" << endl << flush;
		return 1;
	}

	google::InitGoogleLogging(argv[0]);

	Mat input = imread(argv[1]);
	CHECK_GT(input.cols, 0) << "Image not loaded!";
	cvtColor(input, input, CV_BGR2GRAY);

	//divide images into BGR
	cout << "Splitting images" << endl;
	vector<Mat> chn(3); //BGR
	cv::Size singleSize(input.cols, input.rows / 3);
	for (auto i = 0; i < chn.size(); ++i) {
		chn[i] = input(Range(i * singleSize.height, (i + 1) * singleSize.height), Range::all()).clone();
		imwrite("channel"+to_string(i)+".png", chn[i]);
		cout << chn[i].cols << ' ' << chn[i].rows << endl;
	}

	cout << "Computing gradient..." << endl;
	vector<Mat> gm(3); //gradient magnitude
//	for (auto i = 0; i < 3; ++i) {
//		Mat gx, gy;
//		Sobel(chn[i], gx, CV_32F, 1, 0);
//		Sobel(chn[i], gy, CV_32F, 0, 1);
//		gm[i] = Mat(gx.rows, gx.cols, CV_32F);
//		float *pGx = (float *) gx.data;
//		float *pGy = (float *) gy.data;
//		float *pGm = (float *) gm[i].data;
//		for (int j = 0; j < gm[i].cols * gm[i].rows; ++j) {
//			pGm[j] = sqrt(pGx[j] * pGx[j] + pGy[j] * pGy[j]);
//		}
//		imwrite("gm"+to_string(i)+".png", gm[i]);
//	}
	for(auto i=0; i<3; ++i)
		chn[i].convertTo(gm[i], CV_32F);


	//align each channel to first channel
	const int kLevel = 4;
	vector<Vector2i> offset(3, Vector2i(0, 0));
	vector<Mat> pyramid_tgt(kLevel);
	pyramid_tgt[0] = gm[0].clone();
	for (auto j = 1; j < pyramid_tgt.size(); ++j)
		myPynDown(pyramid_tgt[j - 1], pyramid_tgt[j]);
	for(auto i=0; i<kLevel; ++i)
		imwrite("pyramid"+to_string(i)+".png", pyramid_tgt[i]);

	for (auto i = 1; i < 3; ++i) {
		//construct pyramid
		cout << "=============" << endl;
		cout << "Channel " << i << endl;
		cout << "Constructing pyrmid" << endl;
		vector<Mat> pyramid(kLevel);
		pyramid[0] = gm[i].clone();
		for (auto j = 1; j < pyramid.size(); ++j)
			myPynDown(pyramid[j - 1], pyramid[j]);
		//align bottom up
		cout << "Computing offset" << endl;
		for (int j = pyramid.size() - 1; j >= 0; --j) {
			cout << j << ' ';
			offset[i] = computeOffset(pyramid[j], pyramid_tgt[j], offset[i] * 2);
		}
		cout << endl;
		cout << "Offset for channel " << i << ": " << offset[i][0] << ' ' << offset[i][1] << endl;
	}

	//compose color image and auto corp
	cout << "Composing and corpping" << endl;
	Mat colorImg(chn[0].rows, chn[0].cols, CV_8UC3, Scalar(0, 0, 0));
	vector<int> bound(4);
	bound[0] = colorImg.cols;
	bound[1] = colorImg.rows;
	bound[2] = 0;
	bound[3] = 0;

	uchar *pColor = colorImg.data;
	vector<const uchar *> pCh(3);
	for (auto j = 0; j < 3; ++j)
		pCh[j] = chn[j].data;

	for (auto y = 0; y < colorImg.rows; ++y) {
		for (auto x = 0; x < colorImg.cols; ++x) {
			const int idx = y * colorImg.cols + x;
			bool valid = true;
			for (auto j = 0; j < 3; ++j) {
				int curx = x + offset[j][0];
				int cury = y + offset[j][1];
				if (curx < 0 || cury < 0 || curx >= colorImg.cols || cury >= colorImg.rows) {
					valid = false;
					break;
				}
				pColor[idx * 3 + j] = pCh[j][cury * chn[j].cols + curx];
			}
			//update bound
			if (valid) {
				bound[0] = min(bound[0], x);
				bound[1] = min(bound[1], y);
				bound[2] = max(bound[2], x);
				bound[3] = max(bound[3], y);
			}
		}
	}

	Mat corpped = colorImg(cv::Rect(bound[0], bound[1], bound[2] - bound[0], bound[3] - bound[1])).clone();
	imwrite("result.png", corpped);
	return 0;
}

Vector2i computeOffset(const Mat& src, const Mat& tgt, const Vector2i init, const int radius){
	CHECK_EQ(src.size(), tgt.size());
	CHECK_EQ(src.type(), CV_32F);
	CHECK_EQ(tgt.type(), CV_32F);


	Vector2i bestOffset(0,0);
	float bestScore = numeric_limits<float>::max();
	const float* pSrc = (float*)src.data;
	const float* pTgt = (float*)tgt.data;
	const int w = src.cols;
	const int h = src.rows;
	const int bx = w / 10;
	const int by = h / 10;
	for(int offx = -1*radius; offx <= radius; ++offx){
		for(int offy = -1*radius; offy <= radius; ++offy){
			float curscore = 0;
			for(auto x=bx; x<w-bx; ++x){
				for(auto y=by; y<h-by; ++y){
					int curx = x + offx + init[0], cury = y + offy + init[1];
					if(curx < 0 || cury < 0 || curx >= w || cury >= h)
						continue;
					float diff = pSrc[cury * w + curx] - pTgt[y * w + x];
					curscore += diff * diff;
				}
			}
			if(curscore < bestScore){
				bestScore = curscore;
				bestOffset[0] = init[0] + offx;
				bestOffset[1] = init[1] + offy;
			}
		}
	}
	cout << "bestScore:" << bestScore << " best offset: " << bestOffset[0] << ' ' << bestOffset[1] << endl << flush;
	return bestOffset;
}

void myPynDown(const Mat& input, Mat& output) {
	CHECK_EQ(input.type(), CV_32F);
	Mat blurred;
	blur(input, blurred, cv::Size(3,3));
	output = Mat(input.rows / 2, input.cols / 2, input.type());
	const int kInputPix = input.cols * input.rows;
	float *pOutput = (float *) output.data;
	const float *pInput = (float *) blurred.data;
	for (auto x = 0; x < output.cols; ++x) {
		for (auto y = 0; y < output.rows; ++y) {
			int orix = x * 2;
			int oriy = y * 2;
			int oriidx = oriy * input.cols + orix;
			CHECK_LT(oriidx, kInputPix);
			pOutput[y * output.cols + x] = pInput[oriidx];
		}
	}
}