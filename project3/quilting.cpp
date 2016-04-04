//
// Created by Yan Hang on 3/19/16.
//

#include "quilting.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace quilting{

	cv::Mat seamCut(const cv::Mat& b1, const cv::Mat& b2){
		CHECK_EQ(b1.size(), b2.size());
		CHECK_GE(b1.cols, b1.rows);
		const int w = b1.cols;
		const int h = b1.rows;
		vector<vector<double> > Dis(b1.cols); //squared distance
		for(auto& d: Dis)
			d.resize(b1.rows);
		const uchar* pb1 = b1.data;
		const uchar* pb2 = b2.data;
		for(auto y=0; y<h; ++y){
			for(auto x=0; x<w; ++x){
				int idx = y  * w + x;
				Vector3d diff = Vector3d(pb1[3*idx], pb1[3*idx+1], pb1[3*idx+2]) - Vector3d(pb2[3*idx], pb2[3*idx+1], pb2[3*idx+2]);
				Dis[x][y] = diff.norm();
			}
		}

		//DP
		vector<vector<double> > DP(b1.cols);
		for(auto& dp: DP)
			dp.resize(b1.rows, numeric_limits<double>::max());

		//for back track. track[i][j] = 0: (i-1,j); -1:(i-1,j-1); 1:(i-1,j+1);
		vector<vector<int> > track(b1.cols);
		for(auto& t: track)
			t.resize(b1.rows);
		for(auto i=0; i<h; ++i)
			DP[0][i] = Dis[0][i];
		for(auto x=1; x<w; ++x) {
			for (auto y = 0; y < h; ++y){
				double minE = numeric_limits<double>::max();
				if(y > 0){
					double e = Dis[x][y] + DP[x-1][y-1];
					if(e < minE){
						minE = e;
						DP[x][y] = e;
						track[x][y] = -1;
					}
				}
				if(y < h-1){
					double e = Dis[x][y] + DP[x-1][y+1];
					if(e < minE){
						minE = e;
						DP[x][y] = e;
						track[x][y] = 1;
					}
				}
				double e = Dis[x][y] + DP[x-1][y];
				if(e < minE){
					minE = e;
					DP[x][y] = e;
					track[x][y] = 0;
				}
			}
		}

		//back track
		vector<int> cut(w);
		double minE_last = numeric_limits<double>::max();
		for(auto y=0; y<h; ++y){
			if(DP[w-1][y] < minE_last){
				minE_last = DP[w-1][y];
				cut.back() = y;
			}
		}
		for(auto x=w-2; x>=0; --x) {
			CHECK_LT(cut[x+1], h);
			cut[x] = cut[x + 1] + track[x + 1][cut[x + 1]];
		}

		//compose output patch
		Mat output = b1.clone();
		uchar* pOut = output.data;
		for(auto x=0; x<w; ++x){
			for(auto y=cut[x]; y<h; ++y){
				int idx = y * w + x;
				for(auto i=0; i<3; ++i)
					pOut[idx*3+i] = pb2[idx*3+i];
			}
		}
		return output;
	}

	cv::Mat samplePatch(const cv::Mat &input, const cv::Mat& inputMap, const cv::Mat &output, const cv::Mat& guide, const cv::Mat &mask, const int cx, const int cy,
	                    const QuiltingConfig &config){
		CHECK_EQ(output.size(), mask.size());
		int R = config.blockSize / 2;
		double minDiff = numeric_limits<double>::max();

		typedef pair<double, Vector2i> PatchT;
		vector<PatchT> patches;
		patches.reserve(input.cols * input.rows);
		const uchar* pI = input.data;
		const uchar* pO = output.data;
		const float* pG = (float*) guide.data;
		const float* pIM = (float*) inputMap.data;
		//compute minimum error
		for(auto y=R; y<input.rows-R; ++y){
			for(auto x=R; x<input.cols-R; ++x){

				double kTex = 0.0, kMap = 0.0;
				double eTex = 0.0, eMap = 0.0;
				for(int dx = -1 * R; dx < R; ++dx) {
					for (int dy = -1 * R; dy < R; ++dy) {
						int idxO = (cy + dy) * output.cols + cx + dx;
						int idxI = (y + dy) * input.cols + x + dx;
						if (mask.at<uchar>(cy + dy, cx + dx) > 0) {
							Vector3d pixI(pI[idxI * 3], pI[idxI * 3 + 1], pI[idxI * 3 + 2]);
							Vector3d pixO(pO[idxO * 3], pO[idxO * 3 + 1], pO[idxO * 3 + 2]);
							Vector3d diff = pixI - pixO;
							eTex += diff.norm();
							kTex += 1.0;
						}
						eMap += (pG[idxO] - pIM[idxI]) * (pG[idxO] - pIM[idxI]);
						kMap += 1.0;
					}
				}
				CHECK_GT(kMap, 0.99);
				CHECK_GT(kTex, 0.99);

				double curDiff = (1-config.alpha) * eTex / kTex + config.alpha * eMap / kMap;
				patches.push_back(PatchT(curDiff, Vector2i(x,y)));
				if(curDiff < minDiff)
					minDiff = curDiff;
			}
		}

		//randomly pick one, whose error is less than 110% minimum error
		vector<int> patchIdx;
		patchIdx.reserve(patches.size());
		const double maxValidE = minDiff * 1.1;
		for(auto i=0; i<patches.size(); ++i){
			if(patches[i].first <= maxValidE)
				patchIdx.push_back(i);
		}

		default_random_engine generator;
		uniform_int_distribution<int> distribution(0, (int)patchIdx.size() - 1);
		int idx = distribution(generator);

		int resx = patches[patchIdx[idx]].second[0];
		int resy = patches[patchIdx[idx]].second[1];
		printf("(%d,%d): patch center: (%d,%d), error: %.3f\n", cx, cy, resx, resy, patches[patchIdx[idx]].first);
		Mat sampled = input(Rect(resx-R, resy-R, 2*R, 2*R)).clone();
		return sampled;
	}

	void expandTexture(const cv::Mat &input, const cv::Mat& guide, cv::Mat &output, cv::Mat& mask, const QuiltingConfig &config) {
		CHECK_EQ(output.cols, config.outputWidth);
		CHECK_EQ(output.rows, config.outputHeight);
		CHECK_EQ(output.size(), mask.size());
		CHECK(output.data);
		CHECK(mask.data);
		CHECK_EQ(mask.channels(), 1);

		const int R = config.blockSize / 2;
		const int &OL = config.overlap;

		//rondomly choose one patch to start
		default_random_engine generator;
		uniform_int_distribution<int> distrix(R, input.cols - R);
		uniform_int_distribution<int> distriy(R, input.rows - R);
		int ranx = distrix(generator);
		int rany = distriy(generator);

		Mat maskBlock(2*R, 2*R, CV_8UC1, Scalar(255));

		input(Range(rany - R, rany + R), Range(ranx - R, ranx + R)).copyTo(
				output(Range(0, 2 * R), Range(0, 2 * R)));
		maskBlock.copyTo(mask(Range(0, 2 * R), Range(0, 2 * R)));

		Mat inputMap;
		cvtColor(input, inputMap, CV_RGB2GRAY);
		inputMap.convertTo(inputMap, CV_32FC1);
		blur(inputMap, inputMap, Size(config.blurR, config.blurR));

		//repeat sample and cut
		for (auto y = R; y < output.rows; y += 2 * R - OL) {
			for (auto x = R; x < output.cols; x += 2 * R - OL) {
				if(x == R && y == R)
					continue;
				int xOL = OL, yOL = OL;
				if(x >= output.cols - R) {
					xOL += x - output.cols + R;
					x = output.cols - R;

				}
				if(y >= output.rows - R) {
					yOL += y - output.rows + R;
					y = output.rows - R;
				}
				printf("(%d,%d)\n", x, y);
				Mat patch = samplePatch(input, inputMap, output, guide, mask, x, y, config);
				//seam cut
				if(y > R){
					Mat b1 = output(Rect(x-R, y-R, 2*R, yOL));
					Mat b2 = patch(Rect(0,0,2*R, yOL));
					Mat b3 = seamCut(b1,b2);
					b3.copyTo(patch(Range(0,yOL), Range(0,2*R)));
				}
				if(x > R){
					Mat b1 = output(Rect(x-R, y-R, xOL, 2*R));
					Mat b2 = patch(Rect(0,0,xOL,2*R));
					cv::transpose(b1,b1);
					cv::transpose(b2,b2);
					Mat b3 = seamCut(b1,b2);
					cv::transpose(b3,patch(Range(0,2*R), Range(0,xOL)));
				}
				patch.copyTo(output(Range(y-R,y+R), Range(x-R,x+R)));
				maskBlock.copyTo(mask(Range(y-R,y+R), Range(x-R,x+R)));
				imshow("output", output);
				waitKey(10);
			}
		}
	}
}//namespace quilting