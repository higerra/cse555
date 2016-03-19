//
// Created by Yan Hang on 3/19/16.
//
#include "quilting.h"
#include <iostream>
#include <gflags/gflags.h>

using namespace std;
using namespace quilting;

DEFINE_string(source, "", "Path to input source texture");
DEFINE_string(guide, "", "Path to guide image");
DEFINE_int32(blockSize, 32, "block size");
DEFINE_int32(overlap, 6, "overlap");
DEFINE_int32(outWidth, 1920, "width of output image");
DEFINE_int32(outHeight, 1080, "height of output image");

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv,true);
	CHECK(!FLAGS_source.empty()) << "Path to input source cannot be empty!";



	return 0;
}

