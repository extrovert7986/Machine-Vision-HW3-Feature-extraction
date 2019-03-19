#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

vector<Mat> target;
vector<KeyPoint> kpts[5], kptsframe;
Mat des[5], desframe;
const int contour = 4;
void AKAKZ_Match(Mat& img1, Mat& img2,vector<KeyPoint> kpts1,vector<KeyPoint> kpts2,Mat des1,Mat des2, vector<KeyPoint>& matched1, vector<KeyPoint>& matched2,const float rate) {
	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch>> matchePoints;

	matcher.knnMatch(des1, des2, matchePoints, 2);
	for (int i = 0; i < matchePoints.size(); i++) {
		DMatch mostSimilar = matchePoints[i][0];
		float distance1 = mostSimilar.distance;
		float distance2 = matchePoints[i][1].distance;
		if (distance1 < distance2*rate) {
			matched1.push_back(kpts1[mostSimilar.queryIdx]);
			matched2.push_back(kpts2[mostSimilar.trainIdx]);
		}
	}
}

void Draw_Rect(Mat& output,vector<KeyPoint> match1, vector<KeyPoint> match2, vector<Point2f> src_corner,const Scalar color) {
	vector<Point2f> src(match1.size()), dst(match2.size());
	for (int index = 0; index < src.size(); index++) {
		src[index] = match1[index].pt;
		dst[index] = match2[index].pt;
	}
	vector<Point2f> dst_corner(4);
	Mat H = findHomography(src, dst, CV_RANSAC);
	perspectiveTransform(src_corner, dst_corner, H);
	dst_corner[0] = dst_corner[0] + Point2f(110, 0);
	dst_corner[1] = dst_corner[1] + Point2f(110, 0);
	dst_corner[2] = dst_corner[2] + Point2f(110, 0);
	dst_corner[3] = dst_corner[3] + Point2f(110, 0);

	for (int i = 0; i < 4; i++) {
		line(output, dst_corner[i%4], dst_corner[(i + 1)%4], color, 3);
	}
	
}

int main() {
	target.push_back(imread("new_target0.png", 1));
	target.push_back(imread("new_target1.png", 1));
	target.push_back(imread("new_target2.png", 1));
	target.push_back(imread("new_target3.png", 1));
	target.push_back(imread("new_target4.png", 1));

	Ptr<AKAZE> akaze = AKAZE::create();
	
	/*open the film*/
	VideoCapture cap("new_test1.avi");

	VideoWriter videoout;
	videoout.open("output.wmv", VideoWriter::fourcc('X', 'V', 'I', 'D'), (int)cap.get(CV_CAP_PROP_FPS), Size(1070,540), true);
	if (!videoout.isOpened()) {
		cap.release();
		exit(0);
		return 1;
	}
	Mat frame;
	Mat output(Size(1070,540), CV_8UC3);
	int key = 0;
	vector<KeyPoint> match1, match2;

	for (int i = 0; i < target.size(); i++) {
		akaze->detectAndCompute(target[i], noArray(), kpts[i], des[i]);
		resize(target[i], target[i], Size(110, 90));
	}

	vector<Point2f> src_corner(4);
	src_corner[0] = Point2f(0, 0);
	src_corner[1] = Point2f(0, 361);
	src_corner[2] = Point2f(330, 361);
	src_corner[3] = Point2f(330, 0);

	for (;cap.isOpened() && tolower(key) != 'q' && key != 27;) {
		cap >> frame;
		for (int i = 0; i < 5; i++) {
			target[i].copyTo(Mat(output, Rect(0, i * 90, 110, 90)));
		}
		if (frame.empty())
			break;
		/****************************/
		frame.copyTo(Mat(output, Rect(target[0].size().width, 0, frame.size().width, frame.size().height)));
		akaze->detectAndCompute(frame, noArray(), kptsframe, desframe);
		/***************************/
		AKAKZ_Match(target[0], frame, kpts[0], kptsframe, des[0], desframe, match1, match2, 0.76);
		if (match1.size() >= 15) {
			Scalar color(255, 0, 0);
			Draw_Rect(output, match1, match2, src_corner, color);
			for (int i = 0; i < match1.size(); i++) {
				match1[i].pt = Point2f(match1[i].pt.x / 3.0, match1[i].pt.y / 4.0);
				match2[i].pt = match2[i].pt + Point2f(110.0, 0.0);
				line(output, match1[i].pt, match2[i].pt, color);
			}
			
		}
		match1.clear();
		match2.clear();
		/*****************************/
		AKAKZ_Match(target[1], frame, kpts[1], kptsframe, des[1], desframe, match1, match2, 0.62);
		if (match1.size() >= 6) {
			Scalar color(255, 255, 0);
			Draw_Rect(output, match1, match2, src_corner, color);
			for (int i = 0; i < match1.size(); i++) {
				match1[i].pt = Point2f(match1[i].pt.x / 3.0, match1[i].pt.y / 4.0 + 90.0);
				match2[i].pt = match2[i].pt + Point2f(110.0, 0.0);
				line(output, match1[i].pt, match2[i].pt, color);
			}
		}
		match1.clear();
		match2.clear();
		/******************************/
		AKAKZ_Match(target[2], frame, kpts[2], kptsframe, des[2], desframe, match1, match2, 0.65);
		if (match1.size() >= 6) {
			Scalar color(0, 255, 0);
			Draw_Rect(output, match1, match2, src_corner, color);
			for (int i = 0; i < match1.size(); i++) {
				match1[i].pt = Point2f(match1[i].pt.x / 3.0, match1[i].pt.y / 4.0 + 180.0);
				match2[i].pt = match2[i].pt + Point2f(110.0, 0.0);
				line(output, match1[i].pt, match2[i].pt, color);
			}
		}
		match1.clear();
		match2.clear();
		/*******************************/
		AKAKZ_Match(target[3], frame, kpts[3], kptsframe, des[3], desframe, match1, match2, 0.68);
		if (match1.size() >= 6) {
			Scalar color(0, 255, 255);
			Draw_Rect(output, match1, match2, src_corner, color);
			for (int i = 0; i < match1.size(); i++) {
				match1[i].pt = Point2f(match1[i].pt.x / 3.0, match1[i].pt.y / 4.0 + 270.0);
				match2[i].pt = match2[i].pt + Point2f(110.0, 0.0);
				line(output, match1[i].pt, match2[i].pt, color);
			}
		}
		match1.clear();
		match2.clear();
		/************************************/
		AKAKZ_Match(target[4], frame, kpts[4], kptsframe, des[4], desframe, match1, match2, 0.58);
		if (match1.size() >= 6) {
			Scalar color(0, 0, 255);
			Draw_Rect(output, match1, match2, src_corner, color);
			for (int i = 0; i < match1.size(); i++) {
				match1[i].pt = Point2f(match1[i].pt.x / 3.0, match1[i].pt.y / 4.0 + 360.0);
				match2[i].pt = match2[i].pt + Point2f(110.0, 0.0);
				line(output, match1[i].pt, match2[i].pt, color);
			}
		}
		match1.clear();
		match2.clear();
		/********************************/
		imshow("output", output);
		
		videoout << output;
		waitKey(33);
		
	}

	videoout.release();
	return 0;
}

