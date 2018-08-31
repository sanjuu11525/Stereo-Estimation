#include "SGM.h"

int main(int argc, char *argv[]){

  cv::Mat imageLeft, imageRight, imageLeftLast;
	 cv::Mat grayLeft, grayRight, grayLeftLast;
	 imageLeft = cv::imread("data_stereo_flow_2012/training/colored_0/000013_10.png",CV_LOAD_IMAGE_COLOR);
	 imageRight = cv::imread("data_stereo_flow_2012/training/colored_1/000013_10.png",CV_LOAD_IMAGE_COLOR);
	 imageLeftLast = cv::imread("data_stereo_flow_2012/training/colored_0/000013_10.png",CV_LOAD_IMAGE_COLOR);
	 cv::cvtColor(imageLeft,grayLeft,CV_BGR2GRAY);
	 cv::cvtColor(imageRight,grayRight,CV_BGR2GRAY);
	 cv::cvtColor(imageLeftLast,grayLeftLast,CV_BGR2GRAY);

  sgmstereo::SGMStereoParameters parameter;
  parameter.width  = imageLeft.cols;
  parameter.height = imageLeft.rows;

  //-- Compute the stereo part
	 cv::Mat disparity(grayLeft.rows, grayLeft.cols, CV_8UC1);
  sgmstereo::SGMStereo sgm(grayLeftLast, grayLeft, grayRight, parameter);
  sgm.process(disparity);
	 imwrite("./disparity.jpg", disparity);

	 cv::waitKey(0);
	
	 return 0;
}





