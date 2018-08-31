#include "SGM.h"
#include "opencv2/photo.hpp"

namespace sgmstereo {

SGM::SGM(const cv::Mat &imgLeftLast, const cv::Mat &img_left_, const cv::Mat &imgRight, parameter_type parameters)
: img_left_last_(imgLeftLast), img_left_(img_left_), img_right_(imgRight), parameters_(parameters) {
  
  const int width  = parameters_.width;
  const int height = parameters_.height;

  census_image_left_      = cv::Mat::zeros(height, width, CV_8UC1);
  census_image_right_     = cv::Mat::zeros(height, width, CV_8UC1);
  census_image_left_last_ = cv::Mat::zeros(height, width, CV_8UC1);
  cost_left_to_right_  = cv::Mat::zeros(height, width, CV_32FC(DISP_RANGE));
  cost_right_to_left_  = cv::Mat::zeros(height, width, CV_32FC(DISP_RANGE));
  accumulated_cost_    = cv::Mat::zeros(height, width, CV_32SC(DISP_RANGE));
};

void SGM::computeCensus(const cv::Mat &image, cv::Mat &censusImg) {
  const int width  = parameters_.width;
  const int height = parameters_.height;
  const int winRadius = parameters_.winRadius;
  
  for (int y = winRadius + 1; y < height - winRadius - 1; ++y) {
    for (int x = winRadius; x < width - winRadius; ++x) {
      unsigned char centerValue = image.at<uchar>(y, x);
      int censusCode = 0;
      for (int neiY = -winRadius - 1; neiY <= winRadius + 1; ++neiY) {
        for (int neiX = -winRadius; neiX <= winRadius; ++neiX) {
          censusCode = censusCode << 1;
          if (image.at<uchar>(y + neiY, x + neiX) >= centerValue) censusCode += 1;
        }
      }
      censusImg.at<uchar>(y, x) = static_cast<unsigned char>(censusCode);
    }
  }
};

int SGM::computeHammingDist(const uchar left, const uchar right) {
  int var = static_cast<int>(left ^ right);
  int count = 0;
  while (var) {
    var = var & (var - 1);
    count++;
  }
  return count;
};

void SGM::sumOverAllCost(cv::Mat& pathWiseCost) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      accumulated_cost_.at<vector_type>(y, x) += pathWiseCost.at<vector_type>(y, x);
    }
  }
};

void SGM::createDisparity(cv::Mat& disparity) {

  const int width  = parameters_.width;
  const int height = parameters_.height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float imax = std::numeric_limits<float>::max();
      int min_index = 0;
      vector_type vec = accumulated_cost_.at<vector_type>(y, x);

      for (int d = 0; d < DISP_RANGE; d++) {
        if (vec[d] < imax) {
          imax = vec[d];
          min_index = d;
        }
      }
      disparity.at<uchar>(y, x) = static_cast<uchar>(DIS_FACTOR * min_index);
    }
  }
};

void SGM::process(cv::Mat &disparity) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  computeCensus(img_left_    , census_image_left_);
  computeCensus(img_right_   , census_image_right_);
  computeCensus(img_left_last_, census_image_left_last_);
  computeDerivative();
  computeCost();

  aggregation<1, 0>(cost_left_to_right_);
  aggregation<0, 1>(cost_left_to_right_);
  aggregation<-1, 0>(cost_left_to_right_);
  aggregation<0, -1>(cost_left_to_right_);

  cv::Mat disparity_left(height, width, CV_8UC1);
  cv::Mat disparity_temp(height, width, CV_8UC1);
  createDisparity(disparity_temp);
  fastNlMeansDenoising(disparity_temp, disparity_left);

  accumulated_cost_.setTo(0.0);

  computeCostRight();

  aggregation<1, 0>(cost_right_to_left_);
  aggregation<0, 1>(cost_right_to_left_);
  aggregation<0, -1>(cost_right_to_left_);
  aggregation<-1, 0>(cost_right_to_left_);

  cv::Mat disparity_right(height, width, CV_8UC1);
  createDisparity(disparity_temp);
  fastNlMeansDenoising(disparity_temp, disparity_right);

  consistencyCheck(disparity_left, disparity_right, disparity, 0);
};

SGM::~SGM() {
  census_image_right_.release();
  census_image_left_.release();
  census_image_left_last_.release();
  cost_left_to_right_.release();
  cost_right_to_left_.release();
  accumulated_cost_.release();
};

SGM::vector_type SGM::addPenalty(vector_type const &priorL, vector_type &local_cost) {

  const int penalty1 = parameters_.penalty1;
  const int penalty2 = parameters_.penalty2;

  vector_type penalized_disparity;

  for (int d = 0; d < DISP_RANGE; d++) {
    float e_smooth = std::numeric_limits<float>::max();
    for (int d_p = 0; d_p < DISP_RANGE; d_p++) {
      if (d_p - d == 0) {
        //e_smooth = std::min(e_smooth,priorL[d_p]);
        e_smooth = std::min(e_smooth, priorL[d]);
      } else if (abs(d_p - d) == 1) {
        // Small penality
        e_smooth = std::min(e_smooth, priorL[d_p] + penalty1);
      } else {
        // Large penality
        e_smooth = std::min(e_smooth, priorL[d_p] + penalty2);
      }
    }
    penalized_disparity[d] = local_cost[d] + e_smooth;
  }

  double minVal;
  cv::minMaxLoc(priorL, &minVal);

  // Normalize by subtracting min of priorL cost_
  for (int i = 0; i < DISP_RANGE; i++) {
    penalized_disparity[i] -= static_cast<float>(minVal);
  }

  return penalized_disparity;
};

template<int DIRX, int DIRY>
void SGM::aggregation(cv::Mat cost) {

  const int width  = parameters_.width;
  const int height = parameters_.height;
  cv::Mat pathWiseCost = cv::Mat::zeros(height, width, CV_32SC(DISP_RANGE));
  
  if (DIRX == -1 && DIRY == 0) {
    // RIGHT MOST EDGE
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = width - 2; x >= 0; --x) {
      for (int y = 0; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  // Walk along the edges in a clockwise fashion
  if (DIRX == 1 && DIRY == 0) {
    // Process every pixel along this edge
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 1; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 0 && DIRY == 1) {
    //TOP MOST EDGE
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }
    for (int y = 1; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 0 && DIRY == -1) {
    // BOTTOM MOST EDGE
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }
    for (int y = height - 2; y >= 0; --y) {
      for (int x = 0; x < width; ++x) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 1; x < width; ++x) {
      for (int y = 1; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 1; x < width; ++x) {
      for (int y = height - 2; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 2; x >= 0; --x) {
      for (int y = 1; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 2; x >= 0; --x) {
      for (int y = height - 2; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 2 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == 2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -2 && DIRY == 1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }


  if (DIRX == -1 && DIRY == 2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(0, x) = cost.at<vector_type>(0, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = 2; y < height; ++y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 2 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == 1 && DIRY == -2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, 0) = cost.at<vector_type>(y, 0);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = 2; x < width; ++x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -2 && DIRY == -1) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  if (DIRX == -1 && DIRY == -2) {
    for (int y = 0; y < height; ++y) {
      pathWiseCost.at<vector_type>(y, width - 1) = cost.at<vector_type>(y, width - 1);
    }
    for (int x = 0; x < width; ++x) {
      pathWiseCost.at<vector_type>(height - 1, x) = cost.at<vector_type>(height - 1, x);
    }

    for (int x = width - 3; x >= 0; --x) {
      for (int y = height - 3; y >= 0; --y) {
        pathWiseCost.at<vector_type>(y, x) = addPenalty(pathWiseCost.at<vector_type>(y - DIRY, x - DIRX),
                                                     cost.at<vector_type>(y, x));
      }
    }
  }

  sumOverAllCost(pathWiseCost);
};

void SGMStereo::computeDerivative(){

  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const float sobel_cap_value = parameters_.sobel_cap_value;
  
  cv::Mat gradx(height, width, CV_32FC1);
  cv::Sobel(img_left_, gradx, CV_32FC1,1,0);
  
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobel_value = gradx.at<float>(y,x);

      if (sobel_value > sobel_cap_value) sobel_value = 2.0 * sobel_cap_value;
      else if (sobel_value < -sobel_cap_value) sobel_value = 0.0;
      else sobel_value += sobel_cap_value;

      derivativeStereoLeft_.at<float>(y,x) = sobel_value;
    }
  }

  cv::Sobel(img_right_, gradx, CV_32FC1,1,0);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float sobel_value = 1*gradx.at<float>(y,x);

      if (sobel_value > sobel_cap_value) sobel_value = 2.0 * sobel_cap_value;
      else if (sobel_value < -sobel_cap_value) sobel_value = 0.0;
      else sobel_value += sobel_cap_value;

      derivativeStereoRight_.at<float>(y, width - x -1) = sobel_value;
    }
  }
  
  gradx.release();
};

void SGMStereo::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity, bool interpl){

  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int winRadius = parameters_.winRadius;
  
  std::vector<cv::Point2i> occFalse;
  cv::Mat disparityWoConsistency(height, width, CV_8UC1);

//based on left
  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x = winRadius; x < width - winRadius; ++x){
      unsigned short disparityLeftValue =  static_cast<unsigned short>(disparityLeft.at<uchar>(y,x));
      unsigned short disparityRightValue =  static_cast<unsigned short>(disparityRight.at<uchar>(y,x - disparityLeftValue));
      disparity.at<uchar>(y,x) = static_cast<uchar>(disparityLeftValue);
      disparityWoConsistency.at<uchar>(y,x) = static_cast<uchar>(disparityLeftValue * visualDisparity);
      if(abs(disparityRightValue - disparityLeftValue) > disparityThreshold){
        disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
        occFalse.push_back(cv::Point2i(x,y));
      }
    }
  }

  std::vector<cv::Point2i> occRefine;
  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x = DISP_RANGE; x < width - winRadius - DISP_RANGE; ++x){
      unsigned short top 	= static_cast<unsigned short>(disparity.at<uchar>(y+1,x)) == Dinvd ? 1 : 0;
      unsigned short bottom 	= static_cast<unsigned short>(disparity.at<uchar>(y-1,x)) == Dinvd ? 1 : 0;
      unsigned short left 	= static_cast<unsigned short>(disparity.at<uchar>(y,x-1)) == Dinvd ? 1 : 0;
      unsigned short right	= static_cast<unsigned short>(disparity.at<uchar>(y,x+1)) == Dinvd ? 1 : 0;
      unsigned short self	= static_cast<unsigned short>(disparity.at<uchar>(y,x  )) != Dinvd ? 1 : 0;
      if(((top + bottom + left + right) >= 3) && (self == 1)){
        occRefine.push_back(cv::Point2i(x,y));
        occFalse.push_back(cv::Point2i(x,y));
      }

    }
  }

  for(int i = 0; i < occRefine.size(); i++){
    int x = occRefine[i].x;
    int y = occRefine[i].y;
    disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
  }

  if(interpl){
    const int len = occFalse.size();

    std::vector<int> newDisparity(len);

    for(int i = 0; i < len; i++){
      std::vector<int> neiborInfo;
      bool occlusion;
      int x = occFalse[i].x;
      int y = occFalse[i].y;


      {
        int xx = x + 1;
        int dirx = 0;
        int dirxDisparityPx = 0;
        while((dirx <= 10) && (xx <= width - winRadius - DISP_RANGE)){
          if(disparity.at<uchar>(y,xx) == static_cast<uchar>(Dinvd)) {xx++;continue;}
          dirxDisparityPx += static_cast<int>(disparity.at<uchar>(y,xx)); xx++; dirx++;
        }
        if(dirx != 0){neiborInfo.push_back(round(dirxDisparityPx/(float)dirx));}
      }

      {
        int xx = x - 1;
        int dirx = 0;
        int dirxDisparityNx = 0;
        while((dirx <= 10) && (xx >= winRadius + DISP_RANGE)){
          if(disparity.at<uchar>(y,xx) == static_cast<uchar>(Dinvd)) {xx--;continue;}
          dirxDisparityNx += static_cast<int>(disparity.at<uchar>(y,xx)); xx--; dirx++;
        }
        if(dirx != 0){neiborInfo.push_back(round(dirxDisparityNx/(float)dirx));}
      }

      if(neiborInfo.size() == 2){ occlusion = std::abs((neiborInfo[0]-neiborInfo[1])/std::min(neiborInfo[0], neiborInfo[1])) > 0.2 ? true : false;}
      else{occlusion = false;}

      {
        int yy = y + 1;
        int diry = 0;
        int dirxDisparityPy = 0;
        while((diry < 1) && (yy < height - winRadius)){
          if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy++;continue;}
          dirxDisparityPy = static_cast<int>(disparity.at<uchar>(yy,x)); yy++; diry++;
        }
        if(diry != 0){neiborInfo.push_back(round(dirxDisparityPy/(float)diry));}
      }

      {
        int yy = y - 1;
        int diry = 0;
        int dirxDisparityNy = 0;
        while((diry < 1) && (yy >= winRadius)){
          if(disparity.at<uchar>(yy,x) == static_cast<uchar>(Dinvd)) {yy--;continue;}
          dirxDisparityNy = static_cast<int>(disparity.at<uchar>(yy,x)); yy--; diry++;
        }
        if(diry != 0){neiborInfo.push_back(round(dirxDisparityNy/(float)diry));}

      }

      {
        int dirxy = 0;
        int yy = y + 1;
        int xx = x - 1;
        int dirxDisparityNxPy = 0;
        while((dirxy < 1) && (yy < height - winRadius) && (xx >= winRadius + DISP_RANGE)){
          if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx--;continue;}
          dirxDisparityNxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx--; dirxy++;
        }
        if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxPy/(float)dirxy));}
      }

      {
        int dirxy = 0;
        int yy = y + 1;
        int xx = x + 1;
        int dirxDisparityPxPy = 0;
        while((dirxy < 1) && (yy < height - winRadius) && (xx < width - winRadius - DISP_RANGE)){
          if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy++; xx++;continue;}
          dirxDisparityPxPy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy++; xx++; dirxy++;
        }
        if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxPy/(float)dirxy));}
      }

      {
        int dirxy = 0;
        int yy = y - 1;
        int xx = x + 1;
        int dirxDisparityPxNy = 0;
        while((dirxy < 1) && (yy >= winRadius) && (xx < width - winRadius - DISP_RANGE)){
          if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx++;continue;}
          dirxDisparityPxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx++; dirxy++;
        }
        if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityPxNy/(float)dirxy));}
      }

      {
        int dirxy = 0;
        int yy = y - 1;
        int xx = x - 1;
        int dirxDisparityNxNy = 0;
        while((dirxy < 1) && (yy >= winRadius) && (xx >= winRadius + DISP_RANGE)){
          if(disparity.at<uchar>(yy,xx) == static_cast<uchar>(Dinvd)) {yy--; xx--;continue;}
          dirxDisparityNxNy = static_cast<int>(disparity.at<uchar>(yy,xx)); yy--; xx--; dirxy++;
        }
        if(dirxy != 0){neiborInfo.push_back(round(dirxDisparityNxNy/(float)dirxy));}
      }


      std::sort(neiborInfo.begin(), neiborInfo.end());


      int secLow = neiborInfo[1];
      int median = neiborInfo[floor(neiborInfo.size()/2.f)];


      unsigned short newValue = 0;
      if(occlusion == true){
        newValue = secLow;
      }else{
        newValue = median;
      }

      newDisparity[i] = newValue ;

    }

    for(int i = 0; i < len; i++){
      int x = occFalse[i].x;
      int y = occFalse[i].y;
      disparity.at<uchar>(y,x) = static_cast<uchar>(newDisparity[i]);

    }
  }

};

void SGMStereo::computeCostRight(){
  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int winRadius = parameters_.winRadius;
  
  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x = winRadius; x < width - DISP_RANGE - 1; ++x){
      for(int d = 0; d < DISP_RANGE; d++){

        for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
          for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
            cost_right_to_left_.at<vector_type>(y,x)[d] +=
                          fabs(derivativeStereoRight_.at<float>(neiY, neiX)- derivativeStereoLeft_.at<float>(neiY, neiX + d))
                          + CENSUS_W * computeHammingDist(census_image_right_.at<uchar>(neiY, neiX), census_image_left_.at<uchar>(neiY, neiX + d));
          }

        }
      }

    }
  }

  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x =  width - DISP_RANGE - 1; x < width - winRadius; ++x){
      int end = width - winRadius - 1 - x;
      for(int d = 0; d < end; d++){
        for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
          for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
            cost_right_to_left_.at<vector_type>(y,x)[d] +=
                            fabs(derivativeStereoRight_.at<float>(neiY, neiX)- derivativeStereoLeft_.at<float>(neiY, neiX + d))
                            + CENSUS_W * computeHammingDist(census_image_right_.at<uchar>(neiY, neiX), census_image_left_.at<uchar>(neiY, neiX + d));

          }
        }
      }
      float val = cost_right_to_left_.at<vector_type>(y,x)[end - 1];
      for(int d = end; d < DISP_RANGE; d++){
        cost_right_to_left_.at<vector_type>(y,x)[d] = val;
      }
    }
  }

};

void SGMStereo::computeCost(){
  const int width     = parameters_.width;
  const int height    = parameters_.height;
  const int winRadius = parameters_.winRadius;
  
  //pixel intensity matching
  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x = DISP_RANGE; x < width - winRadius; ++x){
      for(int d = 0; d < DISP_RANGE; d++){
        for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
          for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
            cost_left_to_right_.at<vector_type>(y,x)[d] +=
                              fabs(derivativeStereoLeft_.at<float>(neiY, neiX)- derivativeStereoRight_.at<float>(neiY, neiX - d))
                              + CENSUS_W * computeHammingDist(census_image_left_.at<uchar>(neiY, neiX), census_image_right_.at<uchar>(neiY, neiX - d));

          }

        }
      }

    }
  }

  //for x -> disparity
  for(int y = winRadius; y < height - winRadius; ++y){
    for(int x = winRadius; x < DISP_RANGE; ++x){
      for(int d = 0; d < x; d++){
        for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
          for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
            cost_left_to_right_.at<vector_type>(y,x)[d] +=
                              fabs(derivativeStereoLeft_.at<float>(neiY, neiX)- derivativeStereoRight_.at<float>(neiY, neiX - d))
                              + CENSUS_W * computeHammingDist(census_image_left_.at<uchar>(neiY, neiX), census_image_right_.at<uchar>(neiY, neiX - d));

          }
        }
      }
      float val = cost_left_to_right_.at<vector_type>(y,x)[x - 1];
      for(int d = x; d < DISP_RANGE; d++){
        cost_left_to_right_.at<vector_type>(y,x)[d] = val;
      }
    }
  }

};

SGMStereo::SGMStereo(const cv::Mat &imgLeftLast, const cv::Mat &imgLeft, const cv::Mat &imgRight, parameter_type parameters)
:SGM(imgLeftLast, imgLeft, imgRight, parameters){
  const int width  = parameters_.width;
  const int height = parameters_.height;
  
  derivativeStereoLeft_  = cv::Mat::zeros(height, width, CV_32FC1);
  derivativeStereoRight_ = cv::Mat::zeros(height, width, CV_32FC1);
};

SGMStereo::~SGMStereo(){
  derivativeStereoLeft_.release();
  derivativeStereoRight_.release();
};
}