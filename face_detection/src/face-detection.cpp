#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
  // Loading the cascade classifier trained  on the data from the .xml file
  cv::CascadeClassifier face_cascade = cv::CascadeClassifier("../haarcascade_frontalface_default.xml");
  // Reading the input image
  
  return 0;
}
