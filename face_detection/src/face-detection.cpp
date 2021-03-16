#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
  cv::Mat image;

  cv::namedWindow("Face detection", cv::WINDOW_AUTOSIZE);
  // Loading the cascade classifier trained  on the data from the .xml file
  cv::CascadeClassifier face_cascade = cv::CascadeClassifier("../haarcascade_frontalface_default.xml");
  // Reading the input image
  image = cv::imread("../eu.jpg");
  cv::imshow("Face detection", image);
  return 0;
}
