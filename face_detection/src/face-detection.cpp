#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
  cv::Mat image, gray;

  cv::namedWindow("Face detection", cv::WINDOW_AUTOSIZE);
  // Loading the cascade classifier trained  on the data from the .xml file
  cv::CascadeClassifier face_cascade = cv::CascadeClassifier("../haarcascade_frontalface_default.xml");
  // Reading the input image
  image = cv::imread("/home/bacamartes/Documents/UFRN/2020_2/PDI/projeto_final/face_detection/eu.jpg");
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  if (!image.data)
	std::cerr << "Couldn't open the requested image. Exiting!" << std::endl;

  cv::imshow("Face detection", image);
  cv::waitKey();
  return 0;
}
