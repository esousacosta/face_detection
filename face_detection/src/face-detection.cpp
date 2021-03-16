#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

int main(int argc, char *argv[])
{
  cv::Mat image, gray_image;
  std::vector<cv::Rect> faces;

  cv::namedWindow("Face detection", cv::WINDOW_AUTOSIZE);
  // Loading the cascade classifier trained  on the data from the .xml file
  cv::CascadeClassifier face_cascade = cv::CascadeClassifier("/home/bacamartes/Documents/UFRN/2020_2/PDI/projeto_final/face_detection/haarcascade_frontalface_default.xml");
  // Reading the input image
  image = cv::imread("/home/bacamartes/Documents/UFRN/2020_2/PDI/projeto_final/face_detection/eu.jpg");
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

  if (!image.data)
	std::cerr << "Couldn't open the requested image. Exiting!" << std::endl;

  face_cascade.detectMultiScale(gray_image, faces, 1.1, 4);

  for (int i = 0; i < faces.size(); i++)
	cv::rectangle(image, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(255, 0, 0), 2);

  cv::imshow("Face detection", image);
  cv::waitKey();
  return 0;
}
