#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <stdexcept>

void drawFaceLandmarks(cv::Mat &image, dlib::full_object_detection faceLandmark){

    //Loop over all face landmarks
    for(int i=0; i< faceLandmark.num_parts(); i++){
        int x = faceLandmark.part(i).x();
        int y = faceLandmark.part(i).y();
		std::string text = std::to_string(i+1);

        //Draw a small circle at face landmark over the image using opencv
        circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA );

        //Draw text at face landmark to show index of current face landmark over the image using opencv
        putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
    }
}

int main(int argc, char *argv[])
{
  cv::Mat image, gray_image;
  int k = 0;
  std::vector<cv::Rect> faces;
  std::vector<cv::Mat> aux_matrix(2);

  for (cv::Mat& aux : aux_matrix)
	aux = cv::Mat::zeros(250, 250, CV_8UC3);

  cv::namedWindow("Face detection", cv::WINDOW_AUTOSIZE);
  //Load the dlib face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  //Load the dlib face landmark detector and initialize the shape predictor

  dlib::shape_predictor faceLandmarkDetector ;


  dlib::deserialize("../../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> faceLandmarkDetector;

  // // Loading the cascade classifier trained  on the data from the .xml file
  // cv::CascadeClassifier face_cascade = cv::CascadeClassifier("/home/bacamartes/Documents/UFRN/2020_2/PDI/projeto_final/face_detection/haarcascade_frontalface_default.xml");

  // Reading the input image
  cv::VideoCapture cap ("/dev/video0");

  while (true) {
	static cv::Mat image;
	cap.read(image);

	// Checking if anything was captured by the camera
	if (!image.data)
	  std::cerr << "Couldn't open the requested image. Exiting!" << std::endl;

	// Convert the image from the openCV format to dlib format

	IplImage ipl_img = cvIplImage(image);
	dlib::cv_image<dlib::bgr_pixel> dlibImage(&ipl_img);
	// dlib::cv_image<dlib::bgr_pixel> dlibImage(&image);
	// cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

	//Detect faces in the image and print the number of faces detected
    std::vector<dlib::rectangle> faces = faceDetector(dlibImage);
	std::cout << "Number of faces detected:" << faces.size() << std::endl;

	//Get Face landmarks of all detected faces
	std::vector<dlib::full_object_detection> facelandmarks;
	for(int i=0; i<faces.size(); i++){

	  //Get the face landmark and print number of landmarks detected
	  dlib::full_object_detection facelandmark = faceLandmarkDetector(dlibImage, faces[i]);
	  std::cout<< "Number of face landmarks detected:" << facelandmark.num_parts() << std::endl;

	  //Push face landmark to array of All face's landmarks array
	  facelandmarks.push_back(facelandmark);

	  //Draw landmarks on image
	  drawFaceLandmarks(image, facelandmark);

	  //Write face Landmarks to a file on disk to analyze
	  std::string landmarksFilename = "face_landmarks_" + std::to_string(i+1) + ".txt";
	}

  
	/* This whole commented region below was part of the original haarscascade algorithm without using dlib
	//	
	// Detecting the faces on the grayscale image
	// face_cascade.detectMultiScale(gray_image, faces, 1.1, 4);

	// for (int i = 0; i < faces.size(); i++) {
	//   cv::rectangle(image, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(255, 0, 0), 2);
	//   if (i < 2) {
	// 	aux_matrix[i] = image({faces[i].y, faces[i].y + faces[i].height}, {faces[i].x, faces[i].x + faces[i].width});
	//   }
	// }

	// if (faces.size() == 2) {
	//   image({faces[1].y, faces[1].y + faces[1].height}, {faces[1].x, faces[1].x + faces[1].width}) = aux_matrix[2];
	//   image({faces[2].y, faces[2].y + faces[2].height}, {faces[2].x, faces[2].x + faces[2].width}) = aux_matrix[1];
	// }
	*/

	cv::imshow("Face detection", image);
	// cv::imshow("Teste", aux_matrix_1);
	k = cv::waitKey(10);

	if (k == 27)
	  break;
  }

  cap.release();
  return 0;
}
