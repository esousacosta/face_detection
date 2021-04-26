/* The landmark detection and drawing parts of this code are inspired by a very similar code developed by anishakd4. */
/* The Delaunay Triangulation and the face copying portions of the code were, mostly, taken from
 Learnopencv's "Face swap" (https://github.com/spmallick/learnopencv/tree/master/FaceSwap) repository" */

#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
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
#include <vector>

std::vector<cv::Point2f> drawFaceLandmarks(cv::Mat &image, dlib::full_object_detection faceLandmark)
{
  std::vector<cv::Point2f> points;
  //Loop over all face landmarks
  for (int i=0; i< faceLandmark.num_parts(); i++) {
	int x = faceLandmark.part(i).x();
	int y = faceLandmark.part(i).y();
	std::string text = std::to_string(i+1);
	points.push_back(cv::Point2f(x, y));
	//Draw a small circle at face landmark over the image using opencv
	circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA );
	//Draw text at face landmark to show index of current face landmark over the image using opencv
	putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
  }
  return points;
}

// Here the Delaunay triangles are calculated for the given set of points, and the indexes for the 3 points of each
// triangle are stored in the delaunayTri argument.
static void calculateDelaunayTriangles(cv::Rect rectangle, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri)
{
  cv::Subdiv2D subdivs(rectangle);

  // Inserting each point in points inside the subdiv object
  for (auto& element: points) {
	subdivs.insert(element);
	std::vector<cv::Vec6f> triangleList;
	subdivs.getTriangleList(triangleList);
	std::vector<cv::Point2f> pt(3);
	std::vector<int> ind(3);

	for( int i = 0; i < triangleList.size(); i++ ) {
	  cv::Vec6f t = triangleList[i];
	  pt[0] = cv::Point2f(t[0], t[1]);
	  pt[1] = cv::Point2f(t[2], t[3]);
	  pt[2] = cv::Point2f(t[4], t[5 ]);

	  if (rectangle.contains(pt[0]) && rectangle.contains(pt[1]) && rectangle.contains(pt[2])) {
		for(int j = 0; j < 3; j++)
		  for(int k = 0; k < points.size(); k++)
			if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)						
			  ind[j] = k;					

		// The indexes for each triangle are then pushed into the delaunayTri(angles) vector
		// to be used during the mask creating phase.
		delaunayTri.push_back(ind);
	  }

	}

  }

}


int main(int argc, char *argv[])
{
  cv::Mat image, gray_image;
  int k = 0;
  std::vector<cv::Rect> faces;
  std::vector<cv::Mat> aux_matrix(2);
  std::vector<dlib::full_object_detection> facelandmarks;
  std::vector< std::vector<cv::Point2f> > points;
  std::vector< std::vector<cv::Point2f> > hulls;
  std::vector<cv::Point2f> hull;
  std::vector< std::vector<int> > dt;

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
	static cv::Mat copied_image;
	cap.read(image);
	copied_image = image.clone();

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

	for(int i=0; i<faces.size(); i++){

	  //Get the face landmark and print number of landmarks detected
	  dlib::full_object_detection facelandmark = faceLandmarkDetector(dlibImage, faces[i]);
	  std::cout<< "Number of face landmarks detected:" << facelandmark.num_parts() << std::endl;

	  //Push face landmark to array of All face's landmarks array
	  facelandmarks.push_back(facelandmark);

	  //Draw landmarks on image
	  points.push_back(drawFaceLandmarks(image, facelandmark));

	  //Write face Landmarks to a file on disk to analyze
	  std::string landmarksFilename = "face_landmarks_" + std::to_string(i+1) + ".txt";
	}

	if (faces.size() == 2) {
	  for (auto element: points) {
		cv::convexHull(element, hull);
		hulls.push_back(hull);
	  }

	  cv::Rect rect(0, 0, copied_image.cols, copied_image.rows);
	  calculateDelaunayTriangles(rect, hulls[0], dt);
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
	hulls.clear();
	points.clear();
	dt.clear();
	// cv::imshow("Teste", aux_matrix_1);
	k = cv::waitKey(10);

	if (k == 27)
	  break;
  }

  cap.release();
  return 0;
}
