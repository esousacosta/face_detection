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


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform(srcTri, dstTri);
    
    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

	std::cout << "Passei da função applyAffineTransform..." << std::endl;
}


// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2)
{
	
  cv::Rect r1 = boundingRect(t1);
  cv::Rect r2 = boundingRect(t2);
	
  // Offset points by left top corner of the respective rectangles
  std::vector<cv::Point2f> t1Rect, t2Rect;
  std::vector<cv::Point> t2RectInt;
  for(int i = 0; i < 3; i++)
	{

	  t1Rect.push_back( cv::Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
	  t2Rect.push_back( cv::Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
	  t2RectInt.push_back( cv::Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

	}
	
  // Get mask by filling triangle
  cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
  fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
	
  // Apply warpImage to small rectangular patches
  // Here I'm taking the small rectangle delimited by r1 and copying its content (on the original image)
  // to a small rectangle that will be "patched" onto the destination image.
  cv::Mat img1Rect;
  img1(r1).copyTo(img1Rect);
	
  cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
	
  // Here I'm applying the affine transform to path the piece
  // of the original image (contained in img1Rect) onto the destination
  // rectangle (img2Rect).
  applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
	
  // Here I'm finally patching the second image by adding the content from the
  // original one. This is done for each of the delaunay triangle's regions.

  std::cout << "Passei 1!" << std::endl;
  multiply(img2Rect, mask, img2Rect);
  std::cout << "Passei 2!" << std::endl;
  multiply(img2(r2), cv::Scalar(1.0,1.0,1.0) - mask, img2(r2));
  std::cout << "Passei 3!" << std::endl;
  img2(r2) = img2(r2) + img2Rect;
  std::cout << "Passei 4!" << std::endl;
	
}


int main(int argc, char *argv[])
{
  static cv::Mat image, gray_image;
  int k = 0;
  std::vector<cv::Rect> faces;
  std::vector<cv::Mat> aux_matrix(2);
  std::vector<dlib::full_object_detection> facelandmarks;
  std::vector< std::vector<cv::Point2f> > points;
  std::vector< std::vector<cv::Point2f> > hulls;
  std::vector<cv::Point2f> hull;
  std::vector< std::vector<int> > dt;
  std::vector<cv::Point> hull8U;

  static cv::Mat copied_image;
  static cv::Mat destination_image;
  
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
	cap.read(image);
	image.copyTo(copied_image);//cv::Mat::zeros(copied_image.rows, copied_image.cols, CV_32F);//image.clone();
	// destination_image = cv::Mat::zeros(copied_image.rows, copied_image.cols, CV_32F);//image.clone();

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


	std::cout << "Passei do desenho das máscaras..." << std::endl;
	
	if (faces.size() == 2) {
	  image.convertTo(image, CV_32F);
	  copied_image.convertTo(copied_image, CV_32F);

	  for (auto element: points) {
		cv::convexHull(element, hull);
		hulls.push_back(hull);
	  }

	  std::cout << "Passei do cálculo dos hulls..." << std::endl;

	  cv::Rect rect(0, 0, copied_image.cols, copied_image.rows);
	  calculateDelaunayTriangles(rect, hulls[0], dt);

	  std::cout << "Passei do cálculo dos triangulos de delaunay..." << std::endl;

	  // Now we need to apply affine transformation to delaunay triangles in order
	  // to swap the correct points between the two faces.

	  for (int i = 0; i < dt.size(); i++) {

		std::vector<cv::Point2f> t1, t2;

		for (int j = 0; j < 3; j++) {

		  t1.push_back(hulls[0][dt[i][j]]);
		  t2.push_back(hulls[1][dt[i][j]]);
		  
		}
		// After finishing to collect the equivalente points of both contours,
		// it's time to apply the warping to the copy of the second image.
		warpTriangle(image, copied_image, t1, t2);
	  }

	  std::cout << "Passei do cálculo da transformada afim e do warpTriangle..." << std::endl;

	  // Calculate mask
	  for(int i = 0; i < hulls[0].size(); i++)
		{
		  cv::Point pt(hulls[0][i].x, hulls[0][i].y);
		  hull8U.push_back(pt);
		}

	  std::cout << "Já calculei o novo hull em 8U" << std::endl;
	  cv::Mat mask = cv::Mat::zeros(copied_image.rows, copied_image.cols, CV_32F);
	  fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255,255,255));

	  std::cout << "Quase clonando..." << std::endl;
	  // Clone seamlessly.
	  cv::Rect r = boundingRect(hulls[0]);
	  cv::Point center = (r.tl() + r.br()) / 2;

	  std::cout << "Clonei!" << std::endl;
	
	  cv::Mat output;
	  copied_image.convertTo(copied_image, CV_8UC3);
	  std::cout << "Converti!" << std::endl;
	  std::cout << CV_VERSION << std::endl;

	  // cv::imshow("clonned", copied_image);

	  // cv::seamlessClone(copied_image, destination_image, mask, center, output, cv::NORMAL_CLONE);
	  std::cout << "Seamless cloning feito!" << std::endl;

	  // cv::imshow("clonned", output);
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

	cv::imshow("Face detection", copied_image);
	hulls.clear();
	points.clear();
	dt.clear();
	hull.clear();
	hull8U.clear();
	faces.clear();
	facelandmarks.clear();
	// cv::imshow("Teste", aux_matrix_1);
	k = cv::waitKey(10);

	if (k == 27)
	  break;
  }

  cap.release();
  return 0;
}
