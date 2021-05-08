/* The landmark detection and drawing parts of this code are inspired by a very similar code developed by anishakd4. */
/* The Delaunay Triangulation and the face copying portions of the code were, mostly, taken from

 Learnopencv's "Face swap" (https://github.com/spmallick/learnopencv/tree/master/FaceSwap) repository" */

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
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

	//Draw a small circle at face landmark over the image using opencv. Uncommenting this will result
	// in the apparition of small circles on the patched face.
	/* circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA ); */
	//Draw text at face landmark to show index of current face landmark over the image using opencv
	/* putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1); */
  }
  return points;
}


// Here the Delaunay triangles are calculated for the given set of points, and the indexes for the 3 points of each
// triangle are stored in the delaunayTri argument.
static void calculateDelaunayTriangles(cv::Rect rectangle, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri)
{
  cv::Subdiv2D subdivs(rectangle);
  std::vector<cv::Vec6f> triangleList;
  std::vector<cv::Point2f> pt(3);
  std::vector<int> ind(3);
  cv::Vec6f t;

  // Inserting each point in points inside the subdiv object
  for (auto element: points) {
	subdivs.insert(element);
	subdivs.getTriangleList(triangleList);

	for( int i = 0; i < triangleList.size(); i++ ) {
	  t = triangleList[i];
	  pt[0] = cv::Point2f(t[0], t[1]);
	  pt[1] = cv::Point2f(t[2], t[3]);
	  pt[2] = cv::Point2f(t[4], t[5]);

	  if (rectangle.contains(pt[0]) && rectangle.contains(pt[1]) && rectangle.contains(pt[2])) {
		for(int j = 0; j < 3; j++)
		  for(int k = 0; k < points.size(); k++)
			if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)	{		  
			  ind[j] = k;
			}

		// The indexes for each triangle are then pushed into the delaunayTri(angles) vector
		// to be used during the mask creating phase.
		delaunayTri.push_back(ind);
	  }
	}

	triangleList.clear();
  }
}


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform(srcTri, dstTri);
    
    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

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
  multiply(img2Rect, mask, img2Rect);
  multiply(img2(r2), cv::Scalar(1.0,1.0,1.0) - mask, img2(r2));
  img2(r2) = img2(r2) + img2Rect;
	
  t1Rect.clear();
  t2Rect.clear();
  t2RectInt.clear();
}


int main(int argc, char *argv[])
{
  static cv::Mat image, gray_image;
  int k = 0;
  static std::vector<dlib::rectangle> faces;
  static std::vector<cv::Mat> aux_matrix(2);
  static std::vector<dlib::full_object_detection> facelandmarks;
  static std::vector< std::vector<cv::Point2f> > points;
  static std::vector< std::vector<cv::Point2f> > hulls;
  static std::vector<cv::Point2f> hull;
  static std::vector< std::vector<int> > dt;
  static std::vector<cv::Point> hull8U;
  static std::vector<cv::Point2f> t1, t2;
  static std::vector<cv::Point2f> point1, point2;
  cv::Mat mask;

  static cv::Mat copied_image;
  static cv::Mat destination_image;
  
  //Load the dlib face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  //Load the dlib face landmark detector and initialize the shape predictor

  dlib::shape_predictor faceLandmarkDetector ;


  dlib::deserialize("../../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> faceLandmarkDetector;

  // Reading the input image
  cv::VideoCapture cap ("/dev/video0");

  while (true) {
	cap.read(image);
	image.copyTo(copied_image);//cv::Mat::zeros(copied_image.rows, copied_image.cols, CV_32F);//image.clone();
	image.copyTo(destination_image);
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
    // std::vector<dlib::rectangle> faces = faceDetector(dlibImage);
	faces = faceDetector(dlibImage);
	std::cout << "Number of faces detected:" << faces.size() << std::endl;

	//Get Face landmarks of all detected faces

	for(int i=0; i<faces.size(); i++){

	  //Get the face landmark and print number of landmarks detected
	  dlib::full_object_detection facelandmark = faceLandmarkDetector(dlibImage, faces[i]);

	  //Push face landmark to array of All face's landmarks array
	  facelandmarks.push_back(facelandmark);

	  //Draw landmarks on image
	  points.push_back(drawFaceLandmarks(image, facelandmark));

	  //Write face Landmarks to a file on disk to analyze
	  std::string landmarksFilename = "face_landmarks_" + std::to_string(i+1) + ".txt";
	}


	
	if (faces.size() == 2) {
	  image.convertTo(image, CV_32F);
	  copied_image.convertTo(copied_image, CV_32F);
	  // destination_image.convertTo(destination_image, CV_32F);

	  std::vector<int> hull_index;
	  // This for is not necessary
	  for (auto element: points) {
		cv::convexHull(element, hull_index, false, false);
	  }

	  for (int i = 0; i < hull_index.size(); ++i) {
		point1.push_back(points[0][hull_index[i]]);
		point2.push_back(points[1][hull_index[i]]);
	  }

	  hulls.push_back(point1);
	  hulls.push_back(point2);

	  cv::Rect rect(0, 0, copied_image.cols, copied_image.rows);
	  calculateDelaunayTriangles(rect, hulls[1], dt);


	  // Now we need to apply affine transformation to delaunay triangles in order
	  // to swap the correct points between the two faces.

	  for (int i = 0; i < dt.size(); i++) {

		for (int j = 0; j < 3; j++) {

		  t1.push_back(hulls[0][dt[i][j]]);
		  t2.push_back(hulls[1][dt[i][j]]);
		  
		}
		// After finishing to collect the equivalente points of both contours,
		// it's time to apply the warping to the copy of the second image.
		warpTriangle(image, copied_image, t1, t2);
		t1.clear();
		t2.clear();
	  }



	  // Calculate mask
	  for(int i = 0; i < hulls[1].size(); i++)
		{
		  cv::Point pt(hulls[1][i].x, hulls[1][i].y);
		  hull8U.push_back(pt);
		}

	  mask = cv::Mat::zeros(destination_image.rows, destination_image.cols, CV_8UC3);
	  fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255,255,255));

	  // Clone seamlessly.
	  cv::Rect r = boundingRect(hulls[1]);
	  cv::Point center = (r.tl() + r.br()) / 2;

	
	  copied_image.convertTo(copied_image, CV_8UC3);

	  // cv::imshow("mask", mask);

	  cv::Mat output;
	  cv::seamlessClone(copied_image, destination_image, mask, center, output, cv::NORMAL_CLONE);

	  cv::imshow("blended", output);
	  /* cv::imshow("raw-copy", copied_image); */
	}

	cv::imshow("landmarks", image);
	point1.clear();
	point2.clear();
	hulls.clear();
	points.clear();
	dt.clear();
	hull.clear();
	hull8U.clear();
	faces.clear();
	facelandmarks.clear();
	aux_matrix.clear();
	k = cv::waitKey(10);

	if (k == 27)
	  break;
  }

  cap.release();
  return 0;
}
