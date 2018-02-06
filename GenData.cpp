
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<filesystem>
#include<iostream>
#include<vector>
namespace fs = std::experimental::filesystem;


const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


int main() {

	cv::Mat imgTrainingNumbers;         // input image
	cv::Mat imgGrayscale;               // 
	cv::Mat imgBlurred;                 // declare various images
	cv::Mat imgThresh;                  //
	cv::Mat imgThreshCopy;              //

	std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
	std::vector<cv::Vec4i> v4iHierarchy;                    // declare contours hierarchy

	cv::Mat matClassificationInts;      // these are our training classifications, note we will have to perform some conversions before writing to file later

										// these are our training images, due to the data types that the KNN object KNearest requires, we have to declare a single Mat,
										// then append to it as though it's a vector, also we will have to perform some conversions before writing to file later
	cv::Mat matTrainingImagesAsFlattenedFloats;


	std::string path = "dataset/";
	fs::path temp;
	std::string finalpath;
	for (auto & p : fs::directory_iterator(path))
	{
		temp = p.path();
		finalpath = temp.string();
		imgTrainingNumbers = cv::imread(finalpath);         // read in training numbers image


		if (imgTrainingNumbers.empty()) {                               // if unable to open image
			std::cout << "error: image not read from file\n\n";         // show error message on command line
			return(0);                                                  // and exit program
		}

		cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);        // convert to grayscale

		cv::GaussianBlur(imgGrayscale,              // input image
			imgBlurred,                             // output image
			cv::Size(5, 5),                         // smoothing window width and height in pixels
			0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

													// filter image from grayscale to black and white
		cv::adaptiveThreshold(imgBlurred,           // input image
			imgThresh,                              // output image
			255,                                    // make pixels that pass the threshold full white
			cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
			cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
			11,                                     // size of a pixel neighborhood used to calculate threshold value
			2);                                     // constant subtracted from the mean or weighted mean

		// cv::imshow("imgThresh", imgThresh);         // show threshold image for reference

		imgThreshCopy = imgThresh.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the image

		cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
			ptContours,                             // output contours
			v4iHierarchy,                           // output hierarchy
			cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
			cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

		for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
			if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // if contour is big enough to consider
				cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

				cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      // draw red rectangle around each contour as we ask user for input

				cv::Mat matROI = imgThresh(boundingRect);           // get ROI image of bounding rect

				cv::Mat matROIResized;
				cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

				//cv::imshow("matROI", matROI);                               // show ROI image for reference
				//cv::imshow("matROIResized", matROIResized);                 // show resized ROI image for reference
				//cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // show training numbers image, this will now have red rectangles drawn on it

				int intChar = finalpath[8];           // get key press


				matClassificationInts.push_back(intChar);       // append classification char to integer list of chars

				cv::Mat matImageFloat;                          // now add the training image (some conversion is necessary first) . . .
				matROIResized.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

				cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // flatten

				matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // add to Mat as though it was a vector, this is necessary due to the
																							// data types that KNearest.train accepts
			}   // end if
		}   // end if
	}

	std::cout << "training complete\n\n";

	// save classifications to file ///////////////////////////////////////////////////////

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           // open the classifications file

	if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
		std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
		return(0);                                                                                      // and exit program
	}

	fsClassifications << "classifications" << matClassificationInts;        // write classifications into classifications section of classifications file
	fsClassifications.release();                                            // close the classifications file

																			// save training images to file ///////////////////////////////////////////////////////

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // open the training images file

	if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
		std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
		return(0);                                                                              // and exit program
	}

	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images section of images file
	fsTrainingImages.release();                                                 // close the training images file

	return(0);
}




