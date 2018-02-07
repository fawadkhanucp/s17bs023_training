#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<filesystem>
#include<iostream>
#include<vector>

namespace fs = std::experimental::filesystem;

// Minimum size of the contour area for the identified contour to be considered a character.
const int MIN_CONTOUR_AREA = 100;

// Resized size of the identified characters before being stored.
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

int main() {

	// Original image and all the different images created.
	cv::Mat imgTrainingNumbers;
	cv::Mat imgGrayscale;
	cv::Mat imgBlurred;
	cv::Mat imgThresh;
	cv::Mat imgThreshCopy;

	//Decleration of vectors for contours.
	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	// Mats to store classifications and images for use in KNN.
	cv::Mat matClassificationInts;
	cv::Mat matTrainingImagesAsFlattenedFloats;

	// Variables used to access the dataset images for learning.
	std::string path = "dataset/";
	std::string finalpath;

	// Loop iterates through all the files in the dataset folder.
	for (auto & p : fs::directory_iterator(path))
	{
		finalpath = p.path().string();
		imgTrainingNumbers = cv::imread(finalpath);

		if (imgTrainingNumbers.empty()) {                         
			std::cout << "Training images not found.";     
			return(0);                 
		}

		// Image converted to grayscale.
		cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);

		// Gaussian blur to smooth out the image and remove any noise.
		cv::GaussianBlur(imgGrayscale, imgBlurred, cv::Size(5, 5), 0);                                 

		// Adaptive threshold to convert grayscale image to black and white image.
		cv::adaptiveThreshold(imgBlurred, imgThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);                                    

		// Create a copy of the previous image as the findContours step modifies the original image.
		imgThreshCopy = imgThresh.clone();          

		// Finds contours in the image to recognize all the characters.
		cv::findContours(imgThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// For loop that goes through every single character in the image and stores it in a matrix along with it's ASCII value.
		for (int i = 0; i < ptContours.size(); i++) { 
			if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) { 
				cv::Rect boundingRect = cv::boundingRect(ptContours[i]);   

				// Gets the region of interest from the selected contours that we think are alphabets/numbers.
				cv::Mat matROI = imgThresh(boundingRect);

				cv::Mat matROIResized;

				//Characters are resized so that they all remain the same width and height when stored as images.
				cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

				// Saves the classification as the name of the file. E.G All A's are stored under A.jpg so all classifications will be saved as A.
				int intChar = finalpath[8];

				// Adds the name from the previous step in to the classifications mat to save later on.
				matClassificationInts.push_back(intChar);

				// Image is converted in to a float and then flattened so it can be stored in the images.xml file and be used in KNN.
				cv::Mat matImageFloat;                         
				matROIResized.convertTo(matImageFloat, CV_32FC1);
				cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
				matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);
			}
		}
	}

	std::cout << "Training complete.";

	// Creation of the classifications.xml file which stores the ASCII values for each identified character.
	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);
	if (fsClassifications.isOpened() == false) {
		std::cout << "Cannot write classifications.xml.";
		return(0);
	}
	fsClassifications << "classifications" << matClassificationInts;
	fsClassifications.release();

	// Creation of the images.xml file which stores the images as their pixel values.
	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);
	if (fsTrainingImages.isOpened() == false) {
		std::cout << "Cannot write images.xml.";
		return(0);
	}
	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();

	return(0);
}




