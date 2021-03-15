#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\ml\ml.hpp>
#include <iostream>
#include "dirent.h"

using namespace cv;
using namespace std;

std::vector<string> getFiles(char* folder) {
	vector<string> files;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(folder)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			files.push_back(ent->d_name);
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}
	return files;
}

void train(char* databasePath) {
	string databasePathString = string(databasePath);
	vector<string> folders = getFiles(databasePath);
	BOWKMeansTrainer trainer(100);
	SiftFeatureDetector detector = SiftFeatureDetector();
	SiftDescriptorExtractor descriptor = SiftDescriptorExtractor();
	vector<KeyPoint> keypoints;
	vector<string> files;
	//Reades every folder in root folder
	for (int i = 2; i < folders.size(); i++) {
		string folderPath = databasePathString + "\\" + folders[i];

		//Converts string to char* so it can be used by getFiles
		std::string str = folderPath;
		char * folderPathCHAR = new char[str.size() + 1];
		std::copy(str.begin(), str.end(), folderPathCHAR);
		folderPathCHAR[str.size()] = '\0';
		files = getFiles(folderPathCHAR);

		//Reads every files in folder[i]
		for (int j = 2; j < files.size(); j++) {
			string imagePath = databasePathString + "\\" + folders[i] + "\\" + files[j];
			Mat image = imread(imagePath);
			detector.detect(image, keypoints);
			Mat descriptors;
			descriptor.compute(image, keypoints, descriptors);
			trainer.add(descriptors);
			keypoints.clear();
		}
	}

	//Creating and saving vocabulary, after collecting information from all images
	cv::Mat vocabulary = trainer.cluster();
	cv::FileStorage file("vocab.xml", cv::FileStorage::WRITE);
	file << "vocab" << vocabulary;
	file.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract(extractor, matcher);
	dextract.setVocabulary(vocabulary);
	Mat alldescs;

	//Creates alldescs table, which stores the specified number of
	//keypoints for every image in train set

	for (int i = 2; i < folders.size(); i++) {

		string folderPath = databasePathString + "\\" + folders[i];
		std::string str = folderPath;
		char * folderPathCHAR = new char[str.size() + 1];
		std::copy(str.begin(), str.end(), folderPathCHAR);
		folderPathCHAR[str.size()] = '\0';
		files = getFiles(folderPathCHAR);

		for (int j = 2; j < files.size(); j++) {

			string imagePath = databasePathString + "\\" + folders[i] + "\\" + files[j];

			Mat image = imread(imagePath);
			detector.detect(image, keypoints);
			Mat descriptors;
			descriptor.compute(image, keypoints, descriptors);
			cv::Mat desc;
			dextract.compute(image, keypoints, desc);
			alldescs.push_back(desc);

			keypoints.clear();
		}
	}




	CvSVM svm1, svm2, svm3, svm4, svm5, svm6, svm7, svm8, svm9, svm10;
	CvSVMParams params;


	int counter = 0;										 //Used for puting 0 in the right line of table alllabels
	Mat alllabels = Mat::ones(alldescs.rows, 10, CV_32FC1); //Each column represents one class, each row represents an image
	for (int i = 2; i < folders.size(); i++) {


		string folderPath = databasePathString + "\\" + folders[i];
		std::string str = folderPath;
		char * folderPathCHAR = new char[str.size() + 1];
		std::copy(str.begin(), str.end(), folderPathCHAR);
		folderPathCHAR[str.size()] = '\0';
		files = getFiles(folderPathCHAR);



		


		for (int j = 2; j < files.size(); j++) {



			alllabels.at<float>(counter,i-2) = 0;


			counter += 1;
		
		}


	}

	//Training 10 svm's for one vs all calssification.
	//Using alllabels.col() to seperate the appropriate column from table alllabels
	svm1.train_auto(alldescs, alllabels.col(0), Mat(), Mat(), params);
	svm2.train_auto(alldescs, alllabels.col(1), Mat(), Mat(), params);
	svm3.train_auto(alldescs, alllabels.col(2), Mat(), Mat(), params);
	svm4.train_auto(alldescs, alllabels.col(3), Mat(), Mat(), params);
	svm5.train_auto(alldescs, alllabels.col(4), Mat(), Mat(), params);
	svm6.train_auto(alldescs, alllabels.col(5), Mat(), Mat(), params);
	svm7.train_auto(alldescs, alllabels.col(6), Mat(), Mat(), params);
	svm8.train_auto(alldescs, alllabels.col(7), Mat(), Mat(), params);
	svm9.train_auto(alldescs, alllabels.col(8), Mat(), Mat(), params);
	svm10.train_auto(alldescs, alllabels.col(9), Mat(), Mat(), params);










	svm1.save("svm1.xml");
	svm2.save("svm2.xml");
	svm3.save("svm3.xml");
	svm4.save("svm4.xml");
	svm5.save("svm5.xml");
	svm6.save("svm6.xml");
	svm7.save("svm7.xml");
	svm8.save("svm8.xml");
	svm9.save("svm9.xml");
	svm10.save("svm10.xml");



}


int main(int argc, char** argv) {
	//train("c:\\imagedb");



	cv::Mat vocabulary;
	cv::FileStorage file("vocab.xml", cv::FileStorage::READ);
	file["vocab"] >> vocabulary;
	file.release();

	string databasePathString = string("c:\\imagedb_test");
	string folderPath;
	string imagePath;
	std::string str;

	vector<KeyPoint> keypoints;
	vector<string> files;
	vector<string> folders = getFiles("c:\\imagedb_test");

	int right=0; //Counts number of images correctly classfied


	for (int k = 2; k < folders.size(); k++)
	{

		folderPath = databasePathString + "\\" + folders[k];
		std::string str = folderPath;
		char * folderPathCHAR = new char[str.size() + 1];
		std::copy(str.begin(), str.end(), folderPathCHAR);
		folderPathCHAR[str.size()] = '\0';
		files = getFiles(folderPathCHAR);
		int rightPerClass = 0;

		//Classification of every image in our test set
		for (int l = 2; l < files.size(); l++) {

			
			imagePath = databasePathString + "\\" + folders[k] + "\\" + files[l];
			Mat test_img = imread(imagePath);

			cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
			SiftFeatureDetector detector = SiftFeatureDetector();
			cv::Ptr<cv::DescriptorExtractor> descriptor = new cv::SiftDescriptorExtractor();
			cv::BOWImgDescriptorExtractor dextract(descriptor, matcher);
			dextract.setVocabulary(vocabulary);

			detector.detect(test_img, keypoints);
			Mat descriptors;
			descriptor->compute(test_img, keypoints, descriptors);
			cv::Mat desc;
			dextract.compute(test_img, keypoints, desc);

			CvSVM svm1, svm2, svm3, svm4, svm5, svm6, svm7, svm8, svm9, svm10;
			svm1.load("svm1.xml");
			svm2.load("svm2.xml");
			svm3.load("svm3.xml");
			svm4.load("svm4.xml");
			svm5.load("svm5.xml");
			svm6.load("svm6.xml");
			svm7.load("svm7.xml");
			svm8.load("svm8.xml");
			svm9.load("svm9.xml");
			svm10.load("svm10.xml");

			//Calculation prediction of every svm for each image
			float prediction[10];
			prediction[0] = svm1.predict(desc, true);
			prediction[1] = svm2.predict(desc, true);
			prediction[2] = svm3.predict(desc, true);
			prediction[3] = svm4.predict(desc, true);
			prediction[4] = svm5.predict(desc, true);
			prediction[5] = svm6.predict(desc, true);
			prediction[6] = svm7.predict(desc, true);
			prediction[7] = svm8.predict(desc, true);
			prediction[8] = svm9.predict(desc, true);
			prediction[9] = svm10.predict(desc, true);

			//Printing out predictions
			cout << "We are currently in subfolder: " << folders[k] << "\n \n";
			cout << "Prediction values for each class: \n \n";
			cout << "1)Cannon= " << prediction[0] << "\n";
			cout << "2)Chair= " << prediction[1] << "\n";
			cout << "3)Crocodile= " << prediction[2] << "\n";
			cout << "4)Elephant= " << prediction[3] << "\n";
			cout << "5)Flamingo= " << prediction[4] << "\n";
			cout << "6)Helicopter= " << prediction[5] << "\n";
			cout << "7)Motorbikes= " << prediction[6] << "\n";
			cout << "8)Scissors= " << prediction[7] << "\n";
			cout << "9)Strawberry= " << prediction[8] << "\n";
			cout << "10)Sunflower= " << prediction[9] << "\n";
			
			//Making decision using the maximum of the decision values
			float max = -1000;
			int position = 0;
			for (int i = 0; i < 10; i++)
			{
				if (prediction[i] > max)
				{
					max = prediction[i];
					position = i + 1;
				}

			}
			


			cout << "This image belongs to class " << position << " \n \n";

			//Calculating number right guesses
			if (k - 2 == position - 1)
			{
				right += 1;
				rightPerClass += 1;
			}

			if (l== files.size() - 1)
			{
				cout << "\n \n In Class " << folders[k] << " succesfull guesses " << rightPerClass << "/" << (files.size() - 2) << "\n \n";

			}
			keypoints.clear();
		}
	}
	//Calculating percentage of success
	float percent=right * 100.0 / 56;

	cout << "\n" << "We have find " << right << " out of 56";
	cout << "\n" << "Percentage : " << percent << "\n \n";



	system("pause");

	return 0;
}