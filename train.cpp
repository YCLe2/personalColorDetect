#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Convert RGB to LAB (simple conversion function for b-value calculation)
double rgbToLabB(Vec3b rgb) {
    double r = rgb[2] / 255.0;
    double g = rgb[1] / 255.0;
    double b = rgb[0] / 255.0;

    double x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    double y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    double z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

    // Convert to Lab (only b-channel)
    double labB = (200 * (y - z));
    return labB;
}

// Function to calculate the average LAB b value and HSV S value for a set of images
pair<double, double> calculateAverageLabBandS(const vector<string>& imagePaths) {
    double totalLabB = 0.0;
    double totalS = 0.0;
    int count = 0;

    for (const auto& imagePath : imagePaths) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "이미지를 불러올 수 없습니다: " << imagePath << endl;
            continue;
        }

        // Convert image to LAB color space
        Mat labImage;
        cvtColor(image, labImage, COLOR_BGR2Lab);

        // Convert image to HSV color space
        Mat hsvImage;
        cvtColor(image, hsvImage, COLOR_BGR2HSV);

        // Calculate the average LAB b value and HSV S value for the image
        for (int y = 0; y < labImage.rows; ++y) {
            for (int x = 0; x < labImage.cols; ++x) {
                Vec3b labPixel = labImage.at<Vec3b>(y, x);
                Vec3b hsvPixel = hsvImage.at<Vec3b>(y, x);
                totalLabB += labPixel[2]; // b channel
                totalS += hsvPixel[1]; // S channel
                ++count;
            }
        }
    }

    double avgLabB = (count > 0) ? (totalLabB / count) : 0.0;
    double avgS = (count > 0) ? (totalS / count) : 0.0;
    return make_pair(avgLabB, avgS);
}

// Function to get all image paths from a directory
vector<string> getImagePaths(const string& directory) {
    vector<string> imagePaths;
    if (!fs::exists(directory)) {
        cout << "디렉토리가 존재하지 않습니다: " << directory << endl;
        return imagePaths;
    }
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

int main() {
    // Define the directories for each season
    vector<string> seasons = {"spring", "summer", "fall", "winter"};
    string baseDir = "../res/train/";

    // Open a file to save the average values
    ofstream outFile("../season_averages.txt");
    if (!outFile.is_open()) {
        cout << "파일을 열 수 없습니다!" << endl;
        return -1;
    }

    // Calculate and print the average LAB b value and HSV S value for each season
    for (const auto& season : seasons) {
        string seasonDir = baseDir + season;
        cout << "Processing directory: " << seasonDir << endl; // 디버깅 출력
        vector<string> imagePaths = getImagePaths(seasonDir);
        if (imagePaths.empty()) {
            cout << season << " 폴더에 이미지가 없습니다." << endl;
            continue;
        }
        pair<double, double> avgLabBandS = calculateAverageLabBandS(imagePaths);
        outFile << season << " " << avgLabBandS.first << " " << avgLabBandS.second << endl;
        cout << season << " 평균 LAB b 값: " << avgLabBandS.first << ", 평균 HSV S 값: " << avgLabBandS.second << endl;
    }

    outFile.close();
    return 0;
}