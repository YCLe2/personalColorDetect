#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h> // 추가된 헤더
#include <dlib/opencv.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <fstream>
#include <map>

using namespace cv;
using namespace std;

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

// Calculate dominant colors using k-means
vector<Vec3b> getDominantColors(const Mat& image, const vector<Point>& points, int clusters = 3) {
    Mat samples(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); ++i) {
        Vec3b color = image.at<Vec3b>(points[i]);
        samples.at<float>(i, 0) = color[0];
        samples.at<float>(i, 1) = color[1];
        samples.at<float>(i, 2) = color[2];
    }

    Mat labels;
    Mat centers;
    kmeans(samples, clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    vector<Vec3b> dominantColors;
    for (int i = 0; i < centers.rows; ++i) {
        Vec3b color;
        color[0] = centers.at<float>(i, 0);
        color[1] = centers.at<float>(i, 1);
        color[2] = centers.at<float>(i, 2);
        dominantColors.push_back(color);
    }

    return dominantColors;
}

// Function to load season averages from file
map<string, pair<double, double>> loadSeasonAverages(const string& filename) {
    map<string, pair<double, double>> seasonAverages;
    ifstream inFile(filename);
    if (!inFile.is_open()) {
        cout << "파일을 열 수 없습니다: " << filename << endl;
        return seasonAverages;
    }

    string season;
    double avgLabB, avgS;
    while (inFile >> season >> avgLabB >> avgS) {
        seasonAverages[season] = make_pair(avgLabB, avgS);
    }

    inFile.close();
    return seasonAverages;
}

// Function to analyze personal color
string analyzePersonalColor(const vector<double>& labB, const vector<double>& hsvS, const map<string, pair<double, double>>& seasonAverages) {
    map<string, double> distances;

    for (const auto& [season, averages] : seasonAverages) {
        double avgLabB = averages.first;
        double avgS = averages.second;

        double labBDist = 0.0;
        double sDist = 0.0;
        for (int i = 0; i < labB.size(); i++) {
            labBDist += abs(labB[i] - avgLabB);
            sDist += abs(hsvS[i] - avgS);
        }

        double totalDist = labBDist + sDist;
        distances[season] = totalDist;
    }

    string bestSeason;
    double minDist = numeric_limits<double>::max();
    for (const auto& [season, dist] : distances) {
        if (dist < minDist) {
            minDist = dist;
            bestSeason = season;
        }
    }

    return bestSeason;
}

// 화이트 밸런스를 조정하는 함수
void adjustWhiteBalance(Mat& frame) {
    Mat lab_image;
    cvtColor(frame, lab_image, COLOR_BGR2Lab);

    vector<Mat> lab_planes(3);
    split(lab_image, lab_planes);

    equalizeHist(lab_planes[0], lab_planes[0]);

    merge(lab_planes, lab_image);
    cvtColor(lab_image, frame, COLOR_Lab2BGR);
}

int main(int argc, char** argv) {
    // 웹캠을 열기
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "웹캠을 열 수 없습니다!" << endl;
        return -1;
    }

    // dlib의 얼굴 검출기 및 랜드마크 검출기 로드
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Load season averages
    map<string, pair<double, double>> seasonAverages = loadSeasonAverages("../season_averages.txt");

    while (true) {
        Mat frame;
        cap >> frame; // 웹캠에서 프레임 캡처
        if (frame.empty()) {
            cout << "프레임을 캡처할 수 없습니다!" << endl;
            break;
        }

        // 화이트 밸런스 조정
        adjustWhiteBalance(frame);

        // OpenCV Mat을 dlib 이미지로 변환
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);

        // 얼굴 검출
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        // 검출된 얼굴에 랜드마크 그리기 및 색상 추출
        for (auto face : faces) {
            dlib::full_object_detection shape = pose_model(dlib_img, face);

            // 랜드마크 그리기
            for (unsigned int i = 0; i < shape.num_parts(); ++i) {
                circle(frame, Point(shape.part(i).x(), shape.part(i).y()), 2, Scalar(0, 255, 0), -1);
            }

            // 피부, 눈동자, 눈썹 영역의 대표 색상 추출
            vector<Point> skinPoints, eyePoints, eyebrowPoints;
            for (int i = 0; i < 68; ++i) {
                if (i >= 0 && i <= 16) skinPoints.push_back(Point(shape.part(i).x(), shape.part(i).y())); // 얼굴 외곽
                if (i >= 36 && i <= 41) eyePoints.push_back(Point(shape.part(i).x(), shape.part(i).y())); // 왼쪽 눈
                if (i >= 42 && i <= 47) eyePoints.push_back(Point(shape.part(i).x(), shape.part(i).y())); // 오른쪽 눈
                if (i >= 17 && i <= 26) eyebrowPoints.push_back(Point(shape.part(i).x(), shape.part(i).y())); // 눈썹
            }

            vector<Vec3b> skinColors = getDominantColors(frame, skinPoints);
            vector<Vec3b> eyeColors = getDominantColors(frame, eyePoints);
            vector<Vec3b> eyebrowColors = getDominantColors(frame, eyebrowPoints);

            // LAB b값 계산
            vector<double> labB;
            for (const auto& color : skinColors) {
                labB.push_back(rgbToLabB(color));
            }

            // HSV 채도 값 계산 (예시로 임의의 값 사용)
            vector<double> hsvS = {30.0, 25.0, 20.0};

            // 퍼스널 컬러 진단
            string tone = analyzePersonalColor(labB, hsvS, seasonAverages);

            // 프레임에 텍스트 표시
            putText(frame,tone, Point(10, 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 8);
        }

        // 결과 프레임 표시
        imshow("웹캠 얼굴 인식", frame);

        // 'q' 키를 누르면 종료
        if (waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}