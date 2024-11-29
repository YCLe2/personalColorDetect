#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <vector>
#include <iostream>
#include <filesystem>

using namespace dlib;
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

Scalar getDominantColor(const Mat& region) {
    Mat data;
    region.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    Mat labels, centers;
    kmeans(data, 1, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    return Scalar(centers.at<float>(0, 0), centers.at<float>(0, 1), centers.at<float>(0, 2));
}

bool is_warm(const std::vector<float>& lab_b, const std::vector<int>& a) {
    std::vector<float> warm_b_std = {11.6518, 11.71445, 3.6484};
    std::vector<float> cool_b_std = {4.64255, 4.86635, 0.18735};

    float warm_dist = 0, cool_dist = 0;
    for (int i = 0; i < 3; ++i) {
        warm_dist += abs(lab_b[i] - warm_b_std[i]) * a[i];
        cool_dist += abs(lab_b[i] - cool_b_std[i]) * a[i];
    }
    return warm_dist <= cool_dist;
}

bool is_spr(const std::vector<float>& hsv_s, const std::vector<float>& a) {
    std::vector<float> spr_s_std = {29.1440, 25.8621, 19.6220};
    std::vector<float> fal_s_std = {35.0035, 33.8140, 20.3128};

    float spr_dist = 0, fal_dist = 0;
    for (int i = 0; i < 3; ++i) {
        spr_dist += abs(hsv_s[i] - spr_s_std[i]) * a[i];
        fal_dist += abs(hsv_s[i] - fal_s_std[i]) * a[i];
    }
    return spr_dist <= fal_dist;
}

bool is_smr(const std::vector<float>& hsv_s, std::vector<float>& a) {
    std::vector<float> smr_s_std = {13.4806, 14.3909, 13.7015};
    std::vector<float> wnt_s_std = {20.1396, 20.0621, 22.9022};
    a[1] = 0.5;

    float smr_dist = 0, wnt_dist = 0;
    for (int i = 0; i < 3; ++i) {
        smr_dist += abs(hsv_s[i] - smr_s_std[i]) * a[i];
        wnt_dist += abs(hsv_s[i] - wnt_s_std[i]) * a[i];
    }
    return smr_dist <= wnt_dist;
}

void analyzePersonalColor(const string& imgpath) {
    try {
        // Initialize dlib's face detector and shape predictor
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Load the image
        Mat frame = imread(imgpath);
        if (frame.empty()) {
            cerr << "Unable to load image: " << imgpath << endl;
            return;
        }

        cv_image<bgr_pixel> cimg(frame);
        std::vector<dlib::rectangle> faces = detector(cimg);

        if (faces.empty()) {
            cerr << "No faces detected in image: " << imgpath << endl;
            return;
        }

        full_object_detection shape = pose_model(cimg, faces[0]);

        // Extract regions
        std::vector<Point> skin_points;
        for (int i = 0; i <= 16; ++i) {
            skin_points.push_back(Point(shape.part(i).x(), shape.part(i).y()));
        }
        Rect skin_rect = boundingRect(skin_points);
        skin_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure ROI is within image bounds
        Scalar skin_color = getDominantColor(frame(skin_rect));

        if (skin_color == Scalar(0, 0, 0)) {
            cerr << "Invalid skin color detected in image: " << imgpath << endl;
            return;
        }

        Point left_eye_center(0, 0);
        for (int i = 36; i <= 41; ++i) {
            left_eye_center += Point(shape.part(i).x(), shape.part(i).y());
        }
        left_eye_center.x /= 6;
        left_eye_center.y /= 6;
        Rect left_eye_rect(left_eye_center.x - 5, left_eye_center.y - 5, 10, 10);
        left_eye_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure ROI is within image bounds
        Scalar left_eye_color = getDominantColor(frame(left_eye_rect));

        Point right_eye_center(0, 0);
        for (int i = 42; i <= 47; ++i) {
            right_eye_center += Point(shape.part(i).x(), shape.part(i).y());
        }
        right_eye_center.x /= 6;
        right_eye_center.y /= 6;
        Rect right_eye_rect(right_eye_center.x - 5, right_eye_center.y - 5, 10, 10);
        right_eye_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure ROI is within image bounds
        Scalar right_eye_color = getDominantColor(frame(right_eye_rect));

        std::vector<Point> left_eyebrow_points, right_eyebrow_points;
        for (int i = 17; i <= 21; ++i) {
            left_eyebrow_points.push_back(Point(shape.part(i).x(), shape.part(i).y()));
        }
        for (int i = 22; i <= 26; ++i) {
            right_eyebrow_points.push_back(Point(shape.part(i).x(), shape.part(i).y()));
        }
        Rect left_eyebrow_rect = boundingRect(left_eyebrow_points);
        left_eyebrow_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure ROI is within image bounds
        Scalar left_eyebrow_color = getDominantColor(frame(left_eyebrow_rect));

        Rect right_eyebrow_rect = boundingRect(right_eyebrow_points);
        right_eyebrow_rect &= Rect(0, 0, frame.cols, frame.rows); // Ensure ROI is within image bounds
        Scalar right_eyebrow_color = getDominantColor(frame(right_eyebrow_rect));

        // Calculate average colors
        Scalar cheek_color = (skin_color + skin_color) / 2;
        Scalar eyebrow_color = (left_eyebrow_color + right_eyebrow_color) / 2;
        Scalar eye_color = (left_eye_color + right_eye_color) / 2;

        // Convert colors to Lab and HSV
        std::vector<float> Lab_b, hsv_s;
        std::vector<Scalar> colors = {cheek_color, eyebrow_color, eye_color};
        for (const auto& color : colors) {
            Mat rgb(1, 1, CV_8UC3, color);
            Mat lab, hsv;
            cvtColor(rgb, lab, COLOR_BGR2Lab);
            cvtColor(rgb, hsv, COLOR_BGR2HSV);
            Lab_b.push_back(lab.at<Vec3b>(0, 0)[2] - 128);
            hsv_s.push_back(hsv.at<Vec3b>(0, 0)[1] * 100 / 255.0);
        }

        // Personal color analysis
        std::vector<int> Lab_weight = {30, 20, 5};
        std::vector<float> hsv_weight = {10, 1, 1};
        string tone;
        if (is_warm(Lab_b, Lab_weight)) {
            if (is_spr(hsv_s, hsv_weight)) {
                tone = "봄웜톤(spring)";
            } else {
                tone = "가을웜톤(fall)";
            }
        } else {
            if (is_smr(hsv_s, hsv_weight)) {
                tone = "여름쿨톤(summer)";
            } else {
                tone = "겨울쿨톤(winter)";
            }
        }

        Point result;
        cout << imgpath << "의 퍼스널 컬러는 " << tone << "입니다." << endl;

        cout << "Lab b values: ";
        float sum = 0;
        for (const auto& b : Lab_b) {
            cout << b << " ";
        }
        cout << endl;

        cout << "HSV s values: ";
        for (const auto& s : hsv_s) {
            cout << s << " ";
        }
        cout << endl;
    } catch (const cv::Exception& e) {
        cerr << "OpenCV error: " << e.what() << endl;
    } catch (const std::exception& e) {
        cerr << "Standard exception: " << e.what() << endl;
    } catch (...) {
        cerr << "Unknown error occurred." << endl;
    }
}

int main() {
    std::vector<string> seasons = {"nsummer", "nwinter"};
    string base_path = "../res/test/";

    for(const auto& season : seasons) {
        string season_path = base_path + season;
        for(const auto& entry : fs::directory_iterator(season_path)) {
            if(entry.is_regular_file()) {
                string imgpath = entry.path().string();
                analyzePersonalColor(imgpath);
            }
        }
    }
    return 0;
}