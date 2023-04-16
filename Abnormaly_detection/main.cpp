#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void detect_DiffSpot_Money() {
    Mat templ = imread("normal_banknote.jpg", IMREAD_GRAYSCALE);
    Mat sample = imread("banknote_stain.jpg", IMREAD_GRAYSCALE);


    Mat diff_image;
    subtract(templ, sample, diff_image);
    imshow("result", diff_image);

    // Threshold the difference image to separate the foreground from the background
    Mat thresholded_image;
    threshold(diff_image, thresholded_image, 40, 255, cv::THRESH_BINARY);

    // Calculate the percentage of the thresholded image that is foreground (i.e., non-zero)
    double foreground_percentage = cv::countNonZero(thresholded_image) / (double)(thresholded_image.total()) * 100.0;

    // Determine if the money is real or fake based on the percentage of foreground pixels
    if (foreground_percentage > 10.0) {
        cout << "Fake money detected";

    }
    else {
        cout << "Real money detected" << std::endl;
    }
    imshow("real_money", templ);
    imshow("fake_money", sample);
    waitKey(0);
}

int getMaxPixel(Mat src) {
    int histogram[256] = {};
    int maxPos = 0;

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            histogram[src.at<uchar>(i, j)]++;
            if (histogram[src.at<uchar>(i, j)] > histogram[maxPos]) {
                maxPos = src.at<uchar>(i, j);
            }
        }
    return maxPos;
}

Mat scaleHistogram(Mat src, double rate) {
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {

            src.at<uchar>(i, j) = src.at<uchar>(i, j) * rate > 255 ? 255 : (src.at<uchar>(i, j) * rate < 0 ? 0 : src.at<uchar>(i, j) * rate);

        }
    return src;
}

void detect_DiffBrightness_Money() {

    Mat templ = imread("normal_banknote.jpg", IMREAD_GRAYSCALE);
    Mat sample = imread("thieusang.jpg", IMREAD_GRAYSCALE);
    imshow("sample", sample);

    sample = scaleHistogram(sample, (double)getMaxPixel(templ) / getMaxPixel(sample));

    Mat result = abs(sample - templ);
    threshold(result, result, 40, 255, THRESH_BINARY);
    imshow("template", templ);
    imshow("diff", result);

    // Tìm các đường viền trong ảnh kết quả
    vector<vector<Point>> contours;
    findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Vẽ đường viền lên ảnh gốc
    Mat highlighted = imread("thieusang.jpg");
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(highlighted, contours, i, Scalar(0, 0, 255), 2);
    }
    imshow("highlighted", highlighted);
}

int countHorTenPixel(Mat src, int r, int c, bool v)
{
    int count = 0;
    for (int i = c - 5; i < c + 5; i++)
        if (i >= 0 && i <= src.cols) {
            if (src.at<uchar>(r, i) <= 10 && !v) count++;
            else if (src.at<uchar>(r, i) >= 100 && v) count++;
        }
    return count;
}

Point2i getTopLeanPoint(Mat src) {
    for (int i = 1; i < src.rows; i++)
        for (int j = 1; j < src.rows; j++) {
            if ((int)src.at<uchar>(i, j) <= 10 && countHorTenPixel(src, i, j, true) > 5)
                return(Point2i(i, j));
        }

    return Point2i(0, 0);
}


Point2i getBotLeanPoint(Mat src) {
    for (int i = src.rows - 1; i >= 0; i--)
        for (int j = 1; j < src.rows; j++) {
            if ((int)src.at<uchar>(i, j) <= 10 && countHorTenPixel(src, i, j, true) > 5)
                return(Point2i(i, j));
        }

    return Point2i(0, 0);
}

int detectOrientation(Mat templ, Mat sample) {

    Mat shape;
    threshold(sample, shape, 250, 255, THRESH_BINARY);
    Point2i top = getTopLeanPoint(shape);
    Point2i bot = getBotLeanPoint(shape);
    if (top.x == 0 && top.y == 0) return 0;
    return (top.y < bot.y ? 1 : -1) * nearbyint((atan((double)abs(top.x - bot.x) / abs(top.y - bot.y)) - atan((double)templ.rows / templ.cols)) / CV_PI * 180);

}

Mat rotate(Mat src, double angle)
{
    Mat dst, dst1;
    Point2f pt(src.cols / 2., src.rows / 2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    warpAffine(src, dst1, r, Size(src.cols, src.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
    threshold(dst1, dst1, 250, 255, THRESH_BINARY);
    return dst + dst1;
}

Mat getTemplateArea(Mat src, Mat _template)
{
    Point2i topLeft = Point2i((src.rows - _template.rows) / 2, (src.cols - _template.cols) / 2);
    Point2i botRight = Point2i(topLeft.x + _template.rows, topLeft.y + _template.cols);

    return src(Range(topLeft.x, botRight.x), Range(topLeft.y, botRight.y));
}

void insertionSort(int window[])
{
    int temp, i, j;
    for (i = 0; i < 9; i++) {
        temp = window[i];
        for (j = i - 1; j >= 0 && temp < window[j]; j--)
        {
            window[j + 1] = window[j];
        }
        window[j + 1] = temp;
    }
}

Mat medianFilter(Mat src)
{
    int window[9];
    Mat dst = src.clone();

    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            dst.at<uchar>(y, x) = 0.0;

    for (int y = 1; y < src.rows - 1; y++)
        for (int x = 1; x < src.cols - 1; x++)
        {
            window[0] = src.at<uchar>(y - 1, x - 1);
            window[1] = src.at<uchar>(y, x - 1);
            window[2] = src.at<uchar>(y + 1, x - 1);
            window[3] = src.at<uchar>(y - 1, x);
            window[4] = src.at<uchar>(y, x);
            window[5] = src.at<uchar>(y + 1, x);
            window[6] = src.at<uchar>(y - 1, x + 1);
            window[7] = src.at<uchar>(y, x + 1);
            window[8] = src.at<uchar>(y + 1, x + 1);
            insertionSort(window);
            dst.at<uchar>(y, x) = window[4];
        }
    return dst;
}

void detect_DiffRotated_Money() {

    Mat templ = imread("normal_banknote.jpg", IMREAD_GRAYSCALE);
    Mat sample = imread("khachuong.png", IMREAD_GRAYSCALE);
    imshow("sample", sample);

    int angle = detectOrientation(templ, sample);
    sample = rotate(sample, angle);
    sample = getTemplateArea(sample, templ);

    Mat result = abs(sample - templ);
    result = medianFilter(result);
    threshold(result, result, 40, 255, THRESH_BINARY);
    imshow("template", templ);
    imshow("diff", result);
}

void detect_DiffSize_Money() {

    Mat templ = imread("tien_sach.png", IMREAD_GRAYSCALE);
    Mat sample = imread("money_diffsize.png", IMREAD_GRAYSCALE);
    imshow("sample", sample);

    resize(sample, sample, Size(templ.cols, templ.rows), INTER_LINEAR);

    Mat result = abs(sample - templ);
    result = medianFilter(result);
    threshold(result, result, 40, 255, THRESH_BINARY);
    imshow("template", templ);
    imshow("diff", result);
}

void GaussianFilter() {
    Mat templ = imread("tien_sach.png", IMREAD_GRAYSCALE);
    Mat image = imread("money_noise.png", IMREAD_GRAYSCALE);
    int rows = image.rows;
    int cols = image.cols;
    imshow("template", templ);
    for (int i = 0; i < rows; i++)
    {
        Vec3b* ptr = image.ptr<Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            Vec3b pixel = ptr[j];
        }
    }
    imshow("Truoc Gaussian Filter", image);
    Mat image_Gauss = image.clone();
    GaussianBlur(image, image_Gauss, Size(9, 9), 0, 0);
    for (int i = 0; i < rows; i++)
    {
        Vec3b* ptr = image_Gauss.ptr<Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            Vec3b pixel = ptr[j];
        }
    }
    imshow("Sau Gaussian Filter", image_Gauss);
    waitKey(0);
}


int menu() {

    cout << "Select a task:\n";
    cout << "1. Detect_DiffSpot_Mone\n";
    cout << "2. Detect_DiffBrightness_Money\n";
    cout << "3. Detect_DiffRotate_Money\n";
    cout << "4. Detect_DiffSize_Money\n";
    cout << "Selected Task: ";
    int r;
    cin >> r;
    return r;
}

int main() {

    switch (menu())
    {
    case 1:
        detect_DiffSpot_Money();
        break;
    case 2:
        detect_DiffBrightness_Money();
        break;
    case 3:
        detect_DiffRotated_Money();
        break;
    case 4:
        detect_DiffSize_Money();
        break;
    default:
        cout << "error!!!!!\n";
        break;
    }
    waitKey(0);
    return 0;
}
