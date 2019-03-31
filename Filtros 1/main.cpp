#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
string source_window = "Source image";

int main( int argc, char** argv )
{
    Mat m1 = imread("test.jpg" ,CV_LOAD_IMAGE_GRAYSCALE);
    Mat m2 = Mat::zeros( m1.rows, m1.cols, m1.type() );
    Mat gray(m1.rows, m1.cols, CV_8UC1);
    CvScalar sRes;
    Point center = Point( m1.cols/2, m1.rows/2 );
    double angle = 45;
    double scale = 0.5;
    int trasx = 1, trasy = 1;
    int x, y;

    for (int i = 0; i < m1.rows; i++){
        for (int j = 0; j < m1.cols; j++){
            /// inversa
           // m1.at<uchar>(i,j) = 255 - m1.at<uchar>(i,j);
            //m2.at<uchar>(i,j) = m1.at<uchar>(i,j);
            /// log
           // m1.at<uchar>(i,j) = log(m1.at<uchar>(i,j));
            /// inv log
            //m1.at<uchar>(i,j) = pow(m1.at<uchar>(i,j),2);
            /// pow
            m1.at<uchar>(i,j) = pow(m1.at<uchar>(i,j),2);

            ///Girar
            /*x = ((i+center.x)*cos(angle)-(j+center.y)*sin(angle)) + center.x ;
            y = ((i+center.x)*sin(angle)+(j+center.y)*cos(angle)) + center.y;
            m2.at<uchar>(x,y) = m1.at<uchar>(i,j);*/
            ///Traslacion
            /*x = i+ trasx;
            y = j+ trasy;
            m2.at<uchar>(i,j) = m1.at<uchar>(x,y);*/


        }
    }

    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window,m1 );

    /// Wait until user exits the program
    waitKey(0);
    return 0;
}
