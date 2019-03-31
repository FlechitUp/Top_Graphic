#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

float Pi = 3.1415;

using namespace cv;
using namespace std;
string source_window = "New image";


int main( int argc, char** argv )
{
    Mat m1 = imread("test.jpg" ,CV_LOAD_IMAGE_GRAYSCALE);
    Mat m2 = Mat::zeros( m1.rows, m1.cols, m1.type() );
    Mat gray(m1.rows, m1.cols, CV_8UC1);
    //Point center = Point( m1.cols/2, m1.rows/2 );
    double angle = 90;
    double angleRadians = angle*Pi/180;

    double scale = 0.5;
    int trasx = -15, trasy = 1;
    float x, y;
    float  neww = 300, newh = 300;
    float  x_ratio = (float)m1.cols / neww;
    float  y_ratio = (float)m1.rows / newh;

    Mat m3 = Mat::zeros( neww, newh, m1.type() );

    cout<<m1.rows<<" x "<<m1.cols<<endl;
    for (int i = 0; i < m3.rows; i++){
        for (int j = 0; j < m3.cols; j++){
            /// inversa
            //m1.at<uchar>(i,j) = 255 - m1.at<uchar>(i,j);
            //m2.at<uchar>(i,j) = m1.at<uchar>(i,j);
            /// log
           // m1.at<uchar>(i,j) = log(m1.at<uchar>(i,j));
            /// inv log
            //m1.at<uchar>(i,j) = pow(m1.at<uchar>(i,j),2);
            /// pow
            //m1.at<uchar>(i,j) = pow(m1.at<uchar>(i,j),2);

            ///Traslacion
            /*x = i+ trasx;
            y = j+ trasy;
            m2.at<uchar>(i,j) = m1.at<uchar>(x,y);*/

            ///Escalar
            y = floor(j*y_ratio);
            x = floor(i*x_ratio);
            m3.at<uchar>(j,i) = m1.at<uchar>(x,y);
            ///Rotar

            /**
            (-2,-2) (-1,-2) (0,-2) (1,-2) (2,-2)
            (-2,-1)           ...         (2, -1)
            (-2,0)           (0,0)        (2, 0)
            (-2,1)            ...         (2, 1)
            (-2,2)            ...         (2,2)
            */

            /*x=((i-m1.cols/2)*cos(angleRadians)-(j-m1.rows/2)*sin(angleRadians)+m1.cols/2);
            y=((i-m1.cols/2)*sin(angleRadians)+(j-m1.rows/2)*cos(angleRadians)+m1.rows/2);
            //cout<<i-m1.cols/2<<endl;
            if((x>=0 && x<m1.cols) && (y>=0 && y<m1.rows)){
                m2.at<uchar>(x,y) = m1.at<uchar>(i,j);
            }*/
        }

    }
    /*for (int i = m1.rows-1; i >=0; i--){
        for (int j = m1.cols-1; j >=0 ; j--){
        ///Escalar
            y = floor(j*y_ratio);
            x = floor(i*x_ratio);
            m3.at<uchar>(j*2,i*2) = m1.at<uchar>(y,x);
        }
    }*/
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window,m3 );

    /// Wait until user exits the program
    waitKey(0);
    return 0;
}
