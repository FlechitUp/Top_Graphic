#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
    ///Mask 3x3
    Mat m1 = imread("test.jpg" ,CV_LOAD_IMAGE_GRAYSCALE);
    Mat m2 = Mat::zeros( m1.rows, m1.cols, m1.type() );
    float avrgNeighbors =0;
    for (int i = 1; i < m1.rows; i++){  /// avoiding limits
        for (int j = 1; j < m1.cols; j++){
            //Average of 9 pixels
            avrgNeighbors = ( m1.at<uchar>(i-1,j-1) + m1.at<uchar>(i-1,j)+ m1.at<uchar>(i-1,j+1)+ m1.at<uchar>(i,j-1) +m1.at<uchar>(i,j)+m1.at<uchar>(i,j+1) +m1.at<uchar>(i+1,j-1) +m1.at<uchar>(i+1,j) + m1.at<uchar>(i+1,j+1) ) /9;
            m2.at<uchar>(i,j) = /*m1.at<uchar>(i,j)-*/avrgNeighbors;
        }
    }

    namedWindow( "Original img", CV_WINDOW_AUTOSIZE );
    imshow( "Original img" ,m1 );

    namedWindow( "Mask", CV_WINDOW_AUTOSIZE );
    imshow( "Mask" ,m2 );

    waitKey(0);
    return 0;
}
