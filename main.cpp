#include "ObjLeftDetect.h"
#include "parameter.h"

#include <iostream>
using namespace std;


int OWNER_SEARCH_ROI;			
int GMM_LEARN_FRAME;//frame number for GMM initialization		
int MAX_SFG;					
int MIN_SFG;					
int MAX_FG;						
int MIN_FG;						
int BUFFER_LENGTH;				
double GMM_LONG_LEARN_RATE;		
double GMM_SHORT_LEARN_RATE;	
double INPUT_RESIZE;	



double work_begin;
double work_fps;

int roipointnumber = 0;	
int arr[1] = {4}; 
cv::Point PointArray1[4];
cv::Point *PointArray[2]= {&PointArray1[0],&PointArray1[2]};
cv::Point points_w[1][4];

int imageheight;
int imagewidth;
cv::Mat mask;

void onMouse(int Event,int x,int y,int flags,void* param );

void onMouse(int Event,int x,int y,int flags,void* param )
{
	if ( (Event == 1) && (roipointnumber <=3) )
	{
		PointArray[roipointnumber/2][roipointnumber%2] = cv::Point(x,y);
        points_w[0][roipointnumber] = cv::Point(x, y);
		printf("( %d, %d)\n ",x,y);
		printf("%d\n",roipointnumber);
		roipointnumber = roipointnumber + 1;
	}
	if (  (Event == 1) && (roipointnumber >3))
	{
//		points_w[0][0] = cv::Point(211, 81);
//		points_w[0][1] = cv::Point(272, 91);
//		points_w[0][2] = cv::Point(249, 228);
//		points_w[0][3] = cv::Point(57, 234);
//		IplImage *Image1 = cvCreateImage(cv::Size(imagewidth,imageheight), IPL_DEPTH_8U, 1);
        cv::Mat Image1 = cv::Mat(cv::Size(imagewidth, imageheight), CV_8UC1, cv::Scalar::all(0));
		cv::polylines( Image1, PointArray, arr, 1, 1, CV_RGB(255,255,255));
		const Point* ppt[1] = { points_w[0]};
		cv::fillPoly(Image1, ppt,arr,1,CV_RGB(255,255,255));
		cv::threshold(Image1,Image1,254,255,cv::THRESH_BINARY);
		mask = Image1;
//		opencv_2_myImage(mask,mymask);
		cv::imwrite("mask.jpg",Image1);
		cv::imshow("SetROI",Image1);
//		cv::waitKey(1);
//		cvReleaseImage(&Image1);
	}
}

inline void WorkBegin() 
{ 
	work_begin = getTickCount(); 
}

inline void WorkEnd()
{
    work_begin = getTickCount() - work_begin;
    work_fps = work_begin * 1000 / cv::getTickFrequency();
}

inline string WorkFps()
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}

const int iScale = 1;

int main()
{
    GMM_LEARN_FRAME = 500;
    MAX_SFG = 200;
    MIN_SFG = 20;
    MAX_FG = 1000;
    MIN_FG = 20;
    INPUT_RESIZE = 0.5;
    BUFFER_LENGTH = 900;

	std::string test_video;
//1_short_Trim.mp4
//    test_video = "/Users/zhanghaining/git/obj_left/sss.avi";
    test_video = "/Users/zhanghaining/git/obj_left/test.mp4";

    cv::VideoCapture capture(test_video);
    if (!capture.isOpened())
        std::cout << "fail to open!" << std::endl;

	cv::Mat qImg, myimg;

    int realH = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int realW = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);

    realH /= iScale;
    realW /= iScale;

    myimg = cv::Mat(cv::Size(realW,realH), CV_8UC3);
    cv::Mat mat_myimg(myimg);


	imageheight = (int)realH*INPUT_RESIZE;
	imagewidth = (int)realW*INPUT_RESIZE;

    cv::Mat setroi, setroi2;
    capture.read(setroi);
    cv::resize(setroi, setroi, cv::Size(realW, realH));
    setroi2 = cv::Mat(cv::Size(imagewidth,imageheight),CV_8UC3);

	cv::resize(setroi,setroi2, cv::Size(imagewidth,imageheight));
	cv::imshow("SetROI",setroi2);
	cv::setMouseCallback("SetROI",onMouse,NULL);
	cv::waitKey(0);

	ObjLeftDetect _objleft(mat_myimg,GMM_LEARN_FRAME,MIN_FG,mask);

	bool obj_left = false;
	long currentFrame = 0;
	while(capture.read(qImg))
	{
        WorkBegin();
        cv::resize(qImg, qImg, cv::Size(realW, realH));
        qImg.copyTo(myimg);
        medianBlur( mat_myimg, mat_myimg, 3);
		obj_left = _objleft.process(mat_myimg);
        WorkEnd();
		if (obj_left==true)
		{
			//printf("alram!!\n");
			std::cout << "alarm!!!!!!" << std::endl;
		}

		Mat _qImg(qImg);
		putText(_qImg, "frame: " + std::to_string(currentFrame) + "  time:" + WorkFps(), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 100, 255), 2);
		cv::imshow( "video",qImg);
//		 _writer.write(qImg);
        currentFrame++;

		cv::waitKey(1);
	}

//	system("pause");
	return 0;
}
