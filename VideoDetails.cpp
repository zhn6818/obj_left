#include "VideoDetails.h"
using namespace std;
VideoDetails::VideoDetails(char* filename)
{
	if (strcmp(filename, "")== 0)
	{
		_file  =  cv::VideoCapture(0);
		_fps   =  30;
	}
	else{
		_file  =  cv::VideoCapture(filename);
		_fps   =  (int)_file.get(cv::CAP_PROP_FPS);
	}
	_width =  (int)_file.get(cv::CAP_PROP_FRAME_WIDTH);
	_height = (int)_file.get(cv::CAP_PROP_FRAME_HEIGHT);
	_frameNum = (long)_file.get(cv::CAP_PROP_FRAME_COUNT);
	//cvSetCaptureProperty( _file, CV_CAP_PROP_POS_FRAMES, 0 );/* Return to the beginning */
	//_frame = cvCreateImage(cvSize(_width,_height), IPL_DEPTH_8U, 3);//kv
	_frame = cv::Mat(cv::Size(_width, _height), CV_8UC3);
	cout<<"video fps = "<<_fps << " frameNum: " << _frameNum <<endl;
}
VideoDetails::~VideoDetails()
{
//	cvReleaseCapture(&_file);
//	if (writer_create==true){
//        cvReleaseVideoWriter(&writer_frame);
//    }
    _file.release();
   // writer.release()
};

void VideoDetails::VideoWriter_Initial(cv::VideoWriter writer ,char* filename, int fps)
{
	int AviForamt = 0;
	//int fps = 60; // or 25 
    cv::Size AviSize = cv::Size( _width,_height);
	int AviColor = 1; //0: binary  1:color
	writer = cv::VideoWriter( filename,cv::VideoWriter::fourcc('X','V','I','D'),fps,AviSize,AviColor);
	writer_create = true;
}