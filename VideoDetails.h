#ifndef _VIDEO_DETAILS_H_
#define _VIDEO_DETAILS_H_

#include "opencvheader.h"

class VideoDetails
{
public:
	VideoDetails(char* filename);
	~VideoDetails();

	void VideoWriter_Initial(cv::VideoWriter writer ,char* filename, int fps);
	CvVideoWriter *writer_frame; 
	CvVideoWriter *writer_GMM; 
	CvVideoWriter *writer_Diff3; 
	CvVideoWriter *writer_Foreground; 

	bool writer_create;

	int _currentFrame;  //record current frame
	int _frameNum;      //record total frame number
	int _fps;
    cv::VideoCapture _file;
	cv::Mat _frame;
	char _videoName[50];
	int _width;
	int _height;

};
#endif
