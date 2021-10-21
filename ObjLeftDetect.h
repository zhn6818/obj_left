#ifndef _OBJLEFTDETECT_H_
#define _OBJLEFTDETECT_H_
#include "CBM_model.h"


#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits.h>
#include <time.h>

#include <math.h>
#include <string>

class ObjLeftDetect
{
public:
	ObjLeftDetect(cv::Mat& input, int set_MOG_LearnFrame, int set_min_area, cv::Mat mask);
	~ObjLeftDetect();
	bool process(cv::Mat& input);
	CBM_model *_CBM_model;
    VideoWriter _writer1, _writer2;

	void initialize();
	int new_width, new_height;
	vector<Obj_info *> alarmList;
	cv::Mat myimg2;
    cv::Mat _ImgSynopsis;
	bool object_detected;
	vector<Obj_info*> ObjLocation;
	vector<Obj_info*> LeftLocation;
	bool set_alarm;

};

#endif
