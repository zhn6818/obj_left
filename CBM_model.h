#include "VideoDetails.h"
#include "myUtility.h"
#include "myDIP.h"
#include "myGMM.h"
#include <time.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <list>
//#include <Windows.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "connected.h"
#include <functional>

#include "parameter.h"

#include "opencv2/objdetect/objdetect.hpp"

#define DEBUGIMG 1
#define Bin_num  9
#define Block_size 16
#define Block_stride 8
#define Cell_size 8
#define Win_width 64
#define Win_height 128

using namespace std;
using namespace cv;
struct Obj_info
{
public:	
	int x;
	int y;
	int width;
	int height;
	int label;
	float distance;
	bool tracked;
	double Owner_R[10];
	double Owner_G[10];
	double Owner_B[10];
	int traj_label;
	float traj_dist;

	Obj_info()
	{
		label = 0;
		distance = (float)2147483647.0;
		tracked = false;
		traj_dist = (float)2147483647.0;
		for (int i = 0; i < 10; i++)
		{
			Owner_R[i] = 0.0; Owner_G[i] = 0.0; Owner_B[i] = 0.0;//for validation method 3 
		}
	}
};

struct pixelFSM
{
	//short  long   state
	//  0    0        0
	//  0    1        1
	//  1    0        2
	//  1    1        3
public:
	int state_now;
	int state_pre;
	int static_count;
	bool staticFG_candidate;
	bool staticFG_stable;

	pixelFSM()
	{
		state_now = 0;
		state_pre = 0;
		static_count = 0;
		staticFG_candidate = false;
		staticFG_stable = false;
	}
};

class CBM_model
{
public:

	int MOG_LEARN_FRAMES;
	int MIN_AREA;
	int TEMPORAL_RULE;
//
//	VideoDetails *_video;//input video
	cv::VideoWriter _writer1, _writer2, _writer3, _writer4, _writer5;
	CBM_model(cv::Mat input, int set_MOG_LearnFrame, int set_min_area, int set_buffer_len, float set_resize, cv::Mat mask);
	~CBM_model();
	void Initialize();
	void Uninitialize();
	bool Motion_Detection(cv::Mat img);
//
	bool myClustering2( cv::Mat img, int option);//option: 0 for moving obj, 1 for static obj
	int GetLabeling2( cv::Mat pImg1, int areaThreshold, int option); //option: 0 for moving obj, 1 for static obj
//
	void myFSM(cv::Mat short_term, cv::Mat long_term, pixelFSM ** imageFSM, bool *** Previous_FG);
//
	void myConvertFSM2Img(pixelFSM **Array, cv::Mat Candidate_Fg, cv::Mat Static_Fg);
//
	int check_foreground2( cv::Mat img);
//
	int frame_count;
	int sampling_idx;
//
//	//********RESIZE**************//
	int new_width;
	int new_height;
	float RESIZE_RATE;
//	//********detected result**************//
	vector<Obj_info*> detected_result;//information of the moving objects
	vector<Obj_info*> static_object_result;//information of the static foreground objects
	vector<Obj_info*> GetDetectResult();//get the information of the moving objects
	vector<Obj_info*> GetStaticForegourdResult();//get the information of the static foreground objects
//
	bool ** GetPrevious_nForeground( int n);
//
//    cv::Ptr<cv::ml::SVM> detector;
//	cv::HOGDescriptor hog;//�ŧihog detector
//	vector<cv::Rect>  found;
//
//
	cv::Mat _GetPrevious_nFrame( int n);
//
	void System_Reset();
//
	bool ***Previous_FG;
//
//	//myImage ** _Previous_Img;
	std::vector<cv::Mat> _Previous_Img;
	cv::Mat my_imgStatic, * my_imgStatic2;
	cv::Mat maskROI;
	cv::Mat input_temp;
//private:
	cv::Mat mog_fg;//long term
	cv::Mat mog_fg2;//short term
	cv::Mat imgStatic;
//
//
    cv::Mat my_mog_fg;//long term
    cv::Mat my_mog_fg2;//short term
    cv::Mat my_imgCandiStatic;
//
	cv::Mat my_imgStatict, my_imgCandiStatict;
//
//	VideoDetails *video;
//
	myGMM * _myGMM;//long term
	myGMM * _myGMM2;//short term
//
	float MoG_LearningRate;// long term
	float MoG_LearningRate2;// short term
	int FG_count;
//
	pixelFSM **imageFSM;
//
	int staticFG_pixel_num_now;
	int staticFG_pixel_num_pre;
	int staticFG_pixel_num_pre2;
//
	cv::Mat dpm_gray;//for dpm

};
