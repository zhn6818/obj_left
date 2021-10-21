#include "myGMM.h"
#include <time.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <list>

#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <functional>
#include <vector>
#include "parameter.h"
#include <deque>
#include "opencv2/objdetect/objdetect.hpp"



using namespace std;
using namespace cv;

const int CONSTTIME = 5;
const int staticTime = 300;

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
	std::vector<std::vector<cv::Point>> contours;

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

struct myColor
{
    unsigned char B; unsigned char G;  unsigned char R;
};
myColor myGet2D(cv::Mat& input, int x, int y);
void mySet2D(cv::Mat& input, myColor colors, int x, int y);


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
	// long id;
	int MOG_LEARN_FRAMES;
	int MIN_AREA;
//
//	VideoDetails *_video;//input video
	cv::VideoWriter _writer1, _writer2, _writer3, _writer4, _writer5;
	CBM_model(cv::Mat& input, int set_MOG_LearnFrame, int set_min_area, float set_resize, cv::Mat& mask);
	~CBM_model();
	void Initialize();
	void Uninitialize();
	bool Motion_Detection(cv::Mat& img);
//
	bool myClustering2( cv::Mat& img, int option);//option: 0 for moving obj, 1 for static obj
	int GetLabeling2( cv::Mat& pImg1, int areaThreshold, int option); //option: 0 for moving obj, 1 for static obj
//
	void myFSM(cv::Mat& short_term, cv::Mat& long_term, pixelFSM ** imageFSM);
//
	void myConvertFSM2Img(pixelFSM **Array, cv::Mat& Candidate_Fg, cv::Mat& Static_Fg);
//
	int check_foreground2( cv::Mat& img);
//
	int frame_count;

//
//	//********RESIZE**************//
	int new_width;
	int new_height;

//	//********detected result**************//
	vector<Obj_info*> detected_result;//information of the moving objects
	vector<Obj_info*> static_object_result;//information of the static foreground objects
	vector<Obj_info*> GetDetectResult();//get the information of the moving objects
	vector<Obj_info*> GetStaticForegourdResult();//get the information of the static foreground objects
    bool isEqual();
	void System_Reset();

	cv::Mat _Previous_Img;
	cv::Mat my_imgStatic;
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


	myGMM * _myGMM;//long term
	myGMM * _myGMM2;//short term

	pixelFSM **imageFSM;
	std::deque<int> stateHistory;

};
