#include "ObjLeftDetect.h"
#include "parameter.h"


using namespace std;
using namespace cv;


ObjLeftDetect::ObjLeftDetect(cv::Mat& input, int set_MOG_LearnFrame, int set_min_area, cv::Mat mask)
{
	_CBM_model = new CBM_model(input, set_MOG_LearnFrame, set_min_area, INPUT_RESIZE, mask);
	new_width = (int)(input.cols * INPUT_RESIZE);
	new_height = (int)(input.rows * INPUT_RESIZE);
	initialize();
	alarmList.clear();
}
ObjLeftDetect::~ObjLeftDetect()
{
	delete _CBM_model;


}

void ObjLeftDetect::initialize()
{
	object_detected = false;
	set_alarm = false;


	ObjLocation.clear();
	LeftLocation.clear();

 	myimg2 = cv::Mat(cv::Size(new_width,new_height),CV_8UC3, cv::Scalar::all(0));
	_ImgSynopsis = cv::Mat(cv::Size(new_width,new_height),CV_8UC3, cv::Scalar::all(0));

	_writer1.open("static_to_track.avi",cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),30,cv::Size(new_width,new_height));
	_writer2.open("summary.avi",cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),30,cv::Size(new_width,new_height));

}

bool ObjLeftDetect::process(cv::Mat& input)
{

    cv::Mat test = cv::Mat(cv::Size(new_width, new_height), CV_8UC3, cv::Scalar::all(0));
	set_alarm = false;

	cv::resize(input, myimg2, cv::Size(new_width, new_height), 0, 0, cv::INTER_NEAREST);

	object_detected = _CBM_model->Motion_Detection(myimg2);
	if (object_detected == true)
	{
		ObjLocation = _CBM_model->GetDetectResult();
		LeftLocation = _CBM_model->GetStaticForegourdResult();

		if (LeftLocation.size()>0)
		{

            _ImgSynopsis.setTo(0);
            myimg2.copyTo(_ImgSynopsis);

            for(int i = 0; i < LeftLocation.size(); i++)
            {
                cv::Rect tmp = cv::Rect(LeftLocation[i]->x, LeftLocation[i]->y, LeftLocation[i]->width, LeftLocation[i]->height);

                cv::rectangle(_ImgSynopsis, tmp, cv::Scalar(0, 0, 255));
            }
            cv::imshow("summary", _ImgSynopsis);
            _ImgSynopsis.setTo(0);
            set_alarm = true;
			_CBM_model->System_Reset();
			LeftLocation.clear();
		}
	}

	return set_alarm;
}

