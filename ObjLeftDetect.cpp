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


    A = cv::Mat(cv::Size(new_width,new_height), CV_8UC3, cv::Scalar::all(0));
    B = cv::Mat(cv::Size(new_width,new_height), CV_8UC3, cv::Scalar::all(0));

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
			for (int i = 0; i < _ImgSynopsis.cols; i++)
			{
				for (int j = 0; j < _ImgSynopsis.rows; j++)
				{
					myColor aaa = myGet2D(_CBM_model->my_imgStatic,i,j);
					myColor bbb; bbb.B = 200; bbb.G = 250; bbb.R = 10;
					if ((aaa.B==0)&&(aaa.G==200)&&(aaa.R==255))
					{
						mySet2D(_ImgSynopsis,bbb,i,j);
					}
				}
			}
            test = _ImgSynopsis.clone();
            cv::imshow("summary", test);
//            cv::waitKey(1);
            A = test.clone();
            _ImgSynopsis.setTo(0);
            set_alarm = true;
			_CBM_model->System_Reset();
			LeftLocation.clear();
		}
	}

    _writer1.write(A);
	_writer2.write(B);


	return set_alarm;
}

