#include "ObjLeftDetect.h"
#include "VideoDetails.h"
#include "parameter.h"
#ifdef _WIN64
#include "test.h"
#endif
//#include "opencv2/imgproc/imgproc_c.h"
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



int64 work_begin;
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
	/*if ( (Event == 1) && (roipointnumber <=3) )	
	{
		PointArray[roipointnumber/2][roipointnumber%2] = cv::Point(x,y);
        points_w[0][roipointnumber] = cv::Point(x, y);
		printf("( %d, %d)\n ",x,y);
		printf("%d\n",roipointnumber);
		roipointnumber = roipointnumber + 1;
	}*/
	if (  (Event == 1) )	
	{
		points_w[0][0] = cv::Point(211, 81);
		points_w[0][1] = cv::Point(272, 91);
		points_w[0][2] = cv::Point(249, 228);
		points_w[0][3] = cv::Point(57, 234);
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
		cv::waitKey(1);
//		cvReleaseImage(&Image1);
	}
}

inline void WorkBegin() 
{ 
	work_begin = getTickCount(); 
}

inline void WorkEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string WorkFps()
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}


int main()
{
//    GpuTimer timer;
#ifdef _WIN64
    incres();
#endif
	/************************************************************************/
	/* parameter seeting                                                    */
	/************************************************************************/
	int count = 0;
	double readData[10] = {0.0};
	FILE *InputFile;
	std::cout << __FILE__ << std::endl;
//	InputFile = fopen( "/Users/zhanghaining/CLionProjects/ObjLeft-master/parameter.txt","r");
#ifdef _WIN64
    std::cout << "windows ~~~" << std::endl;
	InputFile = fopen( "E:\\code\\ObjLeft-master\\parameter.txt","r");
#endif
#ifdef __APPLE__
    std::cout << "mac ~~~" << std::endl;
    InputFile = fopen( "/Users/zhanghaining/CLionProjects/ObjLeft-master/parameter.txt","r");
#endif
	if (InputFile == NULL)
	{
		std::cout << "loading txt failed! " << std::endl;
		system("pause");
		return -1;
	}
	else
    {
        std::cout << "loading txt success! " << std::endl;
    }
	for (int i = 0; i < 10; i++ ){
		fscanf( InputFile, "%lf", &readData[i]);
	}
	fclose(InputFile);
	for (int i = 0; i < 10; i++)
	{
		switch(i)
		{
			case 0: OWNER_SEARCH_ROI = (int)readData[i]; break;
			case 1: GMM_LEARN_FRAME = (int)readData[i]; break;
			case 2: MAX_SFG = (int)readData[i]; break;
			case 3: MIN_SFG = (int)readData[i]; break;
			case 4: MAX_FG = (int)readData[i]; break;
			case 5: MIN_FG = (int)readData[i]; break;
			case 6: BUFFER_LENGTH = (int)readData[i]; break;
			case 7: GMM_LONG_LEARN_RATE = readData[i]; break;
			case 8: GMM_SHORT_LEARN_RATE = readData[i]; break;
			case 9: INPUT_RESIZE = readData[i]; break;
		}
	}

	/************************************************************************/
	/* choose input channel                                                 */
	/************************************************************************/
	//char test_video[200] = "/Users/zhanghaining/Downloads/Download/baidunetdiskdownload/pets2006_1.avi";
	std::string test_video;
#ifdef _WIN64
    test_video = "E:\\BaiduNetdiskDownload\\test.mp4";
#endif
#ifdef __APPLE__
    test_video = "/Users/zhanghaining/jetflow/test/test.mp4";
#endif


	/************************************************************************/
	/* Video input setting                                                   */
	/************************************************************************/
	VideoDetails *_video;
	_video = new VideoDetails((char*)test_video.c_str());
	//_video = new VideoDetails("pets2006_1.avi");
	cv::Mat qImg, myimg;
	_video->_currentFrame = 0;
//	cvSetCaptureProperty(_video->_file, CV_CAP_PROP_POS_FRAMES, _video->_currentFrame);
//	myimg = cvCreateImage(cvSize(_video->_width,_video->_height),8,3);
    myimg = cv::Mat(cv::Size((int)_video->_width,(int)_video->_height), CV_8UC3);
	//cv::Mat mat_myimg(myimg,0);
    cv::Mat mat_myimg(myimg);
	//myImage * myimg1 = myCreateImage(_video->_width,_video->_height,3);
    VideoWriter _writer;
	_writer.open("video.avi",cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),30,cv::Size((int)_video->_width,(int)_video->_height),1);

	/************************************************************************/
	/* ROI setting                                                          */
	/************************************************************************/
	imageheight = (int)(_video->_height*INPUT_RESIZE);
	imagewidth = (int)(_video->_width*INPUT_RESIZE);
//	IplImage *setroi = cvQueryFrame(_video->_file);
//	IplImage *setroi2; setroi2 = cvCreateImage(cvSize(imagewidth,imageheight),8,3);
    cv::Mat setroi, setroi2;
    _video->_file.read(setroi);
    setroi2 = cv::Mat(cv::Size(imagewidth,imageheight),CV_8UC3);

	cv::resize(setroi,setroi2, cv::Size(imagewidth,imageheight));
	cv::imshow("SetROI",setroi2);
	cv::setMouseCallback("SetROI",onMouse,NULL);
	cv::waitKey(0);
//	cvDestroyWindow("SetROI");
	
	/************************************************************************/
	/* counstruct object left class                                         */
	/************************************************************************/

	ObjLeftDetect _objleft(mat_myimg,GMM_LEARN_FRAME,MIN_FG,BUFFER_LENGTH,mask);
	

	/************************************************************************/
	/* main loop                                                       */
	/************************************************************************/
	bool obj_left = false;
	while((_video->_file).read(qImg))
	{		
		WorkBegin();
//		cvCopy(qImg,myimg);
        //myimg = qImg.clone();
		qImg.copyTo(myimg);
		medianBlur( mat_myimg, mat_myimg, 3);
//		opencv_2_myImage(myimg,myimg1);//transfer opencv data to myimage data

		/************************************************************************/
		/* abandoned object detection algorithm                                 */
		/************************************************************************/

		obj_left = _objleft.process(mat_myimg);

		if (obj_left==true)
		{
			//printf("alram!!\n");
			std::cout << "alarm!!!!!!" << std::endl;
		}
		WorkEnd();
		Mat _qImg(qImg);
		putText(_qImg, "fps:" + WorkFps(), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 100, 0), 2);
		cv::imshow( "video",qImg);
		cv::imshow("video",qImg);
		 _writer.write(qImg);
		cv::waitKey(1);
	}

	system("pause");
	return 0;
}
