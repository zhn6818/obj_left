#include "CBM_model.h"
#include <iostream>
//#include <omp.h>

using namespace std;
using namespace cv;

CBM_model::CBM_model(cv::Mat &input, int set_MOG_LearnFrame, int set_min_area, float set_resize,
                     cv::Mat &mask) {
    // id = 0 ;
    frame_count = 0;

    MOG_LEARN_FRAMES = set_MOG_LearnFrame;
    MIN_AREA = set_min_area;

    new_width = (int) (input.cols * set_resize);
    new_height = (int) (input.rows * set_resize);

    Initialize();

    // Select parameters for Gaussian model.
    _myGMM = new myGMM(0.0001);//0.0001
    _myGMM2 = new myGMM(0.002);

    maskROI = mask;

}

CBM_model::~CBM_model() {
    Uninitialize();
}

void CBM_model::Initialize() {
    _writer1.open("long.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, cv::Size(new_width, new_height), 0);
    _writer2.open("short.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, cv::Size(new_width, new_height), 0);
    _writer3.open("static.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, cv::Size(new_width, new_height), 1);
    _writer5.open("DPM.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30, cv::Size(new_width, new_height), 1);



    my_mog_fg = cv::Mat(cv::Size(new_width, new_height), CV_8UC1, cv::Scalar::all(0));
    my_mog_fg2 = cv::Mat(cv::Size(new_width, new_height), CV_8UC1, cv::Scalar::all(0));
    my_imgCandiStatic = cv::Mat(cv::Size(new_width, new_height), CV_8UC3, cv::Scalar::all(0));
    my_imgStatic = cv::Mat(cv::Size(new_width, new_height), CV_8UC3, cv::Scalar::all(0));


    imageFSM = (pixelFSM **) malloc((int) new_width * sizeof(pixelFSM *));


    mog_fg.setTo(0);
    mog_fg2.setTo(0);
    imgStatic.setTo(0);

    input_temp = cv::Mat(cv::Size(new_width, new_height), CV_8UC1, cv::Scalar::all(0));

    for (int i = 0; i < new_width; i++) {
        imageFSM[i] = (pixelFSM *) malloc((int) new_height * sizeof(pixelFSM));
        memset(imageFSM[i], 0, (int) new_height * sizeof(pixelFSM));
    }

    printf("..\n");
    _Previous_Img = cv::Mat(cv::Size(new_width, new_height), CV_8UC3, cv::Scalar::all(0));
    printf("....\n");
}

void CBM_model::Uninitialize() {

    _writer1.release();
    _writer2.release();
    _writer3.release();
    _writer5.release();


    free(*imageFSM);
    free(imageFSM);
    cout << "CBM_model Released!" << endl;
}

//
void CBM_model::System_Reset() {
#pragma omp parallel for
    for (int i = 0; i < new_width; i++) {
        for (int j = 0; j < new_height; j++) {
            imageFSM[i][j].state_now = 0;
            imageFSM[i][j].staticFG_stable = false;
            imageFSM[i][j].staticFG_candidate = false;
            imageFSM[i][j].static_count = 0;
        }
    }
    static_object_result.clear();
}

//
bool CBM_model::Motion_Detection(cv::Mat &img) {

    cv::resize(img, _Previous_Img, cv::Size(img.cols, img.rows));

    if (frame_count < MOG_LEARN_FRAMES) {
        //printf("update mog %d\n",MOG_LEARN_FRAMES-frame_count);
        std::cout << "updata mog " << MOG_LEARN_FRAMES - frame_count << std::endl;
        if (frame_count == 0) {
            _myGMM->initial(_Previous_Img);
            _myGMM2->initial(_Previous_Img);
        }
        _myGMM->process(_Previous_Img, my_mog_fg);
        _myGMM2->process(_Previous_Img, my_mog_fg2);

        frame_count++;
        mog_fg = my_mog_fg.clone();
        mog_fg2 = my_mog_fg2.clone();
//		cvWriteFrame( _writer1, mog_fg);
        _writer1.write(mog_fg);
//		cvWriteFrame( _writer2, mog_fg2);
        _writer2.write(mog_fg2);
//		cvWriteFrame( _writer3, imgStatic);
        _writer3.write(imgStatic);
        return false;
    } else {
        _myGMM->process(_Previous_Img, my_mog_fg, maskROI);
        //my_mog_fg = input_temp & maskROI;

        _myGMM2->process(_Previous_Img, my_mog_fg2, maskROI);
        //my_mog_fg2 = input_temp & maskROI;

        cv::Mat structureElement = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), Point(-1, -1));
        cv::dilate(my_mog_fg, my_mog_fg, structureElement, cv::Point(-1, -1));
        cv::dilate(my_mog_fg2, my_mog_fg2, structureElement, cv::Point(-1, -1));

        if (check_foreground2(my_mog_fg) > (my_mog_fg.cols * my_mog_fg.rows * 0.30)) {//if motion detection cannot work well
            _myGMM->ChangeLearningRate(0.02);//speed up long-term model's learning rate to adapt the lighting changes.
        } else {
            _myGMM->ChangeLearningRate(0.0001);//defult long-term model learning rate
        }

        myFSM(my_mog_fg2, my_mog_fg, imageFSM);
        myConvertFSM2Img(imageFSM, my_imgCandiStatic, my_imgStatic);


//        staticFG_pixel_num_pre2 = staticFG_pixel_num_pre;
//        staticFG_pixel_num_pre = staticFG_pixel_num_now;
//        staticFG_pixel_num_now = check_foreground2(my_imgStatic);

        int stateCurrent = check_foreground2(my_imgStatic);
        stateHistory.push_back(stateCurrent);
        if(stateHistory.size() > CONSTTIME)
        {
            stateHistory.pop_front();
        }


        cv::imshow("my_imgCandiStatic", my_imgCandiStatic);
        cv::imshow("static obj", my_imgStatic);
        cv::imshow("Long-term", my_mog_fg);
        cv::imshow("Short-term", my_mog_fg2);


        bool static_object_detected = false;
        if ( isEqual() && ( stateCurrent > 0)) {
            static_object_detected = myClustering2(my_imgStatic, 1);
        }

        _writer1.write(my_mog_fg);
        _writer2.write(my_mog_fg2);
        _writer3.write(my_imgStatic);

        return static_object_detected;
    }
}

//
//
bool CBM_model::isEqual()
{
    bool isEqualDeque = true;
    if(stateHistory.size() < CONSTTIME)
    {
        isEqualDeque = false;
        return isEqualDeque;
    }
    int val = stateHistory[0];
    for(int i = 0; i < stateHistory.size(); i++)
    {
        if(val != stateHistory[i])
        {
            isEqualDeque = false;
        }
    }
    return isEqualDeque;
}
bool CBM_model::myClustering2(cv::Mat &img, int option) {
    int area_threshold = 0;
    cv::Mat temp;
    temp = cv::Mat(cv::Size(new_width, new_height), CV_8UC1, cv::Scalar::all(0));
    if (img.channels() == 3)//static foreground object
    {
        cv::cvtColor(img, temp, COLOR_BGR2GRAY);
        area_threshold = MIN_AREA / 2;//0;
    } else if (img.channels() == 1)//foreground detection
    {
        img.copyTo(temp);
        area_threshold = MIN_AREA;
    }
    int found_objnum = 0;
    found_objnum = GetLabeling2(temp, area_threshold, option);

    if (found_objnum > 0) {
        return true;
    } else {
        return false;
    }
}

//
///************************************************************************/
///*
//GetLabeling : input a binary frame, bounding the connected component.
//Ignore the connected component when :  case1.  It's pixel is more than a areaThreshold.
//                                       case2.  The bounding rectangle is too thin or fat.  */
///************************************************************************/
int CBM_model::GetLabeling2(cv::Mat &pImg1, int areaThreshold, int option) {
    // std::cout << "GetLabeling2" << std::endl;
    int found_objnum = 0;
    if (option == 0) {
        detected_result.clear();//clear the vector
    }
    if (option == 1) {
        static_object_result.clear();//clear the vectors
    }
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(pImg1, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    int areaThreshold_max = 0, areaThreshold_min = 0;
    if (option == 0)//for moving foreground
    {
        areaThreshold_max = MAX_FG;
        areaThreshold_min = MIN_FG;
    } else if (option == 1) {
        areaThreshold_max = MAX_SFG;
        areaThreshold_min = MIN_SFG;
    }
    for (int i = 0; i < contours.size(); i++) {
        cv::Rect currentBox = cv::boundingRect(contours[i]);
        if ((((int) currentBox.width * (int) currentBox.height) > areaThreshold_min) && (((int) currentBox.width * (int) currentBox.height) < (float) areaThreshold_max))
        {
            Obj_info *element;
            element = new Obj_info;
            element->x = currentBox.x;
            element->y = currentBox.y;
            element->width = currentBox.width;
            element->height = currentBox.height;
            if (option == 0)
                detected_result.push_back(element);
            if (option == 1)
                static_object_result.push_back(element);
        }

        found_objnum++;
    }

    return found_objnum;
}

//
//
vector<Obj_info *> CBM_model::GetDetectResult() {
    return detected_result;
}

vector<Obj_info *> CBM_model::GetStaticForegourdResult() {
    return static_object_result;
}



void CBM_model::myFSM(cv::Mat &short_term, cv::Mat &long_term, pixelFSM **imageFSM) {
    //short  long   state
    //  0    0        0
    //  0    1        1
    //  1    0        2
    //  1    1        3
    myColor buffer[2];
#pragma omp parallel for
    for (int i = 0; i < new_width; i++) {
        for (int j = 0; j < new_height; j++) {
            buffer[0] = myGet2D(short_term, i, j);
            buffer[1] = myGet2D(long_term, i, j);

            imageFSM[i][j].state_pre = imageFSM[i][j].state_now;
            imageFSM[i][j].state_now = 0;

            if ((buffer[0].B == 255) && (buffer[0].G == 255) && (buffer[0].R == 255)) {
                imageFSM[i][j].state_now += 2;
            } else {
                imageFSM[i][j].state_now = 0;
            }

            if ((buffer[1].B == 255) && (buffer[1].G == 255) && (buffer[1].R == 255)) {
                imageFSM[i][j].state_now++;
            } else {
                imageFSM[i][j].state_now = 0;
            }

            if ((imageFSM[i][j].state_now == 1) && (imageFSM[i][j].state_pre == 1)) {
                if (imageFSM[i][j].static_count == (450)) {
                    imageFSM[i][j].staticFG_stable = true;
                }

                if (imageFSM[i][j].staticFG_candidate == true) {
                    imageFSM[i][j].static_count++;
                }
            } else {
                imageFSM[i][j].static_count = 0;
                imageFSM[i][j].staticFG_candidate = false;
            }

            if ((imageFSM[i][j].state_now == 1) && (imageFSM[i][j].state_pre == 3)) {
                imageFSM[i][j].staticFG_candidate = true;
            }

        }
    }
}

void CBM_model::myConvertFSM2Img(pixelFSM **Array, cv::Mat &Candidate_Fg, cv::Mat &Static_Fg) {
    myColor color1, color2;
    color1.B = 0;
    color1.G = 0;
    color1.R = 255;
    color2.B = 0;
    color2.G = 200;
    color2.R = 255;
#pragma omp parallel for
    for (int i = 0; i < new_width; i++) {
        for (int j = 0; j < new_height; j++) {
            if (Array[i][j].staticFG_candidate == true)
                mySet2D(Candidate_Fg, color1, i, j);
            else {
                myColor a;
                a.B = 0;
                a.G = 0;
                a.R = 0;
                mySet2D(Candidate_Fg, a, i, j);
            }

            if (Array[i][j].staticFG_stable == true)
                mySet2D(Static_Fg, color2, i, j);
            else {
                myColor a;
                a.B = 0;
                a.G = 0;
                a.R = 0;
                mySet2D(Static_Fg, a, i, j);
            }
        }
    }
}

int CBM_model::check_foreground2(cv::Mat &img) {
    cv::Mat grayImg;
    if(img.channels() == 3)
    {
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    } else{
        grayImg = img;
    }

    std::vector<cv::Point> pNoZero;
    cv::findNonZero(grayImg, pNoZero);

    return pNoZero.size();
}
myColor myGet2D(cv::Mat& input, int x, int y)
{
    int width = input.cols;
    int height = input.rows;
    int depth = input.channels();
    myColor colors;
    if (depth == 1)
    {
        colors.B = input.at<uchar>(y, x);//B
        colors.G = colors.B;//G
        colors.R = colors.B;//R
    }
    else if (depth == 3){
        colors.B = input.at<cv::Vec3b>(y, x)[0];//B
        colors.G = input.at<cv::Vec3b>(y, x)[1];//G
        colors.R = input.at<cv::Vec3b>(y, x)[2];//R
    }
    return colors;
}

/************************************************************************/
/* mySet2D: assign RGB value                        */
/************************************************************************/
void mySet2D(cv::Mat& input, myColor colors, int x, int y)
{
    int width = input.cols;
    int height = input.rows;
    int depth = input.channels();

    if (depth == 1)
    {
        input.at<uchar>(y, x) = colors.B;
    }
    else if (depth == 3)
    {
        input.at<cv::Vec3b>(y, x)[0] = colors.B;
        input.at<cv::Vec3b>(y, x)[1] = colors.G;
        input.at<cv::Vec3b>(y, x)[2] = colors.R;
    }
}
