/************************************************************************/
/* This code was created by Kevin (Ke-Yun) Lin at October, 2013.  
   myDIP includes many basic image processing tools
   feel free to contact me should you have any question!    
   email: kevinlin311.tw[at]gmail.com                                   */
/************************************************************************/

#include "myDIP.h" 

  
using namespace std; 
  
/************************************************************************/
/* Memory_Allocate: construct 3D array                                  */
/************************************************************************/
void*** Memory_Allocate(int width, int height, int depth) 
{ 
    unsigned char ***pixel_index; 
    pixel_index = (unsigned char ***)malloc(width*sizeof(unsigned char **));       
  
    for( int i = 0; i < width; i++)                              
        pixel_index[i] = (unsigned char **)malloc(height*sizeof(unsigned char *));  
  
    for( int i = 0; i < width; i++)                             
        for( int j = 0; j < height; j++)                           
            pixel_index[i][j] = (unsigned char *)malloc(depth*sizeof(unsigned char)); 
  
    for( int i = 0; i < width; i++) 
        for( int j = 0; j < height; j++) 
            for( int k = 0; k < depth; k++) 
                pixel_index[i][j][k] = NULL; 
  
    return (void***)pixel_index; 
} 
  
  
/************************************************************************/
/* Memory_Free: Free 3D array                                           */
/************************************************************************/
void Memory_Free(unsigned char ***pixel_index,int width, int height) 
{ 
    free(**pixel_index); 
    free(*pixel_index); 
    free(pixel_index); 
} 
  
/************************************************************************/
/* Memory_Allocate: construct 1D array                                   */
/************************************************************************/
void* Memory_Allocate1D(int width, int height, int depth) 
{ 
    unsigned char *pixel_index; 
    pixel_index = (unsigned char *)malloc(width*height*depth*sizeof(unsigned char));       
  
    for( int i = 0; i < width*height*depth; i++) 
        pixel_index[i] = NULL; 
  
    return (void*)pixel_index; 
} 
  
  
/************************************************************************/
/* Memory_Free: Free 1D array                                          */
/************************************************************************/
void Memory_Free1D(unsigned char *pixel_index,int width, int height) 
{ 
    free(pixel_index); 
} 
  
void myImageAND(cv::Mat input1, cv::Mat input2, cv::Mat output)
{ 
    if ((input1.rows!=input2.rows)||(input1.cols!=input2.cols)||
        (input1.cols!=output.cols)||(input1.cols!=output.cols)){
        printf("[Error] myImageAND:  height and width cannot match\n"); 
        system("pause"); 
    } 
    int width = input1.cols;
    int height = input1.rows;
    int depth = input1.channels();
  
#pragma omp parallel for 
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if((input1.data[(j + i*width)] == 255) && input2.data[(j + i*width)] == 255)
			{
				output.data[(j + i*width)] = 255;
			}
			else
			{
				output.data[(j + i*width)] = 0;
			}
		}
	}
} 
  
/************************************************************************/
/*  myRGB2Gray: gray scale                                                     */
/************************************************************************/
void myRGB2Gray(myImage * input, myImage * output) 
{ 
    if ((input->height!=output->height)||(input->width!=output->width)){ 
        printf("[Error] myRGB2Gray:   height and width cannot match\n"); 
        system("pause"); 
    } 
  
    int width = input->width; 
    int height = output->height; 
  
	if(output->depth == 1)
	{
#pragma omp parallel for 
		 for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				output->pixelData[(j + i*width)] = (input->pixelData[(j + i*width)]*0.3) + (input->pixelData[(j + i*width) + (width*height*1)]*0.3) + (input->pixelData[(j + i*width) + (width*height*2)]*0.3);
			} 
		} 
	}
	else
	{
#pragma omp parallel for 
		for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				output->pixelData[(j + i*width)] = (input->pixelData[(j + i*width)]*0.3) + (input->pixelData[(j + i*width) + (width*height*1)]*0.3) + (input->pixelData[(j + i*width) + (width*height*2)]*0.3);
				output->pixelData[(j + i*width) + (width*height*1)] = (input->pixelData[(j + i*width)]*0.3) + (input->pixelData[(j + i*width) + (width*height*1)]*0.3) + (input->pixelData[(j + i*width) + (width*height*2)]*0.3);
				output->pixelData[(j + i*width) + (width*height*2)] = (input->pixelData[(j + i*width)]*0.3) + (input->pixelData[(j + i*width) + (width*height*1)]*0.3) + (input->pixelData[(j + i*width) + (width*height*2)]*0.3);
			} 
		} 
	}
} 
  
/************************************************************************/
/*  myBinary: binarization                                                     */
/************************************************************************/
void myBinary(myImage * input, myImage * output, int threshold) 
{ 
    if ((input->height!=output->height)||(input->width!=output->width)){ 
        printf("[Error] myBinary:   height and width cannot match\n"); 
        system("pause"); 
    }    
      
    int width = input->width; 
    int height = input->height; 
  
	if(output->depth == 1)
	{
#pragma omp parallel for 
		for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				if(input->pixelData[(j + i*width)] > threshold)
				{
					output->pixelData[(j + i*width)] = 255;
				}
				else
				{
					output->pixelData[(j + i*width)] = 0;
				}
			} 
		} 
	}
	else
	{
#pragma omp parallel for 
		for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				if(input->pixelData[(j + i*width)] > threshold)
				{
					output->pixelData[(j + i*width)] = 255;
					output->pixelData[(j + i*width) + (width*height*1)] = 255;
					output->pixelData[(j + i*width) + (width*height*2)] = 255;
				}
				else
				{
					output->pixelData[(j + i*width)] = 0;
					output->pixelData[(j + i*width) + (width*height*1)] = 0;
					output->pixelData[(j + i*width) + (width*height*2)] = 0;
				}
			} 
		} 
	}
} 
  
/************************************************************************/
/*  myInverse: inverse operation                                        */
/************************************************************************/
void myInverse(myImage * input, myImage * output) 
{ 
    if ((input->height!=output->height)||(input->width!=output->width)||(input->depth!=output->depth)){ 
        printf("[Error] myInverse:   height and width and depth cannot match\n"); 
        system("pause"); 
    }    
  
    int width = input->width; 
    int height = input->height; 
  
	if(input->depth == 1)
	{
#pragma omp parallel for 
		for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				if(input->pixelData[(j + i*width)] == 255)
				{
					output->pixelData[(j + i*width)] = 0;
				}
				else
				{
					output->pixelData[(j + i*width)] = 255;
				}
			} 
		} 
	}
	else
	{
#pragma omp parallel for 
		for (int i = 0; i < height; i++){ 
			for (int j = 0; j < width; j++){ 
				if(input->pixelData[(j + i*width)] == 255)
				{
					output->pixelData[(j + i*width)] = 0;
					output->pixelData[(j + i*width) + (width*height*1)] = 0;
					output->pixelData[(j + i*width) + (width*height*2)] = 0;
				}
				else
				{
					output->pixelData[(j + i*width)] = 255;
					output->pixelData[(j + i*width) + (width*height*1)] = 255;
					output->pixelData[(j + i*width) + (width*height*2)] = 255;
				}
			} 
		} 
	}
} 
  
/************************************************************************/
/* myImageCopy:                                                        */
/************************************************************************/
void myImageCopy(myImage * input, myImage * output) 
{ 
    if ((input->height!=output->height)||(input->width!=output->width)){ 
        printf("cannot copy!   height and width cannot match\n"); 
        system("pause"); 
    } 
  
    int width = input->width; 
    int height = input->height; 
    int depth = input->depth; 
  
#pragma omp parallel for 
    for (int i = 0; i < width*height*depth; i++) 
        output->pixelData[i] = input->pixelData[i]; 
  
} 
  
void myImageZero(myImage * input) 
{ 
    int width = input->width; 
    int height = input->height; 
    int depth = input->depth; 
  
#pragma omp parallel for 
    for (int i = 0; i < width*height*depth; i++) 
        input->pixelData[i] = 0; 
} 
  
/************************************************************************/
/* myGet2D: return RGB value                                            */
/************************************************************************/
myColor myGet2D(cv::Mat input, int x, int y)
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
        //input.data[ (x + y*input.cols)] = colors.B;//B
		input.at<uchar>(y, x) = colors.B;
    } 
    else if (depth == 3) 
    { 
        //input.data[ (x + y*input.cols)] = colors.B;//B
        //input.data[ (x + y*input.cols) + (width*height*1)] = colors.G;//G
        //input.data[ (x + y*input.cols) + (width*height*2)] = colors.R;//R
		input.at<cv::Vec3b>(y, x)[0] = colors.B;
		input.at<cv::Vec3b>(y, x)[1] = colors.G;
		input.at<cv::Vec3b>(y, x)[2] = colors.R;
    } 
} 
  
/************************************************************************/
/* opencv_2_myImage: format transformation from IplImage to myImage     */
/************************************************************************/
void opencv_2_myImage(cv::Mat input, myImage * output)
{ 
    if ((input.cols!=output->width)||(input.rows!=output->height)){
        printf("[Error] opencv_2_myImage: size not match!\n"); 
        system("pause"); 
    } 
      
    int width = input.cols;
    int height = input.rows;
    int depth = input.channels();
  
#pragma omp parallel for 
    for (int i = 0; i < width; i++){ 
        for (int j = 0; j < height; j++){ 
            for (int k = 0; k < depth; k++){ 
                output->pixelData[ (i + j*output->width) + (width*height*k) ] = ((uchar *)(input.data + j*input.step))[i*input.channels() + k];//B,G,R
//                output->pixelData[ (i + j*output->width) + (width*height*k) ] = ((uchar *)(input->imageData + j*input->widthStep))[i*input->nChannels + k];//B,G,R
            }
        } 
    } 
} 
  
/************************************************************************/
/* myImage_2_opencv: format transformation from myImage to IplImage       */
/************************************************************************/
void myImage_2_opencv(myImage * input, cv::Mat output)
{ 
    if ((input->width!=output.cols)||(input->height!=output.rows)){
        printf("[Error] myImage_2_opencv: size not match!\n"); 
        system("pause"); 
    } 
  
    int width = input->width; 
    int height = input->height; 
    int depth = input->depth; 
  
#pragma omp parallel for 
    for (int i = 0; i < width; i++){ 
        for (int j = 0; j < height; j++){ 
            for (int k = 0; k < depth; k++){ 
                ((uchar *)(output.data + j*output.step))[i*output.channels() + k] = input->pixelData[ (i + j*output.cols) + (width*height*k) ];//B,G,R
            } 
        } 
    } 
} 
  
/************************************************************************/
/* myResize: resize                                                    */
/************************************************************************/
void myResize(myImage * input, myImage * output) 
{ 
    int scale1 = input->height/output->height; 
    int scale2 = input->width/output->width; 
	int width = output->width; 
    int height = output->height; 
  
    if (scale1!=scale2){ 
        printf("[ERROR] myResize:  different scaling parameter in width and height\n"); 
        system("pause"); 
    } 
  
#pragma omp parallel for 
    for (int i = 0; i < output->height; i++){ 
        for (int j = 0; j < output->width; j++){ 
            //myColor colors = myGet2D(input,i*scale1,j*scale1); 
            //mySet2D(output,colors,i,j); 
			output->pixelData[(j + i*width)] = input->pixelData[(j*scale1 + i*scale1*input->width)];
			output->pixelData[(j + i*width) + (width*height*1)] = input->pixelData[(j*scale1 + i*scale1*input->width) + (input->width*input->height*1)];
			output->pixelData[(j + i*width) + (width*height*2)] = input->pixelData[(j*scale1 + i*scale1*input->width) + (input->width*input->height*2)];
        } 
    } 
} 
  
/************************************************************************/
/* myCreateImage: create an image space                                     */
/************************************************************************/
myImage * myCreateImage(int width, int height, int depth) 
{ 
    myImage * _temp = new myImage(width,height,depth); 
    return _temp; 
} 
  
void mymedianFilter(unsigned char* corrupted, unsigned char* smooth, int width, int height) 
{ 
    memcpy ( smooth, corrupted, width*height*sizeof(unsigned char) );   
    for (int j=1;j<height-1;j++)   
    {   
        for (int i=1;i<width-1;i++)   
        {   
            int k = 0;   
            unsigned char window[9];   
            for (int jj = j - 1; jj < j + 2; ++jj)   
                for (int ii = i - 1; ii < i + 2; ++ii)   
                    window[k++] = corrupted[jj * width + ii];   
            //   Order elements (only half of them)   
            for (int m = 0; m < 5; ++m)   
            {   
                int min = m;   
                for (int n = m + 1; n < 9; ++n)   
                    if (window[n] < window[min])   
                        min = n;   
                //   Put found minimum element in its place   
                unsigned char temp = window[m];   
                window[m] = window[min];   
                window[min] = temp;   
            }   
            smooth[ j*width+i ] = window[4];   
        }   
    }  
} 
  
/************************************************************************/
/* myMedianBlur:                                                       */
/************************************************************************/
void myMedianBlur( myImage * input, myImage * output) 
{ 
    if ((input->height!=output->height)||(input->width!=output->width)){ 
        printf("cannot copy!   height and width cannot match\n"); 
        system("pause"); 
    }    
      
    int width = input->width; 
    int height = input->height; 
    int depth = input->depth; 
  
    myImage * temp; 
    temp = myCreateImage(width,height,1); 
      
    if (depth>1) 
        myRGB2Gray(input,temp); 
    else
        myImageCopy(input,temp); 
  
    mymedianFilter( temp->pixelData, output->pixelData, width, height); 
    temp->myReleaseImage(); 
    delete(temp); 
}

/************************************************************************/
/* myCountNonZero: ?                                                    */
/************************************************************************/
int myCountNonZero(myImage * input) 
{ 
    int width = input->width; 
    int height = input->height; 
	int counter = 0;

#pragma omp parallel for 
    for (int i = 0; i < height; i++){ 
        for (int j = 0; j < width; j++){ 
            if (input->pixelData[(j + i*input->width)] != 0){ 
                counter++;
            } 
        } 
    } 
    return counter; 
} 

/************************************************************************/
/* myAbsDiff:                                                         */
/************************************************************************/
void myAbsDiff(myImage * input1, myImage * input2, myImage * output) 
{ 
    if ((input1->height!=input2->height)||(input1->width!=input2->width)|| 
        (input1->width!=output->width)||(input1->width!=output->width)){ 
        printf("[Error] myImageAND:  height and width cannot match\n"); 
        system("pause"); 
    } 
    int width = input1->width; 
    int height = input1->height; 
    int depth = input1->depth; 
	int temp;
#pragma omp parallel for 
    for (int i = 0; i < width*height*depth; i++){ 
		temp = input1->pixelData[i] - input2->pixelData[i];
		output->pixelData[i] = abs(temp);
    } 
} 

/************************************************************************/
/* myredmask:                                                             */
/************************************************************************/
void myredmask( myImage * input, myImage * mask )
{
	 int width = input->width; 
	 int height = input->height; 

	 for(int i = 0; i < height; i++)
	 {
		 for(int j = 0; j < width; j++)
		 {
			 if(mask->pixelData[(j + i*width)] == 255)
			 {
				 input->pixelData[(j + i*width)] = 0;
				 input->pixelData[(j + i*width) + (width*height*1)] = 0;
			 }
		 }
	 }
}

/************************************************************************/
/* mybluemask:                                                             */
/************************************************************************/
void mybluemask( myImage * input, myImage * mask )
{
	 int width = input->width; 
	 int height = input->height; 

	 for(int i = 0; i < height; i++)
	 {
		 for(int j = 0; j < width; j++)
		 {
			 if(mask->pixelData[(j + i*width)] == 255)
			 {
				 input->pixelData[(j + i*width) + (width*height*1)] = 0;
				 input->pixelData[(j + i*width) + (width*height*2)] = 0;
			 }
		 }
	 }
}

/************************************************************************/
/* mygreenmask:                                                             */
/************************************************************************/
void mygreenmask( myImage * input, myImage * mask )
{
	 int width = input->width; 
	 int height = input->height; 

	 for(int i = 0; i < height; i++)
	 {
		 for(int j = 0; j < width; j++)
		 {
			 if(mask->pixelData[(j + i*width)] == 255)
			 {
				 input->pixelData[(j + i*width)] = 0;
				 input->pixelData[(j + i*width) + (width*height*2)] = 0;
			 }
		 }
	 }
}

void myErode( myImage * input, myImage * output)
{
    for (int j=1;j<input->height-1;j++)  
    {  
        for (int i=1;i<input->width-1;i++)  
        {  
            int k = 0;  
            for (int jj = j - 1; jj < j + 2; ++jj)  
                for (int ii = i - 1; ii < i + 2; ++ii){
                    if ( input->pixelData[jj * input->width + ii] > 127 )
						k++;
				}
			
			if (k==9)
				output->pixelData[ j*input->width+i ] = 255; 
			else
				output->pixelData[ j*input->width+i ] = 0;
        }  
    } 
}

void myDilade( myImage * input, myImage * output)
{
    for (int j=1;j<input->height-1;j++)  
    {  
        for (int i=1;i<input->width-1;i++)  
        {  
            if (input->pixelData[j * input->width + i] > 127)
            {
				for (int jj = j - 1; jj < j + 2; ++jj)  
				 for (int ii = i - 1; ii < i + 2; ++ii){
						output->pixelData[jj * input->width + ii] = 255;
				}
            }
			else
				output->pixelData[j * input->width + i] = 0;
        }  
    } 
}
void myDiladeitself(cv::Mat& input, int n)
{
	int width = input.cols;
	int height = input.rows;
	int depth = input.channels();

	//myImage *temp = myCreateImage(width,height,depth);
	cv::Mat temp;
	if(depth == 1)
    {
	    temp = cv::Mat(cv::Size(input.cols, input.rows), CV_8UC1, cv::Scalar::all(0));
    } else{
        temp = cv::Mat(cv::Size(input.cols, input.rows), CV_8UC3, cv::Scalar::all(0));

    }
	for(int r = 0; r < n; r++)
	{
		for (int j=1;j<input.rows-1;j++)
		{  
			for (int i=1;i<input.cols-1;i++)
			{  
				if (input.data[j * input.cols + i] > 127)
				{
					for (int jj = j - 1; jj < j + 2; ++jj)  
					 for (int ii = i - 1; ii < i + 2; ++ii){
							temp.data[jj * input.cols + ii] = 255;
					}
				}
				//else
					//temp->pixelData[j * input->width + i] = 0;
			}  
		}
		input = temp.clone();
		//myImageCopy(temp,input);
	}
	//temp->myReleaseImage();
}

void myErodeitself( myImage * input,int n)
{
	int width = input->width;
	int height = input->height;
	int depth = input->depth;

	myImage *temp = myCreateImage(width,height,depth);
	for(int r= 0; r < n; r++)
	{
		for (int j=1;j<height-1;j++)  
		{  
			for (int i=1;i<width-1;i++)  
			{  
				int k = 0;  
				for (int jj = j - 1; jj < j + 2; ++jj)  
					for (int ii = i - 1; ii < i + 2; ++ii){
						if ( input->pixelData[jj * width + ii] > 127 )
							k++;
					}
		
				if (k==9)
					temp->pixelData[ j*width+i ] = 255; 
				else
					temp->pixelData[ j*width+i ] = 0;
			}  
		} 
		myImageCopy(temp,input);
	}
	temp->myReleaseImage();
}
