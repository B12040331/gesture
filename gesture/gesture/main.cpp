
/*
中值滤波->RGB-HSV->肉色检测->腐蚀/膨胀->侦测
*/

#include "cv.h"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>
using namespace std;

#define WNDCOUNT 3
#define sample_count (frame_count<100?frame_count:100)

char* wndname[7] = { "源图像", "图像1",
"图像2", "图像3",
"图像4", "图像5",
"图像6"
};

int wndposition[7][2] = { { 10, 0 }, { 335, 0 }, { 660, 0 },
{ 10, 260 }, { 335, 260 }, { 660, 260 },
{ 1080, 0 }
};

float hu[100000][7];
int ii = 0;
int n;
float hu11[10000][7];

void SkinRGB(IplImage* rgb, IplImage* _dst);
void cvSkinHSV(IplImage* src, IplImage* dst);
void getHandContour(CvSeq* c, CvSeq** hc);
int simplyConvexHull(CvSeq* h, CvPoint* pts);
void drawConvexHullArray(IplImage* src, CvPoint* pts, int count);
int getConvexityDefectArray(CvSeq* h, CvPoint* pts);
void PrintMat(CvMat *A);
void createtemplate(float *p_hu, int RowCount);
double oshi(float x[][7], float y[][7]);

int main(int argc, char* argv[]) {
	int width, height;
	int count;
	int key;
	int FileCount;
	int CurrentFileNum;

	char filename[200];
	char* XMLfile;
	IplImage* pFrame;
	IplImage* pFrImg;
	
	IplImage* tempImg[10];
	int HandPtCount;
	int HandPt2Count;
	CvPoint HandPt[20];
	CvPoint HandPt2[20];

	CvCapture* pCapture;
	int i, j;
	unsigned char* pSource;
	unsigned char* pResult;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 3, 8);

	int thresh;

	if (argc > 1) {
		XMLfile = argv[1];
		printf("配置XML文件名：%s\r\n", XMLfile);
	}
	else {
		printf("输入参数错误");
		return -1;
	}

	CvFileStorage* fs = cvOpenFileStorage(XMLfile, 0, CV_STORAGE_READ);//读取配置文件
	FileCount = cvReadIntByName(fs, 0, "file_count", 1);              //查询文件节点返回他的值
	CvSeq* s;
	if (1 == FileCount) {
		strcpy(filename, cvReadStringByName(fs, 0, "file_names")); //修改filename大小
		CurrentFileNum = FileCount;
	}
	else if (FileCount>1) {
		s = cvGetFileNodeByName(fs, 0, "file_names")->data.seq;
		strcpy(filename, cvReadString((CvFileNode*)cvGetSeqElem(s, 0)));
		CurrentFileNum = 1;
	}
	else {
		printf("配置文件错误！\n");
		return -1;
	}

	printf("输入文件名：%s\r\n", filename);

	pCapture = cvCaptureFromAVI(filename);        //从文件读取视频
	if (!pCapture) {
		printf("视频没有找到或者无法播放！\n");
		getchar();
		return -1;
	}

	pFrame = cvQueryFrame(pCapture);              //读取一帧

	pFrImg = cvCreateImage(cvSize(pFrame->width / 2, pFrame->height / 2), 8, 3);  //创建图像指针，尺寸减半
	cvResize(pFrame, pFrImg);                                                  //尺寸变换

	width = pFrImg->width;
	height = pFrImg->height;

	//定义存储训练样本hu矩数据的矩阵
	int frame_count = (int)cvGetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_COUNT);
	printf("frame_count=%d\n", frame_count);

	// Load the source image

	// Create a window
	for (i = 0; i<WNDCOUNT; i++) {
		cvNamedWindow(wndname[i], 1);
		cvMoveWindow(wndname[i], wndposition[i][0], wndposition[i][1]);
	}

	tempImg[0] = cvCreateImage(cvSize(width, height), 8, 3);
	tempImg[1] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[2] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[3] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[4] = cvCreateImage(cvSize(width, height), 8, 3);
	tempImg[5] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[6] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[7] = cvCreateImage(cvSize(width, height), 8, 1);
	tempImg[8] = cvCreateImage(cvSize(width, height), 8, 1);

	CvMemStorage* storage = cvCreateMemStorage(0);  //  创建内存块
	CvSeq* contour = NULL;
	CvSeq* HandContour = NULL;

	printf("处理开始啦\n");
	
	do {
		cvCvtColor(pFrImg, tempImg[0], CV_BGR2HSV);//色彩空间转换
		//cvShowImage("hsv", tempImg[0]);
		cvSplit(tempImg[0], NULL, NULL, tempImg[1], NULL);//分割提取单一通道
		//cvShowImage("split", tempImg[0]);
		cvSmooth(tempImg[1], tempImg[1], CV_MEDIAN, 3, 0, 0, 0);//中值滤波
		//cvShowImage("mdeian 1", tempImg[1]);
		//getchar();

		for (int i = 0; i < tempImg[1]->height; ++i) {
			char * ch = tempImg[1]->imageData + i * tempImg[1]->widthStep;
			for (int j = 0; j < tempImg[1]->width; ++j) {
				if (ch[0] >= 7 && ch[0] <= 29) {
					ch += 3;
					continue;
				}
				ch[0] = ch[1] = ch[2] = 0;
			}
		}
		cvShowImage("skin", tempImg[1]);
		
		cvThreshold(tempImg[1], tempImg[2], NULL, 255, CV_THRESH_OTSU);//Otsu法阈值分割？
		cvCanny(tempImg[2], tempImg[3], 10, 240, 3);   //边沿检测上下限分别为240、10,3是什么？
		cvFindContours(tempImg[2], storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //查找最外层轮廓，只保留末端的象素点;
		cvCopy(pFrImg, tempImg[4]);   //复制图像???????
		getHandContour(contour, &HandContour);
		//画出轮廓
		cvDrawContours(pFrImg, contour, CV_RGB(255, 0, 0), CV_RGB(255, 0, 0), 0, 1, 8);
		int L = cvArcLength(HandContour);//计算轮廓周长
		printf("轮廓长度为 %d \n", L);//409

		//检查是否存在缺陷
		if (cvCheckContourConvexity(HandContour) == 0) {//计算轮廓是否为凸
			//计算外接多边形
			CvSeq* hull = cvConvexHull2(HandContour, NULL, CV_CLOCKWISE, 0);

			//提取外接多边形
			HandPtCount = simplyConvexHull(hull, HandPt);

			//画外接多边形
			drawConvexHullArray(tempImg[4], HandPt, HandPtCount);

			//提取内部缺陷
			CvSeq* HandContour2 = cvConvexityDefects(HandContour, hull, NULL);
			HandPt2Count = getConvexityDefectArray(HandContour2, HandPt2);
			//printf("缺陷数量 = %d \n", count);

			for (i = 0; i< HandPt2Count; i++) {
				cvCircle(tempImg[4], HandPt2[i], 3, CV_RGB(255, 0, 0), CV_FILLED); //绘制红圆点
			}

		}
		else {
			printf("不存在缺陷\n");
		}

		cvZero(tempImg[5]);//所有元素置零
		cvDrawContours(tempImg[5], HandContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, 2, 8);//绘制轮廓

		CvMoments moments;
		CvHuMoments Hu;
		//Hu矩计算
		cvMoments(tempImg[5], &moments, 0);
		cvGetHuMoments(&moments, &Hu);
		printf("h1=%.3e \t h2=%.3e \t h3=%.3e \t h4=%.3e \t\n h5=%.3e \t h6=%.3e \t h7=%.3e\n", Hu.hu1, Hu.hu2, Hu.hu3, Hu.hu4, Hu.hu5, Hu.hu6, Hu.hu7);

		/////显示//////////////////////////////////////////////////////////////////////计算hu的对数再显示

		//cvShowImage(wndname[0],pFrImg);
		//cvShowImage(wndname[1],tempImg[4]);
		//cvShowImage(wndname[2],tempImg[5]);
		//cvShowImage(wndname[3],tempImg[3]);
		//cvShowImage(wndname[6],tempImg6);

		// Wait for a key stroke; the same function arranges events processing
		//cvWaitKey(0);
		int jj = (int)*argv[2];
		switch (jj) {
			case 's': {
					if (ii<100000) {
						hu[ii][0] = (float)Hu.hu1;
						hu[ii][1] = (float)Hu.hu2;
						hu[ii][2] = (float)Hu.hu3;
						hu[ii][3] = (float)Hu.hu4;
						hu[ii][4] = (float)Hu.hu5;
						hu[ii][5] = (float)Hu.hu6;
						hu[ii][6] = (float)Hu.hu7;
						//Hu 是hu结构的指针
						// hu[ii]= {(float)Hu.hu1,(float)Hu.hu2,(float)Hu.hu3,
						// (float)Hu.hu4,(float)Hu.hu5,(float)Hu.hu6
						// };

						//                            printf("uh1=%.3f \t uh2=%.3f \t uh3=%.3f \t uh4=%.3f \t\n uh5=%.3f \t uh6=%.3f \t \n", hu[ii][0],
						//                                hu[ii][1], hu[ii][2], hu[ii][3] , hu[ii][4], hu[ii][5]);
						ii++;
					} else {
						//createtemplate((float *)hu, 2);
						return 0;
					}
					break;
			}

			case 'r': {
					if (ii < 100000) {
						hu[ii][0] = (float)Hu.hu1;
						hu[ii][1] = (float)Hu.hu2;
						hu[ii][2] = (float)Hu.hu3;
						hu[ii][3] = (float)Hu.hu4;
						hu[ii][4] = (float)Hu.hu5;
						hu[ii][5] = (float)Hu.hu6;
						hu[ii][6] = (float)Hu.hu7;
						//Hu 是hu结构的指针
						ii++;
					}
					double r[10];
					double min;
					//int ii=0;
					//double r0,r1,r2,r3,r4,r5,r6,r7,r8,r9;
					float hu0[1][7], hu1[1][7], hu2[1][7];
					float hu3[1][7], hu4[1][7], hu5[1][7];
					float hu6[1][7], hu7[1][7], hu8[1][7];
					float hu9[1][7], hu10[1][7];
					//CvMat *A0,*A1,*A2,*A3,*A4,*A5,*A6,*A7,*A8,*A9;

					CvMat* A0 = (CvMat*)cvLoad("template0.xml");
					for (int i = 0; i< A0->rows; i++) {
						for (int j = 0; j<6; j++) {
							hu0[i][j] = CV_MAT_ELEM(*A0, float, i, j);
						}
					}
					//hu0[i][j]=((float*)(A0->data.ptr + A0->step*i))[j];
					CvMat A = cvMat(ii, 7, CV_32F, &hu);
					CvMat *B = cvCreateMat(1, 7, CV_32F);
					cvReduce(&A, B, 0, CV_REDUCE_AVG);
					for (int i = 0; i<1; i++) {
						for (int j = 0; j<7; j++) {
							hu10[i][j] = CV_MAT_ELEM(*B, float, i, j);
						}
					}

					CvMat* A1 = (CvMat*)cvLoad("template1.xml");
					for (int i = 0; i<A1->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu1[i][j] = CV_MAT_ELEM(*A1, float, i, j);
						}
					}
					//hu1[i][j]=((float*)(A1->data.ptr + A1->step*i))[j];

					CvMat* A2 = (CvMat*)cvLoad("template2.xml");
					for (int i = 0; i<A2->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu2[i][j] = CV_MAT_ELEM(*A2, float, i, j);
						}
					}
					// hu2[i][j]=((float*)(A2->data.ptr + A2->step*i))[j];

					CvMat* A3 = (CvMat*)cvLoad("template3.xml");
					for (int i = 0; i<A3->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu3[i][j] = CV_MAT_ELEM(*A3, float, i, j);
						}
					}
					//hu3[i][j]=((float*)(A3->data.ptr + A3->step*i))[j];

					CvMat* A4 = (CvMat*)cvLoad("template4.xml");
					for (int i = 0; i<A4->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu4[i][j] = CV_MAT_ELEM(*A4, float, i, j);
						}
					}
					//hu4[i][j]=((float*)(A4->data.ptr + A4->step*i))[j];

					CvMat* A5 = (CvMat*)cvLoad("template5.xml");
					for (int i = 0; i<A5->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu5[i][j] = CV_MAT_ELEM(*A5, float, i, j);
						}
					}
					// hu5[i][j]=((float*)(A5->data.ptr + A5->step*i))[j];

					CvMat* A6 = (CvMat*)cvLoad("template6.xml");
					for (int i = 0; i<A6->rows; i++) {
						for (int j = 0; j<7; j++) {
							hu6[i][j] = CV_MAT_ELEM(*A6, float, i, j);
						}
					}
					//hu6[i][j]=((float*)(A6->data.ptr + A6->step*i))[j];

					CvMat* A7 = (CvMat*)cvLoad("template7.xml");
					for (int i = 0; i<A7->rows; i++) {
						for (int j = 0; j<6; j++) {
							hu7[i][j] = CV_MAT_ELEM(*A7, float, i, j);
						}
					}
					//hu7[i][j]=((float*)(A7->data.ptr + A7->step*i))[j];

					CvMat* A8 = (CvMat*)cvLoad("template8.xml");
					for (int i = 0; i<A8->rows; i++) {
						for (int j = 0; j<6; j++) {
							hu8[i][j] = CV_MAT_ELEM(*A8, float, i, j);
						}
					}
					//hu8[i][j]=((float*)(A8->data.ptr + A8->step*i))[j];


					CvMat* A9 = (CvMat*)cvLoad("template9.xml");
					for (int i = 0; i<A9->rows; i++) {
						for (int j = 0; j<6; j++) {
							hu9[i][j] = CV_MAT_ELEM(*A9, float, i, j);
						}
					}

					r[0] = oshi(hu10, hu0);
					r[1] = oshi(hu10, hu1);
					r[2] = oshi(hu10, hu2);
					r[3] = oshi(hu10, hu3);
					r[4] = oshi(hu10, hu4);
					r[5] = oshi(hu10, hu5);
					r[6] = oshi(hu10, hu6);
					r[7] = oshi(hu10, hu7);
					r[8] = oshi(hu10, hu8);
					r[9] = oshi(hu10, hu9);
					min = r[0];
					for (int i = 1; i<10; i++) {
						if (min>r[i]) {
							min = r[i];
						}
					}

					if (min == r[0])
						cvPutText(tempImg[4], "zero", cvPoint(200, 200), &font, cvScalar(255, 0, 0));
					if (min == r[1])
						cvPutText(tempImg[4], "one", cvPoint(200, 200), &font, cvScalar(0, 255, 0));
					if (min == r[2])
						cvPutText(tempImg[4], "two", cvPoint(200, 200), &font, cvScalar(255, 255, 0));
					if (min == r[3])
						cvPutText(tempImg[4], "three", cvPoint(200, 200), &font, cvScalar(255, 0, 255));
					if (min == r[4])
						cvPutText(tempImg[4], "four", cvPoint(200, 200), &font, cvScalar(255, 0, 0));
					if (min == r[5])
						cvPutText(tempImg[4], "five", cvPoint(200, 200), &font, cvScalar(0, 255, 0));
					if (min == r[6])
						cvPutText(tempImg[4], "six", cvPoint(200, 200), &font, cvScalar(255, 255, 0));
					if (min == r[7])
						cvPutText(tempImg[4], "seven", cvPoint(200, 200), &font, cvScalar(255, 0, 255));
					if (min == r[8])
						cvPutText(tempImg[4], "eight", cvPoint(200, 200), &font, cvScalar(255, 255, 0));
					if (min == r[9])
						cvPutText(tempImg[4], "nine", cvPoint(200, 200), &font, cvScalar(255, 0, 255));

					break;
		}
		defadult:
			break;
		}
		if (cvWaitKey(30) >= 0) break;
		pFrame = cvQueryFrame(pCapture);
		if (pFrame) {
			cvResize(pFrame, pFrImg);
		}
		else {
			//提取下一个文件
			if (CurrentFileNum<FileCount) {
				strcpy(filename, cvReadString((CvFileNode*)cvGetSeqElem(s, CurrentFileNum)));
				CurrentFileNum++;

				pCapture = cvCaptureFromAVI(filename);
				if (!pCapture) {
					printf("视频没有找到或者无法播放！\n");
					getchar();
					return -1;
				}
				pFrame = cvQueryFrame(pCapture);
			}
			else {
				break;
			}
		}
		cvShowImage(wndname[0], pFrImg);
		cvShowImage(wndname[1], tempImg[4]);
		cvShowImage(wndname[2], tempImg[5]);
	}

	while (1);
	int nn = (int)*argv[2];
	if ('s' == nn) {
		createtemplate((float *)hu, ii);
		printf("frame_count=%d\n", frame_count);

	}
	
	printf("处理结束啦\n");
	if (!pCapture)
		cvReleaseCapture(&pCapture);

	cvReleaseImage(&pFrImg);

	for (i = 0; i < 10; i++) {
		if (!tempImg[i]) cvReleaseImage(&tempImg[i]);
	}

	for (i = 0; i<WNDCOUNT; i++) {
		cvDestroyWindow(wndname[i]);
	}
	CvMat* mat = cvCreateMat(3, 3, CV_32FC1);
	cvReleaseMemStorage(&storage);
	cvReleaseFileStorage(&fs);

	return 0;
}

// skin region location using rgb limitation  
void SkinRGB(IplImage* rgb, IplImage* _dst)
{
	assert(rgb->nChannels == 3 && _dst->nChannels == 3);

	static const int R = 2;
	static const int G = 1;
	static const int B = 0;

	IplImage* dst = cvCreateImage(cvGetSize(_dst), 8, 3);
	cvZero(dst);

	for (int h = 0; h<rgb->height; h++) {
		unsigned char* prgb = (unsigned char*)rgb->imageData + h*rgb->widthStep;
		unsigned char* pdst = (unsigned char*)dst->imageData + h*dst->widthStep;
		for (int w = 0; w<rgb->width; w++) {
			if ((prgb[R]>95 && prgb[G]>40 && prgb[B]>20 &&
				prgb[R] - prgb[B]>15 && prgb[R] - prgb[G]>15/*&&
															!(prgb[R]>170&&prgb[G]>170&&prgb[B]>170)*/) ||//uniform illumination   
															(prgb[R]>200 && prgb[G]>210 && prgb[B]>170 &&
															abs(prgb[R] - prgb[B]) <= 15 && prgb[R]>prgb[B] && prgb[G]>prgb[B])//lateral illumination  
															) {
				memcpy(pdst, prgb, 3);
			}
			prgb += 3;
			pdst += 3;
		}
	}
	cvCopy(dst, _dst);
	cvReleaseImage(&dst);
}

void cvSkinHSV(IplImage* src, IplImage* dst)
{
	IplImage* hsv = cvCreateImage(cvGetSize(src), 8, 3);
	//IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
	//IplImage* cb=cvCreateImage(cvGetSize(src),8,1);  
	cvCvtColor(src, hsv, CV_BGR2HSV);
	//cvSplit(ycrcb,0,cr,cb,0);  

	static const int V = 2;
	static const int S = 1;
	static const int H = 0;

	//IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);  
	cvZero(dst);

	for (int h = 0; h<src->height; h++) {
		unsigned char* phsv = (unsigned char*)hsv->imageData + h*hsv->widthStep;
		unsigned char* psrc = (unsigned char*)src->imageData + h*src->widthStep;
		unsigned char* pdst = (unsigned char*)dst->imageData + h*dst->widthStep;
		for (int w = 0; w<src->width; w++) {
			if (phsv[H] >= 7 && phsv[H] <= 29)
			{
				memcpy(pdst, psrc, 3);
			}
			phsv += 3;
			psrc += 3;
			pdst += 3;
		}
	}
	//cvCopyImage(dst,_dst);  
	//cvReleaseImage(&dst);  
}

void getHandContour(CvSeq* c, CvSeq** hc) {
	while (c != NULL)
	{
		CvRect r = ((CvContour*)c)->rect;
		if (r.height*r.width>10000)
		{
			*hc = c;
		}
		c = c->h_next;
	}

}


int simplyConvexHull(CvSeq* h, CvPoint* pts)
{
	int i, j;
	int count = h->total;
	CvPoint** pt0 = (CvPoint**)(cvGetSeqElem(h, count - 1));
	for (i = 0, j = 0; i<count && j<20; i++)
	{
		CvPoint** pt = (CvPoint**)(cvGetSeqElem(h, i));
		//printf("x = %d \n", (*pt)->x);
		//排除相邻点
		if ((abs((*pt)->x - (*pt0)->x) + abs((*pt)->y - (*pt0)->y)) >20)
		{
			(pts + j)->x = (*pt)->x;
			(pts + j)->y = (*pt)->y;
			j++;
			pt0 = pt;
		}
	}
	return j;
}


void drawConvexHullArray(IplImage* src, CvPoint* pts, int count)
{
	int i;
	cvCircle(src, *(pts + count - 1), 3, CV_RGB(0, 255, 0), CV_FILLED);
	cvLine(src, *pts, *(pts + count - 1), CV_RGB(0, 255, 0), 1, 8, 0);
	for (i = 0; i< count - 1; i++)
	{
		cvCircle(src, *(pts + i), 3, CV_RGB(0, 255, 0), CV_FILLED);
		cvLine(src, *(pts + i), *(pts + i + 1), CV_RGB(0, 255, 0), 1, 8, 0);
	}
}


int getConvexityDefectArray(CvSeq* h, CvPoint* pts)        //计算凹缺陷数
{
	int i, j;
	int count = h->total;
	//printf("缺陷数量 = %d \n", count);

	for (i = 0, j = 0; i<count; i++)
	{
		CvConvexityDefect* cd = (CvConvexityDefect*)cvGetSeqElem(h, i);
		if ((cd) && (cd->depth>20))
		{
			CvPoint* pt = cd->depth_point;
			(pts + j)->x = pt->x;
			(pts + j)->y = pt->y;
			j++;
		}
	}
	return j;
}


void PrintMat(CvMat *A)
{
	int i, j;
	//printf("\nMatrix=:");
	for (i = 0; i<A->rows; i++)
	{
		printf("\n");
		switch (CV_MAT_DEPTH(A->type))
		{
		case CV_32F:
		case CV_64F:
			for (j = 0; j<A->cols; j++)
				printf("%9.3f", (float)cvGetReal2D(A, i, j));
			break;
		case CV_8U:
		case CV_16U:
			for (j = 0; j<A->cols; j++)
				printf("%6d", (int)cvGetReal2D(A, i, j));
			break;
		default:
			break;
		}
	}
	printf("\n");
}


//计算样本hu矩数据的k均值并存储为xml文件作为样本模型
//void createtemplate(float p_hu[100][6])
void createtemplate(float *p_hu, int RowCount)
{

	CvMat A = cvMat(RowCount, 7, CV_32F, p_hu);
	CvFileStorage* fsw = cvOpenFileStorage("template2.xml", 0, CV_STORAGE_WRITE);
	CvMat *B = cvCreateMat(1, 7, CV_32F);

	cvReduce(&A, B, 0, CV_REDUCE_AVG);
	cvWrite(fsw, "Avg", B);

	cvReleaseFileStorage(&fsw);
}

double oshi(float x[][7], float y[][7])
{


	double ODistance = 0;
	for (int i = 0; i<1; i++)
	{
		for (int j = 0; j<6; j++)
			ODistance = ODistance + (x[i][j] - y[i][j])*(x[i][i] - y[i][j]);
	}
	ODistance = ODistance / 2;
	ODistance = sqrt(ODistance);
	return ODistance;
}
