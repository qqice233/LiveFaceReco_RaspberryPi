#ifndef SHARE
#define SHARE
  
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
  
//修改共享内存数据
#define IMAGE_W 640  //图像宽
#define IMAGE_H 480  //图像高
#define IMAGE_TYPE  CV_8UC3           // CV_8UC1 灰度图   CV_8UC3 3通道图像
#define IMAGE_SIZE  IMAGE_W*IMAGE_H*3 //图片像素总大小 CV_8UC1--1通道  CV_8UC3--3通道彩色
#define Shm_address 1209 //共享内存地址标识
// 所有的返回函数必须有返回值 不然调用报错
 
 
namespace MyShare{

    typedef struct ShareData
    {
        int  flag;
        int rows;//图像高
        int cols;//图像宽
        char imgdata[IMAGE_SIZE];//图像数据一维数据，之前用了cv::Mat不行，因为无法在结构体里初始化大小
        char name[20];//人名
        float score;
        float confidence;
    }ShareData_;
  
    class Share_class
    {
        //变量定义
        public:
            //1创建共享内存
            int shmid = shmget((key_t)Shm_address, sizeof(ShareData), 0666|IPC_CREAT);
            //2映射共享内存地址  shm指针记录了起始地址
            void *shm = shmat(shmid, (void*)0, 0);//如果创建一个函数每次调用都执行，需要执行完释放一下shmdt
            //printf("共享内存地址 ： %p\n", (int *)(shm));
  
            //2-1以ShareData结构体类型-访问共享内存
            ShareData *pShareData= (ShareData*)shm;
  
            //用来保存转化后的共享内存的图像结果
            cv::Mat cvoutImg = cv::Mat(IMAGE_H,IMAGE_W,IMAGE_TYPE,cv::Scalar(255, 255, 255));//bufHeight,bufWidth
               
           //未来加速 可以搞一个图像队列 队列大小3  不停存图，然后挨着丢进共享内存，满了就清除。
             
        //函数定义
        public:
            //1初始化执行
            Share_class(){
                printf("共享内存地址 ： %p\n", (int *)(shm));
                //存图要先把图像标志位初始给0，这里给会导致接收端调用开启的时候再次给0覆盖了1导致取图失败。
            }//1构造函数
            //0销毁执行
            ~Share_class() {
                    cout<<"析构函数执行"<<endl;
                    DestroyShare();
            }
            int DestroyShare(){
                //4断开映射 ，保证下次访问不被占用
                shmdt(shm);
                //5释放共享内存地址
                shmctl(shmid, IPC_RMID, 0);
                cout<<"共享内存已经销毁"<<endl;
                return 1;
            }
            int Send_pic2_share_once(cv::Mat Img){
                int i = 0;
                if(pShareData->flag == 0)
                {                   
                //cv::Mat Img=cv::imread("../data_img/1.jpg",cv::IMREAD_COLOR);
                    if(Img.data== nullptr)//nullptr是c++11新出现的空指针常量
                    {
                        printf("图片文件不存在\n");
                        return 0;
                    }
                    //3-1共享内存保存图像宽和高
                    pShareData->rows =Img.rows;
                    pShareData->cols =  Img.cols;
                    //3-2共享内存保存图像数据
                    int size = Img.cols * Img.rows * Img.channels();  
                    char *cvoutImg = (char*)Img.data;
                    memcpy(pShareData->imgdata, cvoutImg, size);
                    //3-3共享内存保存标志位        
                    pShareData->flag = 1;
                    return 1;
                }
                else{
                    return 0;
                }
                //getchar();
            }
            int pySend_pic2_share_once(uchar *frame_data, int height, int width){
                if(pShareData->flag == 0)
                {
                    if(frame_data== nullptr)//nullptr是c++11新出现的空指针常量
                    {
                        printf("图片文件不存在\n");
                        return 0;
                    }                
                    pShareData->cols=width;
                    pShareData->rows=height;
                    int size = height*width*3;
                    memcpy(pShareData->imgdata, frame_data, size);
                    //3-3共享内存保存标志位        
                    pShareData->flag = 1;
                    //printf("数据保存成功 %d\n",pShareData->flag);       
                    return 1;
                }
                else{
                return 0;
                }  
            }
            int  Rec_pic2_data_once()
            {  //cv::Mat &cvoutImg_in
                //3-1共享内存读取标志位    
                if(pShareData->flag == 1)
                {
                    //3-2从共享内存获取图像高和宽
                    int IMAGE_h=pShareData->rows;//从共享内存获取图像高
                    int IMAGE_w=pShareData->cols;//从共享内存获取图像宽
                    //3-3共享内存读取图像数据
                    //cv::Mat cvoutImg = cv::Mat(IMAGE_h,IMAGE_w,CV_8UC3,cv::Scalar(255, 255, 255));//bufHight,bufWidth
                    int size = cvoutImg.cols * cvoutImg.rows * cvoutImg.channels();
                    memcpy((char*)cvoutImg.data, pShareData->imgdata,size);
                    //cv::imshow("RecData_Show",cvoutImg);
                    //cv::waitKey(1);
                    //3-4共享内存修改标志位
                    pShareData->flag = 0;
                    return 1;
                }     
                else{
                    return 0;
                }
            }

            uchar* Img_Cgg2py(){  //uchar* frame_data, int rows, int cols, int channels     
            //将共享内存现有的图像数据发送
                if(pShareData->flag == 1){
                    //  cvoutImg=cv::imread("/home/dongdong/3Code/python2c/1/c++2c++/v4_c++_class_python/img_data/00001.jpg");
                    if(cvoutImg.data== nullptr)//nullptr是c++11新出现的空指针常量
                    {
                        printf("图片文件不存在\n");
                        return 0;
                        }
                        //pShareData->flag = 0;  //等python完成数组转化到图像，在python端完成标志位0
                        return (uchar*)pShareData->imgdata;//这里只试穿了一个数组，瞬间完成
                       // pShareData->flag = 0;  //等python完成数组转化到图像，在python端完成标志位0
                }
            }

            int Get_ImgFlag(){
                return pShareData->flag;
            }
            int Set_ImgFlag(int value){
                pShareData->flag =value;
            }
            int Set_Name(string str){
                strcpy(pShareData->name,"");
                for (int i = 0; i < str.length(); i++)
                {
                    pShareData->name[i] = str[i];
                }
            }
            string Get_Name(){
                string str = "";
                int i = 0;
                while(pShareData->name[i] != '\0'){
                    str += pShareData->name[i];
                    i++;
                }
                return str;
            }
            int Set_Score(float value){
                pShareData->score = value;
            }
            float Get_Score(){
                return pShareData->score;
            }
            int Set_Confidence(float value){
                pShareData->confidence = value;
            }
            float Get_Confidence(){
                return pShareData->confidence;
            }
    };//类定义结束  
}//namespace 定义
  
  
//按照C语言格式重新打包-python调用
//如果调用端开启了线程  这部份无法直接访问到 需要在线程里面重新创建这个类
//如果是单纯的C++调用，不需要这个样子封装
  
extern "C" {
  
    MyShare::Share_class useShare;
  
    int DestroyShare_(){
        useShare.DestroyShare();
    }
    int Send_pic2_share_once_(cv::Mat Img){
        useShare.Send_pic2_share_once(Img);
    }
    int pySend_pic2_share_once_(uchar *frame_data, int height, int width){
        useShare.pySend_pic2_share_once(frame_data,  height, width);
    }
    int  Rec_pic2_data_once_(){
        useShare.Rec_pic2_data_once();
    } 
    uchar* Img_Cgg2py_(){
        useShare.Img_Cgg2py();
    }
    int Get_ImgFlag_(){
        useShare.Get_ImgFlag();
    }
    int Set_ImgFlag_(int value){
        useShare.Set_ImgFlag(value);
    }
    int Set_Name_(string str){
        useShare.Set_Name(str);
    }
    string Get_Name_(){
        useShare.Get_Name();
    }
    int Set_Score_(float value){
        useShare.Set_Score(value);
    }
    float Get_Score_(){
        useShare.Get_Score();
    }
    int Set_Confidence_(float value){
        useShare.Set_Confidence(value);
    }
    float Get_Confidence_(){
        useShare.Get_Confidence();
    }
}
#endif