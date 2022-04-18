#ifndef SHARE
#define SHARE
  
#include "livefacereco.hpp"
#include <unistd.h>
#include <sys/shm.h>
using namespace std;
using namespace cv;
  
//修改共享内存数据
#define IMAGE_W 5472  //图像宽
#define IMAGE_H 3648  //图像高
#define IMAGE_TYPE  CV_8UC3           // CV_8UC1 灰度图   CV_8UC3 3通道图像
#define IMAGE_SIZE  IMAGE_W*IMAGE_H*3 //图片像素总大小 CV_8UC1--1通道  CV_8UC3--3通道彩色
#define Shm_addrees 1209 //共享内存地址标识
// 所有的返回函数必须有返回值 不然调用报错
 
 
namespace MyShare{
  
  
    
  
 
  
     //共享内存-图像
        typedef struct ShareData
    {
        int  flag;
        int rows;//图像高
        int cols;//图像宽
        char imgdata[IMAGE_SIZE];//图像数据一维数据，之前用了cv::Mat不行，因为无法在结构体里初始化大小
        float Gps[4];//保存gps信息 经纬高时间戳
    }ShareData_;
  
  
   // 非共享内存-传送gps混合数据
    typedef struct StructGpsData {
        int flag;
        char *msg;
        float longitude;
        float latitude;
        float high;
        float time;
    } ;
  
  
  
 class Share_class
{
    //变量定义
    public:
  
            //1创建共享内存
            int shmid = shmget((key_t)Shm_addrees, sizeof(ShareData), 0666|IPC_CREAT);
            //2映射共享内存地址  shm指针记录了起始地址
            void *shm = shmat(shmid, (void*)0, 0);//如果创建一个函数每次调用都执行，需要执行完释放一下shmdt
            //printf("共享内存地址 ： %p\n", (int *)(shm));
  
            //2-1以ShareData结构体类型-访问共享内存
            ShareData *pShareData= (ShareData*)shm;
  
            //用来保存转化后的共享内存的图像结果
            cv::Mat cvoutImg = cv::Mat(IMAGE_H,IMAGE_W,IMAGE_TYPE,cv::Scalar(255, 255, 255));//bufHight,bufWidth
               
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
        }   //0结束销毁函数
  
  
        //3销毁共享内存
        int DestroyShare(){
  
            //4断开映射 ，保证下次访问不被占用
            shmdt(shm);
            //5释放共享内存地址
            shmctl(shmid, IPC_RMID, 0);
            cout<<"共享内存已经销毁"<<endl;
            return 1;
  
        }
        /*
        函数功能：   c++-》c++库-》共享内存
            c++ 模式调用,
            c++ 发送图像一张到共享内存 ，
            修改图像标志位1，允许c++和pythoN调用接收函数接受图像。
        函数输入：
            cv::Mat Img   要发送的图像
        函数输出：
            pShareData->flag = 1;   图像标志位
            Mat cvoutImg                           该类图像变量
        */
      
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
                else{
                    //   printf("图片文件存在\n");
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
  
  
       /*
        函数功能：python 或 c++-》c++库-》共享内存
            python 和 c++ 都可调用
            c++ 或python 发送图像一张到共享内存 ，
            修改图像标志位1，允许c++和pythoN调用接收函数接受图像。
              
            如果是python模式通过c++库调用，可开启图像展示验证图像是否完整
        函数输入：
            uchar *frame_data,      要发送图像的图像数据  Mat img.data
            int height,                         图像高
            int width                           图像宽
        函数输出：
            pShareData->flag = 1;   图像标志位
            Mat cvoutImg                    该类图像变量
        */
  
        int pySend_pic2_share_once(uchar *frame_data, int height, int width){
      
          if(pShareData->flag == 0)
            {
                //assert(height*width*3<=IMAGE_SIZE);
  
  
                    if(frame_data== nullptr)//nullptr是c++11新出现的空指针常量
                    {
                        printf("图片文件不存在\n");
                        return 0;
                        }
                    else{
                       //   printf("图片文件存在\n");
                    }
                
                pShareData->cols=width;
                pShareData->rows=height;
                int size = height*width*3;
              
  
                memcpy(pShareData->imgdata, frame_data, size);
  
  
                //3-3共享内存保存标志位        
                pShareData->flag = 1;
                //printf("数据保存成功 %d\n",pShareData->flag);
      
                /*
                //python模式下用来验证发送的图像是否完整。 python-》c++库-》共享内存
                int showimg=0; //如果要显示 打开设置1
                if(!showimg) return 0;
  
                int channel=3;        
                cv::Mat image(height, width, CV_8UC3);
                for (int row = 0; row < height; row++) {
                    uchar *pxvec = image.ptr<uchar>(row);
                    for (int col = 0; col < width; col++) {
                        for (int c = 0; c < channel; c++) {
                            pxvec[col * channel + c] = frame_data[row * width * channel + channel * col + c];
                        }
                    }
                }
      
                cv::imshow("image", image);
                cv::waitKey(3);
             */
 
             return 1;
            }
            else{
 
              return 0;
            }
  
              
        }
  
  
  
       /*
        函数功能： 共享内存 -> c++库-> c++
            C++ 调用
            C++从共享内存读取一张图片
            修改图像标志位0，允许发送端往共享内存存图。
              
        函数输入：
      
        函数输出：
            pShareData->flag = 0;   图像标志位
            Mat cvoutImg                    该类图像变量
        */
  
  
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
            //  printf("数据跟新一次\n");
                //3-4共享内存修改标志位
                pShareData->flag = 0;
                return 1;
            }     
            else{
 
                return 0;
 
            }
        }
  
    
  
       /*
        函数功能： 共享内存 -> c++库-> python 或 c++
            python 调用（C++用Rec_pic2_data_once）  主要是给python用获取共享内存的图像
            python从共享内存读取一张图片
            修改图像标志位0，允许发送端往共享内存存图。
              
        函数输入：
             调用前先用c++端更新共享内存最新的图像
      
        函数输出：
            pShareData->flag = 0;                     图像标志位
            (uchar*)cvoutImg.data                  该类图像变量cvoutImg的数据data指针  
        */
  
uchar* Img_Cgg2py(){  //uchar* frame_data, int rows, int cols, int channels
                 
               
            //将共享内存现有的图像数据发送
                if(pShareData->flag == 1){
                    //  cvoutImg=cv::imread("/home/dongdong/3Code/python2c/1/c++2c++/v4_c++_class_python/img_data/00001.jpg");
                  
  
                    if(cvoutImg.data== nullptr)//nullptr是c++11新出现的空指针常量
                    {
                        printf("图片文件不存在\n");
                        return 0;
                        }
                    else{
                       //   printf("图片文件存在\n");
                    }
  
                        //pShareData->flag = 0;  //等python完成数组转化到图像，在python端完成标志位0
                        return (uchar*)pShareData->imgdata;//这里只试穿了一个数组，瞬间完成
  
                       // pShareData->flag = 0;  //等python完成数组转化到图像，在python端完成标志位0
                }
                //重新建立新的数据发送模式
                /*
              
                Mat image =cv::imread("/home/dongdong/3Code/python2c/1/c++2c++/v4_c++_class_python/img_data/00001.jpg");
  
                
                if (!image.empty()) {
  
                //cout<<  "cgg2py new pic"<<endl;
                //cv::imshow("cgg", image);
                //cv::waitKey(0);
  
                int rows = image.rows;
                int cols = image.cols;
                int channels = image.channels();
                // printf("rows = %d cols = %d channels = %d size = %d\n", rows, cols, channels, rows*cols*channels);
                uchar *data = (uchar*)malloc(sizeof(uchar) * rows * cols * channels);
                memcpy(data, image.data, rows * cols * channels);
                return data;
                }
  
                */
}
  
  
       /*
        函数功能： 共享内存 -> c++库-> python 或 c++
            python 调用（C++直接通过类变量引用）  主要是给python通过函数方式用获取共享内存的int flag数据
            获取图像标志位,用于判断是否可以读写
          
        函数输入：
              
        函数输出：
            pShareData->flag = 0 or 1;                     图像标志位
            
        */
  
        //4-1获取图像保存标志位
         int Get_ImgFlag(){
             return pShareData->flag ;
         }
  
        /*
        函数功能： 共享内存 -> c++库-> python 或 c++
            python 调用（C++直接通过类变量引用）  主要是给python通过函数方式用修改共享内存的int flag数据
            设置图像标志位,用于开启是否可以读写
          
        函数输入：
            int value                   pythoN需要将数据转化为   ctypes.c_int  送进来 默认int不需要
        函数输出：
            pShareData->flag = 0 or 1;           图像标志位    pythoN需要将数据转化为   ctypes.c_int  接受 默认int不需要
            
        */
  
         int Set_ImgFalg(int value){
            pShareData->flag =value;
         }
  
  
  
  
        /*
        函数功能：  python send -> c++库 ->  共享内存 -> c++库-> python rec
            python 调用 （c++端直接通过类的变量引用就可以）
            python 修改共享内存中的gps数据
          
        函数输入：
           float *data      python的数组索引 和 类型
           int len               python的数组长度
          
        函数输出：
            float result      操作结果 可不使用（python接收端需要指明接收数据类型  c float指针 ctypes.POINTER(ctypes.c_float)  ）
            
        */
  
  
        //5传输数组  接受py数组并修改python原数组,返回总和结果
        float pyarr_set_cgg_gps_share(float *data, int len) {
            float result=1;
            for (int i = 0; i < len; i++) {        
                     pShareData->Gps[i]=data[i] ;
                }
              return result;
                
            }
  
  
  
        /*
        函数功能：  python -> c++库 ->  共享内存 -> c++库-> python
            python 调用 （c++端直接通过类的变量引用就可以）
            python 获取共享内存中的gps数据
          
        函数输入：
           float  pShareData->Gps[4]              C++ 共享内存结构体pShareData中的GPS数据
          
        函数输出：
            (uchar*) pShareData->Gps;       （python接收端需要指明接收数据类型  c float指针 ctypes.POINTER(ctypes.c_float)  ）
            
        */
  
  
  
    uchar* py_get_cgg_gps_share(){
         // c++发送端调用其他函数更新GPS数据,保存在共内存（简单举例）
         //pShareData->Gps[0]=1.56;
         //pShareData->Gps[1]=2.34;
         //pShareData->Gps[2]=3.14;
         //pShareData->Gps[3]=4.78;
  
         return (uchar*) pShareData->Gps; //返回指针
    }
      
       
        /*
        函数功能：  python -> c++库 ->  共享内存 -> c++库-> python
            python 调用 （c++端直接通过类的变量引用就可以）
            python 获取共享内存中的gps数据  python传送过来一个结构体，修改pytho原结构体返回
          
        函数输入：
           float  pShareData->Gps[4]              C++ 共享内存结构体pShareData中的GPS数据
           StructGpsData gps                             C++的结构体- python需要将对应的python结构体输入
          
        函数输出：
            StructGpsData gps       （python接收端需要指明接收数据类型     ）
            
        */
  
  
  
    StructGpsData py_get_cgg_gps_Struct( StructGpsData gps){
       // c++发送端调用其他函数更新GPS数据,保存在共内存（简单举例）
         //pShareData->Gps[0]=1.56;
         //pShareData->Gps[1]=2.34;
         //pShareData->Gps[2]=3.14;
         //pShareData->Gps[3]=4.78;
        
      //共享内存数据更新gps数据
         gps.flag=1;
         gps.msg="new share data from c++  share";
         gps.longitude=pShareData->Gps[0];
         gps.latitude=pShareData->Gps[1];
         gps.high=pShareData->Gps[2];
         gps.time=pShareData->Gps[3];
         return gps;
    }
     
  
       
  
        /*
        函数功能：  python -> c++库 ->  共享内存
            python 调用 （c++端直接通过类的变量引用就可以）
            python 修改共享内存中的gps数据  python传送过来一个结构体，修改C++原结构体
          
        函数输入：
           float  pShareData->Gps[4]              C++ 共享内存结构体pShareData中的GPS数据
           StructGpsData gps                             C++的结构体- python需要将对应的python结构体输入
          
        函数输出：
            StructGpsData gps       （python接收端需要指明接收数据类型     ）
            
        */
  
    StructGpsData py_Set_cgg_gps_Struct( StructGpsData gps){
       // c++发送端调用其他函数更新GPS数据,保存在共内存（简单举例）
         gps.flag=1;
         gps.msg="new share have set share c++";
         pShareData->Gps[0]=gps.longitude;
         pShareData->Gps[1]=gps.latitude;
         pShareData->Gps[2]= gps.high;
         pShareData->Gps[3]=gps.time;
        
      
  
         return gps;
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
  
  
    int Set_ImgFalg_(int value){
  
        useShare.Set_ImgFalg(value);
    }
  
          
   float pyarr_set_cgg_gps_share_(float *data, int len) {
        useShare.pyarr_set_cgg_gps_share( data,  len);
    }
  
  
    uchar*  py_get_cgg_gps_share_(){
                useShare.py_get_cgg_gps_share();
      }
  
  
  
   MyShare::StructGpsData py_get_cgg_gps_Struct_( MyShare::StructGpsData gps){
             return useShare.py_get_cgg_gps_Struct(gps);
    }
  
  
    MyShare::StructGpsData py_Set_cgg_gps_Struct_(MyShare::StructGpsData gps){
            return useShare.py_Set_cgg_gps_Struct(gps);
    }
  
  
  
}
  
  
  
  
  
  
  
  
#endif