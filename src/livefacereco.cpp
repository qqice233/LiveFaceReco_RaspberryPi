//
// Created by markson zhang
//
//
// Edited by Xinghao Chen 2020/7/27
//
//
// Refactored and edited by Luiz Correia 2021/06/20
#include <unistd.h>
#include <sys/shm.h>
#include <math.h>
#include "livefacereco.hpp"
#include <time.h>
#include "math.hpp"
#include "ParallelVideoCapture/parallel_video_capture.hpp"
#include "mtcnn_new.h"
#include "FacePreprocess.h"
#include "DatasetHandler/image_dataset_handler.hpp"
#include <algorithm>

#define PI 3.14159265
//修改共享内存数据
#define IMAGE_W 640  //图像宽
#define IMAGE_H 480  //图像高
#define IMAGE_TYPE  CV_8UC3           // CV_8UC1 灰度图   CV_8UC3 3通道图像
#define IMAGE_SIZE  IMAGE_W*IMAGE_H*3 //图片像素总大小 CV_8UC1--1通道  CV_8UC3--3通道彩色
#define Shm_address 1209 //共享内存地址标识
// 所有的返回函数必须有返回值 不然调用报错
using namespace std;

double sum_score, sum_fps,sum_confidence;

#define  PROJECT_PATH "/home/pi/LiveFaceReco_RaspberryPi";
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
std::vector<std::string> split(const std::string& s, char seperator)
{
   std::vector<std::string> output;

    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(seperator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );

        output.push_back(substring);

        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word

    return output;
}

void calculateFaceDescriptorsFromDisk(Arcface & facereco,std::map<std::string,cv::Mat> & face_descriptors_map)
{
    const std::string project_path = PROJECT_PATH;
    std::string pattern_jpg = project_path + "/img/*.jpg";
	std::vector<cv::String> image_names;
    
	cv::glob(pattern_jpg, image_names);
    
    int image_number=image_names.size();

	if (image_number == 0) {
		std::cout << "No image files[jpg]" << std::endl;
        std::cout << "At least one image of 112*112 should be put into the img folder. Otherwise, the program will broke down." << std::endl;
        exit(0);
	}
    //cout <<"loading pictures..."<<endl;
    //cout <<"image number in total:"<<image_number<<endl;
    cv::Mat  face_img;
    unsigned int img_idx = 0;

  
    //convert to vector and store into fc, whcih is benefical to furthur operation
	for(auto const & img_name:image_names)
    {
        //cout <<"image name:"<<img_name<<endl;
        auto splitted_string = split(img_name,'/');
        auto splitted_string_2 = splitted_string[splitted_string.size()-1];
        std::size_t name_length = splitted_string_2.find_last_of('_');
        auto person_name =  splitted_string_2.substr(0,name_length);
        //std::cout<<person_name<<"\n";
        face_img = cv::imread(img_name);

        cv::Mat face_descriptor = facereco.getFeature(face_img);

        face_descriptors_map[person_name] = Statistics::zScore(face_descriptor);
        //cout << "now loading image " << ++img_idx << " out of " << image_number << endl;
        printf("\rloading[%.2lf%%]\n",  (++img_idx)*100.0 / (image_number));
    }
   
    cout <<"loading succeed! "<<image_number<<" pictures in total"<<endl;
    
}
void calculateFaceDescriptorsFromImgDataset(Arcface & facereco,std::map<std::string,std::list<cv::Mat>> & img_dataset,std::map<std::string, std::list<cv::Mat>> & face_descriptors_map)
{
    int img_idx = 0;
    const int image_number = img_dataset.size()*5;
    for(const auto & dataset_pair:img_dataset)
    {
        const std::string person_name = dataset_pair.first;

        std::list<cv::Mat> descriptors;
        if (image_number == 0) {
            cout << "No image files[jpg]" << endl;
            return;
        }
        else{
            cout <<"loading pictures..."<<endl;
            for(const auto & face_img:dataset_pair.second)
            {
                cv::Mat face_descriptor = facereco.getFeature(face_img);
                descriptors.push_back( Statistics::zScore(face_descriptor));
                cout << "now loading image " << ++img_idx << " out of " << image_number << endl;
                //printf("\rloading[%.2lf%%]\n",  (++img_idx)*100.0 / (image_number));
            }
            face_descriptors_map[person_name] = std::move(descriptors);
        }
        
    }
}
void loadLiveModel( Live & live )
{
    //Live detection configs
    struct ModelConfig config1 ={2.7f,0.0f,0.0f,80,80,"model_1",false};
    struct ModelConfig config2 ={4.0f,0.0f,0.0f,80,80,"model_2",false};
    vector<struct ModelConfig> configs;
    configs.emplace_back(config1);
    configs.emplace_back(config2);
    live.LoadModel(configs);
}
cv::Mat createFaceLandmarkGTMatrix()
{
    // groundtruth face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1); 
    memcpy(src.data, v1, 2 * 5 * sizeof(float));
    return src.clone();
}
cv::Mat createFaceLandmarkMatrixfromBBox(const Bbox  & box)
{
    float v2[5][2] =
                {{box.ppoint[0], box.ppoint[5]},
                {box.ppoint[1], box.ppoint[6]},
                {box.ppoint[2], box.ppoint[7]},
                {box.ppoint[3], box.ppoint[8]},
                {box.ppoint[4], box.ppoint[9]},
                };
    cv::Mat dst(5, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2 * 5 * sizeof(float));

    return dst.clone();
}

Bbox  getLargestBboxFromBboxVec(const std::vector<Bbox> & faces_info)
{
    if(faces_info.size()>0)
    {
        int lagerest_face=0,largest_number=0;
        for (int i = 0; i < faces_info.size(); i++){
            int y_ = (int) faces_info[i].y2 * ratio_y;
            int h_ = (int) faces_info[i].y1 * ratio_y;
            if (h_-y_> lagerest_face){
                lagerest_face=h_-y_;
                largest_number=i;                   
            }
        }
        
        return faces_info[largest_number];
    }
    return Bbox();
}

LiveFaceBox Bbox2LiveFaceBox(const Bbox  & box)
{
    float x_   =  box.x1;
    float y_   =  box.y1;
    float x2_ =  box.x2;
    float y2_ =  box.y2;
    int x = (int) x_ ;
    int y = (int) y_;
    int x2 = (int) x2_;
    int y2 = (int) y2_;
    struct LiveFaceBox  live_box={x_,y_,x2_,y2_} ;
    return live_box;
}

cv::Mat alignFaceImage(const cv::Mat & frame, const Bbox & bbox,const cv::Mat & gt_landmark_matrix)
{
    cv::Mat face_landmark = createFaceLandmarkMatrixfromBBox(bbox);

    cv::Mat transf = FacePreprocess::similarTransform(face_landmark, gt_landmark_matrix);

    cv::Mat aligned = frame.clone();
    cv::warpPerspective(frame, aligned, transf, cv::Size(96, 112), INTER_LINEAR);
    resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
     
    return aligned.clone();
}

std::string  getClosestFaceDescriptorPersonName(std::map<std::string,cv::Mat> & disk_face_descriptors, cv::Mat face_descriptor, float & face_score)
{
    vector<double> score_(disk_face_descriptors.size());

    std::vector<std::string> labels;

    int i = 0;

    for(const auto & disk_descp:disk_face_descriptors)
    {
        // cout << "comparing with " << disk_descp.first << endl;

        score_[i] = (Statistics::cosineDistance(disk_descp.second, face_descriptor));
        //cout << "score  " << score_[i] << endl;
        labels.push_back(disk_descp.first);
        i++;
    }
    int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin(); 
    int pos = score_[maxPosition]>face_thre?maxPosition:-1;
    //cout << "score_[maxPosition] " << score_[maxPosition] << endl;
    std::string person_name = "";
    if(pos>=0)
    {
        person_name = labels[pos];
        face_score = score_[maxPosition];
    }
    else face_score = 0.0f;
    score_.clear();

    return person_name;
}
std::string  getClosestFaceDescriptorPersonName(std::map<std::string,std::list<cv::Mat>> & disk_face_descriptors, cv::Mat face_descriptor)
{
    vector<std::list<double>> score_(disk_face_descriptors.size());

    std::vector<std::string> labels;

    int i = 0;

    for(const auto & disk_descp:disk_face_descriptors)
    {
        for(const auto & descriptor:disk_descp.second)
        {
            score_[i].push_back(Statistics::cosineDistance(descriptor, face_descriptor));
        }

        labels.push_back(disk_descp.first);
        i++;
    
    }

    int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin();
    
    auto get_max_from_score_list = 
                            [&]()
                            {
                                double max = *score_[maxPosition].begin();
                                for(const auto & elem:score_[maxPosition])
                                {
                                    if(max<elem)
                                    {
                                        max = elem;
                                    }
                                }
                                return max;
                            }; 

    double max = get_max_from_score_list();

    int pos = max>face_thre?maxPosition:-1;

    std::string person_name = "";
    if(pos>=0)
    {
        person_name = labels[pos];
    }
    score_.clear();

    return person_name;
}

int MTCNNDetection()
{
    MyShare::Share_class useShare;
    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "."
    << CV_MINOR_VERSION << "."
    << CV_SUBMINOR_VERSION << endl;
    useShare.Set_ImgFlag(0);
    Arcface facereco;

    // load the dataset and store it inside a dictionary
    //ImageDatasetHandler img_dataset_handler(project_path + "/imgs/");
    //std::map<std::string,std::list<cv::Mat>> dataset_imgs = img_dataset_handler.getDatasetMap();

    //std::map<std::string,std::list<cv::Mat>> face_descriptors_dict;
    std::map<std::string,cv::Mat> face_descriptors_dict;
    //calculateFaceDescriptorsFromImgDataset(facereco,dataset_imgs,face_descriptors_dict);
    calculateFaceDescriptorsFromDisk(facereco,face_descriptors_dict);

    Live live;
    loadLiveModel(live);

    float factor = 0.709f;
    float threshold[3] = {0.7f, 0.6f, 0.6f};

    //ParallelVideoCapture cap("udpsrc port=5000 ! application/x-rtp, payload=96 ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false",cv::CAP_GSTREAMER,30); //using camera capturing
    //ParallelVideoCapture cap("/home/pi/testvideo.mp4");
    ParallelVideoCapture cap(0);                    
    cap.startCapture();

    std::cout<<"okay!\n";

    if (!cap.isOpened()) {
        cerr << "cannot get image" << endl;
        return -1;
    }

    float confidence;
    vector<float> fps;
    float face_score;
    
    Mat frame;
    Mat result_cnn;

    double score, angle;

    cv::Mat face_landmark_gt_matrix = createFaceLandmarkGTMatrix();
    int count = -1;
    std::string liveface;
    std::string person_name;
    float ratio_x = 1;
    float ratio_y = 1;
    int flag = 0;
    int record_count = 0;
    int file_save_count = 0;

    while(cap.isOpened())
    {
        frame = cap.getFrame();    
        if(frame.empty())
        {
            continue;
        } 
        ++count;
        flag = 0;
        //detect faces
        std::vector<Bbox> faces_info = detect_mtcnn(frame); 
        if(faces_info.size()>=1)
        {
            flag = 1;
            auto large_box = getLargestBboxFromBboxVec(faces_info);
            LiveFaceBox live_face_box = Bbox2LiveFaceBox(large_box);
            cv::Mat aligned_img = alignFaceImage(frame,large_box,face_landmark_gt_matrix);
            cv::Mat face_descriptor = facereco.getFeature(aligned_img);
            // normalize
            face_descriptor = Statistics::zScore(face_descriptor);
            person_name = getClosestFaceDescriptorPersonName(face_descriptors_dict,face_descriptor,face_score);
            if(!person_name.empty())
            {
                record_count = 0;
                cout<<person_name<<endl;
            }
            else{
                if (record_face){
                    if (record_count == 10){
                        record_count = 0;
                        cout << "recording new face..." << endl;
                        cout << "input your new face name:(enter q to quit the record process)"<< endl;
                        std::string new_name;
                        std::cin >> new_name;
                        if (new_name == "q"){
                            cout << "recording process stopped." << endl;
                        }
                        else{
                            imwrite(project_path+"/img/"+new_name+"_0.jpg" , aligned_img);
                            face_descriptors_dict[new_name] = face_descriptor;
                        }
                    }
                    else record_count++;
                }
                else cout<<"unknown person"<<"\n";
                person_name = "";
            }
            
            confidence = live.Detect(frame,live_face_box);
            
            if (confidence<=true_thre)
            {
                liveface="Fake face!!";
            }
            else
            {
                putText(frame, person_name, cv::Point(15, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);  
                liveface="True face";
                cv::rectangle(frame, Point(large_box.x1*ratio_x, large_box.y1*ratio_y), Point(large_box.x2*ratio_x,large_box.y2*ratio_y), cv::Scalar(0, 0, 255), 2);
            }
            cout<<liveface<<"\n";
            cv::putText(frame,liveface,cv::Point(15,40),1,2.0,cv::Scalar(255,0,0));
            cv::putText(frame,to_string(confidence),cv::Point(15,10),1,2.0,cv::Scalar(255,0,0));
            cv::putText(frame,to_string(face_score),cv::Point(15,110),1,2.0,cv::Scalar(255,255,0));
        }
        if (flag == 0)
        {
            cout << "no face detected" << endl;
        }
        // if (file_save_count == 50)
        // {
        //     file_save_count = 0;
        //     cv::imwrite("/home/pi/LiveFaceReco_RaspberryPi/temp.jpg", frame);
        // }
        // else file_save_count++;
        
        if(useShare.Get_ImgFlag() ==0){//读取完毕，允许存图
            useShare.pySend_pic2_share_once((uchar*)frame.data,frame.rows,frame.cols);//发送一张图
            useShare.Set_Name(person_name);
            useShare.Set_Confidence(confidence);
            useShare.Set_Score(face_score);
            useShare.Set_ImgFlag(1);//存储完毕，允许读图
        }

        char k = cv::waitKey(33);

        count ++;
    }
    cap.stopCapture();
    return 0;
}