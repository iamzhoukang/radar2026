#pragma once

#include "MvCameraControl.h"
#include "../sdk.h"

// #include <iostream>
// #include <opencv2/opencv.hpp>

namespace sdk
{
    class HikCamera : public Camera
    {
    private:
        /* data */
        int nRet = MV_OK;
        void * camera_handle_ = nullptr;//相机实例
        size_t error_num = 0;
        size_t nDeviceNum = 0;//当前设备数量

        MV_IMAGE_BASIC_INFO img_info_;
        MV_FRAME_OUT raw;
        MV_CC_PIXEL_CONVERT_PARAM cvt_param;
        MV_CC_PIXEL_CONVERT_PARAM convert_param_;
        MV_CC_DEVICE_INFO_LIST device_list;

    public:
        static bool CameraSDKInit() {
            int nRetinit = MV_OK;
            MV_CC_Initialize();  //初始化sdk
            if(nRetinit != MV_OK) {
                std::cout<<"hik init failed"<<std::endl;
                return false;
            }
            return true;
        }

        /// @brief             根据相机序列号作为选择相机的唯一标识
        /// @param pDeviceInfo 检测到的设备
        /// @param sn          指定相机序列号
        /// @param cameraIndex 相机序列号所对应的在线设备数组伪指针
        /// @return 
        bool ChoiceCamrea(MV_CC_DEVICE_INFO** pDeviceInfo, unsigned char* sn, size_t& cameraIndex);
        // HikCamera(/* args */);

        /// @brief                    初始化相机
        /// @param sn                 相机序列号
        /// @param autoWhiteBalance   自动白平衡
        /// @param expoosureTime      曝光时间
        /// @param gainFactor         增益
        /// @param dGammaParam        gamma               
        /// @return                   初始化是否成功，若成功返回true
        bool CameraInit( char *sn, 
                            bool autoWhiteBalance = false,
                            int expoosureTime = 2000, 
                            double gainFactor = 1.0, 
                            double dGammaParam = 1.0);
        ~HikCamera(){
            capture_stop();
            nRet = MV_CC_CloseDevice(camera_handle_);
             MV_CC_Finalize();
        }


        bool capture_stop();


        bool capture_start(double exposure_us_,double gain, double gamma);


        /// @brief 
        /// @param flip 是否进行垂直翻转。
        /// @param mirror 是否进行水平镜像。
        /// @return 
        cv::Mat getFrame(bool flip = false, bool mirror = false) ;
    };
    
    
} // namespace sdk
