#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using  ClockPoint =  std::chrono::time_point<std::chrono::high_resolution_clock>;
#define    NowTime     std::chrono::high_resolution_clock::now()

namespace sdk
{
    class Camera{
    public:
        typedef enum{
            NullClass,
            Daheng,
            Hik,
        }CameraBrand;
        CameraBrand camera_breand =  NullClass;//相机品牌

        char* cap_sn;//相机序列号
        int64_t sensorWidth = -1, sensorHeight = -1; //相机分辨率
        ClockPoint time_point = NowTime;
        double fps = 0;
        bool cap_init = false;

        // virtual bool CameraSDKInit() = 0;

        virtual bool CameraInit( char *sn, 
                            bool autoWhiteBalance , 
                            int expoosureTime , 
                            double gainFactor , 
                            double dGammaParam 
                            ) = 0 ;

        double update_timer(){
            fps = 1./((double) std::chrono::duration_cast< std::chrono::microseconds> ( NowTime - time_point ).count() / 1000000);
            time_point = NowTime;
            return fps;
        }


    };
} // namespace sdk
