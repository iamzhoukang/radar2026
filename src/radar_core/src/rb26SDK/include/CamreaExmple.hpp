#pragma once

#include "./daheng/daheng.hpp"
#include "./hik/hik.hpp"

#include <csignal>

namespace sdk{


    template<class CamreaType>
    class CameraExmple : public CamreaType{
    public:
        static_assert(
            std::is_same<CamreaType, HikCamera>::value || 
            std::is_same<CamreaType, DahengCamera>::value || 
            "Template parameter CamreaType must be HikCamera, or DahengCamera"
        );
        cv::Mat frame;
        
        ~CameraExmple(){
            CamreaType::~CamreaType();
        }


        void putResolution(){
            std::cout<<"分辨率为： width "<<this->sensorWidth
                    <<"   hight "<<this->sensorHeight<<std::endl;
           
        }

        /// @brief 输出帧率/延迟，其值取决于getFrame调用频率
        void putFps(){ 
            std::cout<<"fps: "<<this->fps<<std::endl;
            // std::cout<<"delay: "<<1./this->fps<<std::endl;
        }

        /// @brief 
        /// @param flip 是否进行垂直翻转。
        /// @param mirror 是否进行水平镜像。
        /// @return 
        cv::Mat getFrame(bool flip = false, bool mirror = false){
            frame = CamreaType::getFrame(flip, mirror);
            return frame;
        }
        
        // CameraExmple(){
        //      signal(SIGINT, this->~T());
        // }
        

    };  


}

