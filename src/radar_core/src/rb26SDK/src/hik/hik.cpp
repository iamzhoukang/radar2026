#include "../../include/hik/hik.hpp"

namespace sdk{
    bool HikCamera::ChoiceCamrea(MV_CC_DEVICE_INFO** pDeviceInfo, unsigned char* sn, size_t& cameraIndex){
        for(size_t i = 0; i < nDeviceNum; i++){
            std::cout<<"pDeviceInfo "<<i<<": "<<pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber<<std::endl;
            // if(*pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber == *sn) {
            //     nDeviceNum = i;
            //     return true;
            // }
            bool wl = true;
            for(int j  = 0; sn[j]!='\0';j++){
                if(sn[j] != pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber[j]) {
                    wl = false;
                    break;
                }
            }
            if(wl) {
                cameraIndex = i;
                return true;
            }
        }
        return false;

    }

    bool HikCamera::CameraInit( char *sn, 
                            bool autoWhiteBalance ,
                            int expoosureTime , 
                            double gainFactor , 
                            double dGammaParam 
                            ){

        cap_sn = sn;
        camera_breand =  Hik;
        // if(!CameraSDKInit()) return false;
        nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);//枚举设备数量
        if(nRet != MV_OK) {
            std::cout<<"hik EnumDevices failed"<<std::endl;
            return false;
        }
        if(device_list.nDeviceNum == 0) {
        std::cout<<"设备数量为0" <<std::endl;
            return false;
        }
        this->nDeviceNum = device_list.nDeviceNum;
        std::cout<<"device_list.nDeviceNum "<<nDeviceNum <<std::endl;
        size_t cameraIndex = 0;
        bool exist = ChoiceCamrea(device_list.pDeviceInfo, (unsigned char*)sn, cameraIndex);
        std::cout<<"camrea exist "<<exist<<std::endl;
        if(!exist){
            std::cout<<"不存在hik相机"<<sn<<std::endl;
                return false;
        }

        MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[cameraIndex]);
        // memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        // nRet = MV_CC_EnumDevicesEx2(MV_GIGE_DEVICE|MV_USB_DEVICE, &device_list, sn, SortMethod_SerialNumber);
        
        if(nRet != MV_OK) {
            std::cout<<"hik EnumDevices USB failed"<<std::endl;
            return false;
        }

        // nRet = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[0]);
        if(nRet != MV_OK) {
            std::cout<<"Create handle failed"<<std::endl;
            return false;
        }


        //打开相机
        nRet = MV_CC_OpenDevice(camera_handle_);
        if(nRet != MV_OK) {
            printf("Open Device fail! nRet [0x%x]\n", nRet);
            return false;
        }
        update_timer();

        if(!capture_start(expoosureTime, gainFactor, dGammaParam) ){
            std::cout<<"capture_start error"<<std::endl;
                return false;
        }
            
        
        // if(!capture_stop()){
        //     std::cout<<"capture_stop"<<std::endl;
        //     return false;
        // }
        

        MVCC_INTVALUE_EX width_max, height_max;
        MV_CC_GetIntValueEx(camera_handle_, "WidthMax", &width_max);
        MV_CC_GetIntValueEx(camera_handle_, "Height", &height_max);
        this->sensorWidth = width_max.nCurValue;
        this->sensorHeight = height_max.nCurValue;
        // cap_init = true;
        return true;
    }

    bool HikCamera::capture_stop()
    {
        // capture_quit_ = true;
        // if (capture_thread_.joinable()) capture_thread_.join();

        // unsigned int ret;

        nRet = MV_CC_StopGrabbing(camera_handle_);
        if (nRet != MV_OK) {
            printf("MV_CC_StopGrabbing fail! nRet = [0x%x]\n", nRet);
            return false;
        }

        nRet = MV_CC_CloseDevice(camera_handle_);
        if (nRet != MV_OK) {
            printf("MV_CC_CloseDevice fail! nRet = [0x%x]\n", nRet);
            return false;
        }

        nRet = MV_CC_DestroyHandle(camera_handle_);
        if (nRet != MV_OK) 
        {
            printf("MV_CC_DestroyHandle fail! nRet = [0x%x]\n", nRet);
            return false;
        }
        return true;
            
    
    }

    bool HikCamera::capture_start(double exposure_us_,double gain, double gamma)
    {

        // unsigned int ret;

        // MV_CC_DEVICE_INFO_LIST device_list;

        // ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[0]);//创建设备句柄
        // if (ret != MV_OK) return false;

        // ret = MV_CC_OpenDevice(camera_handle_);
        // if (ret != MV_OK) return false;

        // set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);

        /**
         * 自动白平衡
            分为“关闭”、“一次”和“连续”三种模式。
            关闭
            选择“关闭”时，可通过白平衡分量选项和白平衡分量设置红、黄、蓝各分量数值。
            一次
            选择“一次”时，相机根据当前场景运行一段时间后停止自动白平衡。
            连续
            选择“连续”时，相机根据当前场景，自动进行白平衡调整。可通过白平衡分量选项和
            白平衡分量查看红、黄、蓝各分量数值。
        */
        nRet = MV_CC_SetEnumValue(camera_handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);//自动白平衡
        if (nRet != MV_OK) {
            printf("BalanceWhiteAuto fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        // set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        nRet = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);//关闭自动曝光
        if (nRet != MV_OK) {
            printf("ExposureAuto fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        // set_enum_value("GainAuto", MV_GAIN_MODE_OFF);

        // MVCC_ENUMENTRY gammaEnableSupport;
        // nRet = MV_CC_IsFeatureSupported(camera_handle_, "GammaEnable", &gammaEnableSupport);
        nRet = MV_CC_SetEnumValue(camera_handle_, "GainAuto", MV_GAIN_MODE_OFF);//关闭自动增益
        if (nRet != MV_OK) {
            printf("GainAuto fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        // set_float_value("ExposureTime", exposure_us_);
        nRet = MV_CC_SetBoolValue(camera_handle_, "GammaEnable", true);//gamma使能
        if (nRet != MV_OK) {
            printf("GammaEnable fail! nRet = [0x%x]\n", nRet);
            // return false;
        }  
        nRet = MV_CC_SetEnumValue(camera_handle_, "GammaSelector", MV_GAMMA_SELECTOR_USER);//gamma选择器 用户模式
        if (nRet != MV_OK) {
            printf("GammaSelector fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        
        nRet = MV_CC_SetFloatValue(camera_handle_, "Gamma", gamma);//gamma调节
        if (nRet != MV_OK) {
            printf("Gamma fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        nRet = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_us_);//设置曝光
        if (nRet != MV_OK) {
            printf("ExposureTime fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        // set_float_value("Gain", gain_);
        MVCC_FLOATVALUE gainRange;
        nRet = MV_CC_GetFloatValue(camera_handle_,"AutoGainUpperLimit", &gainRange);//获取增益值范围
        if (nRet != MV_OK) {
            printf("AutoGainUpperLimit fail! nRet = [0x%x]\n", nRet);
            // return false;
        }
        nRet = MV_CC_SetFloatValue(camera_handle_, "Gain", gainRange.fMax*gain);//设置增益，取值限定为0～1
        if (nRet != MV_OK) {
            printf("Gain fail! nRet = [0x%x]\n", nRet);
            return false;
        }
        // MV_CC_SetFrameRate(camera_handle_, 150);

        
        nRet = MV_CC_StartGrabbing(camera_handle_);
        if (nRet != MV_OK) {
            printf("MV_CC_StartGrabbing fail! nRet = [0x%x]\n", nRet);
            return false;
        }
        return true;
    }

    cv::Mat HikCamera::getFrame(bool flip , bool mirror ) {
        unsigned int nMsec = 100;

        nRet = MV_CC_GetImageBuffer(camera_handle_, &raw, nMsec);
        if (nRet != MV_OK) {
            std::cout << "MV_CC_GetImageBuffer failed: " << nRet << std::endl;
            error_num++;
            return {};
        }
        // if()

        cv::Mat img;
        if (raw.stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8 ||
            raw.stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8 ||
            raw.stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8 ||
            raw.stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8) {
            // 使用OpenCV进行Bayer转换
            cv::Mat raw_img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8UC1, raw.pBufAddr);

            //基于哈希表的查找、插入、删除操作的平均时间复杂度为 O(1)
            const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
                {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
                {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
                {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
                {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
                // std::cout<<"if "<<std::endl;
            cv::cvtColor(raw_img, img, type_map.at(raw.stFrameInfo.enPixelType));
            // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        } else {
            // std::cout<<"else "<<std::endl;
            img = cv::Mat(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8UC3, raw.pBufAddr);
        }

        // 翻转和镜像
        if (flip) {
            cv::flip(img, img, 0); // 垂直翻转
        }
        if (mirror) {
            cv::flip(img, img, 1); // 水平镜像
        }

        nRet = MV_CC_FreeImageBuffer(camera_handle_, &raw);
        if (nRet != MV_OK) {
            std::cout << "MV_CC_FreeImageBuffer failed: " << nRet << std::endl;
        }

        update_timer();
        cap_init = true;
        return img;
        }
}