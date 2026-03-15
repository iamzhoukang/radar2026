#include "../../include/daheng/daheng.hpp"

namespace sdk{
    DahengCamera::~DahengCamera() {
        status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_STOP);

        free(frameData.pImgBuf);
        frameData.pImgBuf = nullptr;
        free(pRaw8Buffer);
        pRaw8Buffer = nullptr;
        free(pRGBframeData);
        pRGBframeData = nullptr;
        if (pGammaLut!= NULL)//新加的
        {
            free(pGammaLut);
            pGammaLut= NULL;
        }
        GXCloseDevice(hDevice);
        GXCloseLib();
    }

    bool DahengCamera::CameraInit( char *sn, 
                            bool autoWhiteBalance , 
                            int expoosureTime , 
                            double gainFactor , 
                            double dGammaParam 
                            ) {
        camera_breand =  Daheng;
        cap_sn = sn;
        // if(!CameraSDKInit()) return false;
        //给相机设备接口结构体创建动态内存
        auto *openParam = new GX_OPEN_PARAM;
        openParam->openMode = GX_OPEN_SN;//通过序列号打开设备
        openParam->accessMode = GX_ACCESS_EXCLUSIVE;//以独占方式打开设备 
        openParam->pszContent = sn;//标准C字符串，由openMode决定，可能是一个IP地址或者是相机序列号等等
        status = GXOpenDevice(openParam, &hDevice);//通过指定唯一标识打开设备，例如指定SN、IP、MAC、Index等,hDevice是接口返回的设备句柄 
        if (status != GX_STATUS_SUCCESS) {//操作成功，没有发生错误 
            std::cout<<"不存在daheng相机"<<sn<<std::endl;
            return false;
        }

        int64_t nPayLoadSize = 0;
        update_timer();

        //调整ROI长度和宽度,不过这种方法限制了分辨率，相当于只是用ROI区域作为相机采集图像区域
        // if(aim){
        //     GXSetInt(hDevice, GX_INT_WIDTH, 1440);
        //     GXSetInt(hDevice, GX_INT_HEIGHT, 1080);
        // }else{
        //     GXSetInt(hDevice, GX_INT_WIDTH, 1088);
        //     GXSetInt(hDevice, GX_INT_HEIGHT, 720);
        // }


        //设置相机当前输出的每一帧图像数据大小，大小为GX_INT_PAYLOAD_SIZE，赋值到nPayLoadSize
        status = GXGetInt(hDevice, GX_INT_PAYLOAD_SIZE, &nPayLoadSize);
        if (status != GX_STATUS_SUCCESS) {
            return false;
        }

        //pImgBuf是图像数据指针（开启帧信息后，pImgBuf 包含图像数据和帧信息数据）
        frameData.pImgBuf = malloc((size_t) nPayLoadSize);
        // GXGetInt(hDevice, GX_INT_SENSOR_WIDTH, &sensorWidth);
        // GXGetInt(hDevice, GX_INT_SENSOR_HEIGHT, &sensorHeight);

        pRaw8Buffer = malloc(nPayLoadSize);
        pMirrorBuffer = malloc(nPayLoadSize * 3);
        pRGBframeData = malloc(nPayLoadSize * 3);

        GXGetEnum(hDevice, GX_ENUM_PIXEL_FORMAT, &PixelFormat);//获取当前的PixelFormat
        GXGetEnum(hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &ColorFilter);//获取当前的Bayer 格式
        GXSetEnum(hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);///设置采集模式为连续模式GX_ACQ_MODE_CONTINUOUS

        // FrameValidFormatChoice();

        //使能采集帧率调节模式
        // status = GXSetEnum(hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE ,
        // GX_ACQUISITION_FRAME_RATE_MODE_ON);
        // //设置采集帧率,假设设置为 10.0，用户按照实际需求设置此值
        // status = GXSetFloat(hDevice, GX_FLOAT_ACQUISITION_FRAME_RATE, 100.0);
        // status = GXSetEnum(hDevice, GX_ENUM_AA_LIGHT_ENVIRMENT, GX_AA_LIGHT_ENVIRMENT_NATURELIGHT);
        // GXSetBool(hDevice, GX_BOOL_COLOR_TRANSFORMATION_ENABLE, false); 
        status = GXSetEnum(hDevice, GX_ENUM_LIGHT_SOURCE_PRESET, GX_LIGHT_SOURCE_PRESET_OFF);//环境光源预设关闭
        // if (status != GX_STATUS_SUCCESS) {
        //     cout<<status<<endl;
        //     return false;
        // }
        GXSetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO,
                    autoWhiteBalance ? GX_BALANCE_WHITE_AUTO_CONTINUOUS : GX_BALANCE_WHITE_AUTO_OFF);//设置自动白平衡使能

        GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, expoosureTime);//设置曝光时间


        GXSetEnum(hDevice, GX_ENUM_GAIN_SELECTOR, GX_GAIN_SELECTOR_ALL);//增益通道选择为所有增益通道
        GX_FLOAT_RANGE gainRange;//GX_FLOAT_RANGE描述了浮点型值的最大值、最小值、步长、单位。
        GXGetFloatRange(hDevice, GX_FLOAT_GAIN, &gainRange);//GX_FLOAT_GAIN该值是一个浮点值，用于以特定于摄像机的单位设置选定的增益控制。
        GXSetFloat(hDevice, GX_FLOAT_GAIN, gainRange.dMax * gainFactor);//gainFactor为最大增益值小于1的分数，最大增益分之几

        status = GXSetBool(hDevice, GX_BOOL_GAMMA_ENABLE, true);//Gamma 使能
        GX_GAMMA_MODE_ENTRY nValue;
        nValue = GX_GAMMA_SELECTOR_USER;//自定义gamma值
        status = GXSetEnum(hDevice, GX_ENUM_GAMMA_MODE, nValue);

        GXGetInt(hDevice, GX_INT_SENSOR_HEIGHT, &this->sensorHeight);
        GXGetInt(hDevice, GX_INT_SENSOR_WIDTH, &this->sensorWidth);

        int nLutLength = 0;
        status = GXSetFloat(hDevice, GX_FLOAT_GAMMA, dGammaParam);//修改gamma值
        // void *pGammaLut;
        do 
        {   
            VxInt32 DXStatus= DxGetGammatLut(dGammaParam, NULL, &nLutLength);//获取gamma查找表长度nLutLength
            if (DXStatus != DX_OK)
            {
                break;
            }
            //为 Gamma 查 找 表 申 请 空 间
            pGammaLut = new int[nLutLength];
            if (pGammaLut== NULL)
            {
                DXStatus= DX_NOT_ENOUGH_SYSTEM_MEMORY;//系统内存不足
                break;
            }
            //计 算 Gamma 查 找 表
            DXStatus = DxGetGammatLut(dGammaParam, pGammaLut, &nLutLength);
            if (DXStatus != DX_OK)
            {
                if (pGammaLut!= nullptr)//新加的
                {
                    free(pGammaLut);
                    pGammaLut= nullptr;
                }
                break;
            }
        }while(0);

        // cout<<status<<dGammaParam<<endl;
        status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_START);//发送开采命令
        if (status != GX_STATUS_SUCCESS) {
            return false;
        }
        // this->cap_init = true;
        return true;
    }

    Mat DahengCamera::getFrame(bool flip , bool mirror ) {
        if (GXGetImage(hDevice, &frameData, 100) == GX_STATUS_SUCCESS) {//在开始采集之后，通过此接口可以直接获取图像，注意此接口不能与回调采集方式混用。

            if (frameData.nStatus == 0) {
                ProcessData(frameData.pImgBuf, pRaw8Buffer, pRGBframeData, frameData.nWidth, frameData.nHeight,
                            (int) PixelFormat, mirror ? 2 : 4, flip, mirror);
                Mat src(Size(frameData.nWidth, frameData.nHeight), CV_8UC3, pRGBframeData);
                update_timer();
                cap_init = true;
                return src.clone();//这里不做深拷贝的话多相机imshow画面会出现横杠
            }
        }

        return {};
    }
    

    void DahengCamera::ProcessData(void *pImageBuf, void *pImageRaw8Buf, void *pImageRGBBuf, int nImageWidth, int nImageHeight,
                         int nPixelFormat, int nPixelColorFilter, bool flip , bool mirror ) {
            
            
        
        switch (nPixelFormat) {
            //DxImageMirror:该函数为产生一个与原图像在水平方向或者垂直方向相对称的镜像图像,输入图像为 8 位的 Raw 图像或 8 位的黑白图像。
            //DxRaw16toRaw8:该函数将 Raw16 图像（实际位数为 16 位，有效位为 10 位或者 12 位）转换成 Raw8 位图像（实际位数和有效位数都是 8 位）。
            //DxRaw8toRGB24:该函数用于将 Bayer 图像转换为 RGB 图像。bFlip 是否翻转；true，将图像进行上下翻转；FALSE，不翻转
            case GX_PIXEL_FORMAT_BAYER_GR12:
            case GX_PIXEL_FORMAT_BAYER_RG12:
            case GX_PIXEL_FORMAT_BAYER_GB12:
            case GX_PIXEL_FORMAT_BAYER_BG12:
                if (mirror) {
                    DxImageMirror(pImageBuf, pMirrorBuffer, nImageWidth, nImageHeight, HORIZONTAL_MIRROR);
                    DxRaw16toRaw8(pMirrorBuffer, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_4_11);
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip);
                } else {
                    DxRaw16toRaw8(pImageBuf, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_4_11);
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip);
                }
                break;

            case GX_PIXEL_FORMAT_BAYER_GR10:
            case GX_PIXEL_FORMAT_BAYER_RG10:
            case GX_PIXEL_FORMAT_BAYER_GB10:
            case GX_PIXEL_FORMAT_BAYER_BG10:
                if (mirror) {
                    DxImageMirror(pImageBuf, pMirrorBuffer, nImageWidth, nImageHeight, HORIZONTAL_MIRROR);
                    DxRaw16toRaw8(pMirrorBuffer, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_2_9);
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip);
                } else {
                    DxRaw16toRaw8(pImageBuf, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_2_9);
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip);
                }
                break;

            case GX_PIXEL_FORMAT_BAYER_GR8:
            case GX_PIXEL_FORMAT_BAYER_RG8:
            case GX_PIXEL_FORMAT_BAYER_GB8:
            case GX_PIXEL_FORMAT_BAYER_BG8:
                if (mirror) {
                    DxImageMirror(pImageBuf, pMirrorBuffer, nImageWidth, nImageHeight, HORIZONTAL_MIRROR);
                    DxRaw8toRGB24(pMirrorBuffer, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip); //RAW2RGB_ADAPTIVE
                } else {
                    DxRaw8toRGB24(pImageBuf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(nPixelColorFilter), flip); //RAW2RGB_ADAPTIVE
                }
                break;

            case GX_PIXEL_FORMAT_MONO12:
            case GX_PIXEL_FORMAT_MONO10:
                if (mirror) {
                    DxRaw16toRaw8(pMirrorBuffer, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_4_11);//DxIma16toRaw8
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(NONE), flip);
                } else {
                    DxRaw16toRaw8(pImageBuf, pImageRaw8Buf, nImageWidth, nImageHeight, DX_BIT_4_11);
                    DxRaw8toRGB24(pImageRaw8Buf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(NONE), flip);
                }
                break;

            case GX_PIXEL_FORMAT_MONO8:
                if (mirror) {
                    DxImageMirror(pImageBuf, pMirrorBuffer, nImageWidth, nImageHeight, HORIZONTAL_MIRROR);
                    DxRaw8toRGB24(pMirrorBuffer, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(NONE), flip);
                } else {
                    DxRaw8toRGB24(pImageBuf, pImageRGBBuf, nImageWidth, nImageHeight, RAW2RGB_NEIGHBOUR,
                                    DX_PIXEL_COLOR_FILTER(NONE), flip);
                }
                break;

            default:
                break;
        }
    }
}