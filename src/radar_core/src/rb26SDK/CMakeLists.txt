cmake_minimum_required(VERSION 3.16)
project(rb26SDK)

include_directories(include)
include_directories(include/daheng)
include_directories(include/hik)


add_library(rb26SDK #./rb26SDKlib.cpp
                    ./src/daheng/daheng.cpp
                    ./src/hik/hik.cpp
                    )

target_include_directories(rb26SDK PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/daheng
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/hik
                                        #    ${CMAKE_CURRENT_SOURCE_DIR}/include/number_classifier
                                           )

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(rb26SDK PUBLIC /opt/MVS/lib/64)#这里要改成实际的安装位置，不过一般不用
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(rb26SDK PUBLIC src/hik/lib/arm64)
else() 
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}

        )



target_link_libraries( rb26SDK     PUBLIC 
                                MvCameraControl  #usb-1.0
                                gxiapi
                                pthread #（可选多线程启动）
                                ${OpenCV_LIBS}
                                )