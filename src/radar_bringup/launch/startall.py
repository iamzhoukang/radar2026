from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import ComposableNodeContainer, Node # 导入 Node 用于启动独立进程
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # ==========================================
    # 0. 定义外部参数开关
    # ==========================================
    use_video_arg = DeclareLaunchArgument(
        'use_video',
        default_value='false',  #true表示使用视频进行离线测试，false表示使用相机进行实时测试
        description='是否使用视频进行离线测试'
    )
    use_video = LaunchConfiguration('use_video')

    # ==========================================
    # 1. 视频组件 (零拷贝插件)
    # ==========================================
    video_node = ComposableNode(
        condition=IfCondition(use_video),
        package='radar_core',
        plugin='radar_core::VideoComponent',
        name='video_test',
        parameters=[{'video_path': '/home/lzhros/Code/RadarStation/video/output.mp4'}],
        extra_arguments=[{'use_intra_process_comms': True}],
        remappings=[('video_topic', 'cs200_topic')] 
    )

    # ==========================================
    # 2. 相机组件 (零拷贝插件)
    # ==========================================
    camera_node = ComposableNode(
        condition=UnlessCondition(use_video),
        package='radar_core',
        plugin='radar_core::CameraOneComponent',
        name='camera_one',
        extra_arguments=[{'use_intra_process_comms': True}],
        remappings=[('camera_original_topic_name', 'cs200_topic')] 
    )

    # ==========================================
    # 3. 神经网络组件 (零拷贝插件 - 车辆装甲板)
    # ==========================================
    detector_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::NetDetectorComponent',
        name='net_detector_component',
        parameters=[{'config_file': '/home/lzhros/Code/RadarStation/config/detector/yolo.yaml'}],
        extra_arguments=[{'use_intra_process_comms': True}]  
    )

    #3.5. 防空神经网络组件
    air_detector_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::AirDetectorComponent',
        name='air_detector_component',
        parameters=[{
            'camera_config_path': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'model_config_path': '/home/lzhros/Code/RadarStation/config/detector/yolo.yaml',
            'enable_debug_stream': True  # <--- 在这里控制：False为极致性能模式，True为开启调试画面
        }],
        extra_arguments=[{'use_intra_process_comms': True}]  
    )

    # ==========================================
    # 4. 单帧标定组件 (零拷贝插件)
    # ==========================================
    solvepnp_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::SolvePnPComponent',
        name='solvepnp_component',
        parameters=[{
            'config_path': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'keypoint_path': '/home/lzhros/Code/RadarStation/config/solver/keypoint_6.txt'              #调试需修改
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 5. 小地图映射组件 (零拷贝插件)
    # ==========================================
    map_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::MapComponent',
        name='map_component',
        parameters=[{
            'camera_yaml': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'map_yaml': '/home/lzhros/Code/RadarStation/config/map/field_image.yaml',                   #调试需修改
            'map_image': '/home/lzhros/Code/RadarStation/config/map/field_image.png'                    #调试需修改                    
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 6. Qt 可视化客户端 (独立进程启动)
    # ==========================================
    visualizer_standalone_node = Node(
        package='radar_visualizer',
        executable='visualizer_node', # 必须对应 CMakeLists.txt 中的 add_executable 名称
        name='qt_visualizer',
        output='screen'
    )

    # ==========================================
    # 7. 核心容器 (零拷贝主板)
    # ==========================================
    container = ComposableNodeContainer(
        name='radar_vision_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt', # 多线程容器，适合包含计算密集型组件
        composable_node_descriptions=[
            video_node,      
            camera_node,   
            detector_node,
            air_detector_node, 
            solvepnp_node,   
            map_node,       
        ],
        output='screen',
    )

    # 同时返回容器（主逻辑）和独立节点（可视化）
    return LaunchDescription([
        use_video_arg, 
        container, 
        visualizer_standalone_node
    ])