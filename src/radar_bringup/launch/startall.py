from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import ComposableNodeContainer, Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # ==========================================
    # 0. 定义外部参数开关 (支持终端动态传参)
    # ==========================================
    use_video_arg = DeclareLaunchArgument(
        'use_video',
        default_value='false',  
        description='是否使用视频进行离线测试'
    )
    use_video = LaunchConfiguration('use_video')

    # 红蓝方阵营开关
    is_blue_team_arg = DeclareLaunchArgument(
        'is_blue_team',
        default_value='true',  # 默认蓝方
        description='雷达站所属阵营:true为蓝方,false为红方'
    )
    is_blue_team = LaunchConfiguration('is_blue_team')

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
    # 3. 神经网络组件 (纯地面装甲板模式)
    # ==========================================
    detector_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::NetDetectorComponent',
        name='net_detector_component',
        parameters=[{'config_file': '/home/lzhros/Code/RadarStation/config/detector/yolo.yaml'}],
        extra_arguments=[{'use_intra_process_comms': True}]  
    )

    # ==========================================
    # 4. 单帧标定组件
    # ==========================================
    solvepnp_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::SolvePnPComponent',
        name='solvepnp_component',
        parameters=[{
            'config_path': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'keypoint_path': '/home/lzhros/Code/RadarStation/config/solver/keypoint_6.txt'
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 5. 小地图映射组件 (已加入 Open3D 场地网格路径与阵营参数)
    # ==========================================
    map_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::MapComponent',
        name='map_component',
        parameters=[{
            'camera_yaml': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'map_yaml': '/home/lzhros/Code/RadarStation/config/map/field_image.yaml',
            'map_image': '/home/lzhros/Code/RadarStation/config/map/field_image.png',
            'mesh_path': '/home/lzhros/Code/RadarStation/config/map/RMUC2025_National.PLY',
            'is_blue_team': is_blue_team  
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 6. Qt 可视化客户端 (独立进程启动)
    # ==========================================
    visualizer_standalone_node = Node(
        package='radar_visualizer',
        executable='visualizer_node',
        name='qt_visualizer',
        output='screen'
    )

    # ==========================================
    # 7. 串口通信节点 (独立进程启动，防止阻塞视觉容器)
    # ==========================================
    serial_standalone_node = Node(
        package='radar_serial',
        executable='serial_node',
        name='serial_node',
        output='screen',
        parameters=[{
            # 注意：若需进行 socat 虚拟串口测试，请将其改回 '/tmp/ttyUSB_RADAR'
            'port_name': '/tmp/ttyUSB_RADAR',
            'is_blue_team': is_blue_team  
        }]
    )

    # ==========================================
    # 8. 核心容器 (单线程零拷贝主板，绝不抢占相机底层中断)
    # ==========================================
    container = ComposableNodeContainer(
        name='radar_vision_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', 
        composable_node_descriptions=[
            video_node,      
            camera_node,   
            detector_node,
            solvepnp_node,   
            map_node,       
        ],
        output='screen',
    )

    return LaunchDescription([
        use_video_arg, 
        is_blue_team_arg, 
        container, 
        visualizer_standalone_node,
        serial_standalone_node  
    ])