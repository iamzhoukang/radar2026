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

    is_blue_team_arg = DeclareLaunchArgument(
        'is_blue_team',
        default_value='true',  # 默认蓝方
        description='雷达站所属阵营:true为蓝方,false为红方'
    )
    is_blue_team = LaunchConfiguration('is_blue_team')

    # ==========================================
    # 1. 视觉感知链路组件 (原有)
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

    camera_node = ComposableNode(
        condition=UnlessCondition(use_video),
        package='radar_core',
        plugin='radar_core::CameraOneComponent',
        name='camera_one',
        extra_arguments=[{'use_intra_process_comms': True}],
        remappings=[('camera_original_topic_name', 'cs200_topic')] 
    )

    detector_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::NetDetectorComponent',
        name='net_detector_component',
        parameters=[{'config_file': '/home/lzhros/Code/RadarStation/config/detector/yolo.yaml'}],
        extra_arguments=[{'use_intra_process_comms': True}]  
    )

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

    map_node = ComposableNode(
        package='radar_core',
        plugin='radar_core::MapComponent',
        name='map_component',
        parameters=[{
            'camera_yaml': '/home/lzhros/Code/RadarStation/config/solver/cs200_calibration.yaml',
            'map_yaml': '/home/lzhros/Code/RadarStation/config/map/field_image.yaml',
            'map_image': '/home/lzhros/Code/RadarStation/config/map/field_image_2026.png',
            'mesh_path': '/home/lzhros/Code/RadarStation/config/map/RB2026_rmuc.ply',
            'is_blue_team': is_blue_team  
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 2. 激光雷达感知链路组件 (新增)
    # ==========================================
    localization_node = ComposableNode(
        package='radar_lidar',
        plugin='radar_lidar::Localization',
        name='localization_node',
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    dynamic_cloud_node = ComposableNode(
        package='radar_lidar',
        plugin='radar_lidar::DynamicCloud',
        name='dynamic_cloud_node',
        parameters=[{
            'map_path': '/home/lzhros/Code/RadarStation/config/lidar/RB2026_rmuc.pcd',
            'threshold': 0.2
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    cluster_node = ComposableNode(
        package='radar_lidar',
        plugin='radar_lidar::ClusterNode',
        name='cluster_node',
        parameters=[{
            'cluster_tolerance': 0.6,
            'min_cluster_size': 15,
            'max_cluster_size': 2000
        }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==========================================
    # 3. 独立进程节点 (可视化与串口)
    # ==========================================
    visualizer_standalone_node = Node(
        package='radar_visualizer',
        executable='visualizer_node',
        name='qt_visualizer',
        output='screen'
    )

    serial_standalone_node = Node(
        package='radar_serial',
        executable='serial_node',
        name='serial_node',
        output='screen',
        parameters=[{
            'port_name': '/tmp/ttyUSB_RADAR',
            'is_blue_team': is_blue_team  
        }]
    )

    # ==========================================
    # 4. 视觉核心容器 (单线程，保障帧率极速流水线)
    # ==========================================
    vision_container = ComposableNodeContainer(
        name='radar_vision_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', # 单线程执行器
        composable_node_descriptions=[
            video_node,      
            camera_node,   
            detector_node,
            solvepnp_node,   
            map_node,       
        ],
        output='screen',
    )

    # ==========================================
    # 5. 雷达核心容器 (多线程，保障点云并发不阻塞)
    # ==========================================
    lidar_container = ComposableNodeContainer(
        name='radar_lidar_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt', # MT = Multi-Threaded 多线程执行器
        composable_node_descriptions=[
            localization_node,
            dynamic_cloud_node,
            cluster_node
        ],
        output='screen',
    )

    return LaunchDescription([
        use_video_arg, 
        is_blue_team_arg, 
        vision_container, 
        lidar_container,
        visualizer_standalone_node,
        serial_standalone_node  
    ])