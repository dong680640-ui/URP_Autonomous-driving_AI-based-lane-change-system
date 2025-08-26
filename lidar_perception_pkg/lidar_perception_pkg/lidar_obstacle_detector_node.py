import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from .lib import lidar_perception_func_lib as LPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
FRONT_SUB_TOPIC_NAME = 'front_lidar_processed'  # 구독할 토픽 이름
REAR_SUB_TOPIC_NAME = 'rear_lidar_processed'

# Publish할 토픽 이름
FRONT_PUB_TOPIC_NAME = 'front_lidar_obstacle_info'  # 물체 감지 여부를 퍼블리시할 토픽 이름
FRONT_LEFT_PUB_TOPIC_NAME = 'front_left_lidar_obstacle_info'  
FRONT_RIGHT_PUB_TOPIC_NAME = 'front_right_lidar_obstacle_info'
REAR_LEFT_PUB_TOPIC_NAME = 'rear_left_lidar_obstacle_info'  # 물체 감지 여부를 퍼블리시할 토픽 이름
REAR_RIGHT_PUB_TOPIC_NAME = 'rear_right_lidar_obstacle_info'
#----------------------------------------------


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector_node')

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 전방 라이다 구독자 및 퍼블리셔
        self.front_subscriber = self.create_subscription(
            LaserScan,
            FRONT_SUB_TOPIC_NAME,
            self.front_lidar_callback,
            self.qos_profile
        )
        self.front_publisher = self.create_publisher(
            Bool,
            FRONT_PUB_TOPIC_NAME,
            self.qos_profile
        )

        self.front_left_publisher = self.create_publisher(
            Bool,
            FRONT_LEFT_PUB_TOPIC_NAME,
            self.qos_profile
        )

        self.front_right_publisher = self.create_publisher(
            Bool,
            FRONT_RIGHT_PUB_TOPIC_NAME,
            self.qos_profile
        )

        # 후방 라이다 구독자 및 퍼블리셔
        self.rear_subscriber = self.create_subscription(
            LaserScan,
            REAR_SUB_TOPIC_NAME,
            self.rear_lidar_callback,
            self.qos_profile
        )
        self.rear_left_publisher = self.create_publisher(
            Bool,
            REAR_LEFT_PUB_TOPIC_NAME,
            self.qos_profile
        )
        self.rear_right_publisher = self.create_publisher(
            Bool,
            REAR_RIGHT_PUB_TOPIC_NAME,
            self.qos_profile
        )

        # 전방 및 후방 장애물 감지 체크
        self.front_detection_checker = LPFL.StabilityDetector(consec_count=5) # 연속적으로 몇 번 감지 여부를 확인할지 설정
        self.front_left_detection_checker = LPFL.StabilityDetector(consec_count=5)
        self.front_right_detection_checker = LPFL.StabilityDetector(consec_count=5) 
        #self.rear_detection_checker = LPFL.StabilityDetector(consec_count=5)
        self.rear_left_detection_checker = LPFL.StabilityDetector(consec_count=5)
        self.rear_right_detection_checker = LPFL.StabilityDetector(consec_count=5)


    def front_lidar_callback(self, msg):
        self.get_logger().debug('Received front lidar data')
        detection_result = self.process_lidar_data(msg, 'front', None)
        front_left_detection_result = self.process_lidar_data(msg, 'front', 'left')
        front_right_detection_result = self.process_lidar_data(msg, 'front', 'right')

        detection_msg = Bool()
        front_left_detection_msg = Bool()
        front_right_detection_msg = Bool()

        detection_msg.data = detection_result       
        front_left_detection_msg.data = front_left_detection_result
        front_right_detection_msg.data = front_right_detection_result

        self.front_publisher.publish(detection_msg)
        self.front_left_publisher.publish(front_left_detection_msg)
        self.front_right_publisher.publish(front_right_detection_msg)

        self.get_logger().info(f'Front Lidar Obstacle detected : {detection_result}')
        self.get_logger().info(f'Front Left Lidar Obstacle detected : {front_left_detection_result}')
        self.get_logger().info(f'Front Right Lidar Obstacle detected : {front_right_detection_result}')

    def rear_lidar_callback(self, msg):
        self.get_logger().debug('Received rear lidar data')
        left_detection_result = self.process_lidar_data(msg, 'rear', 'left')
        right_detection_result = self.process_lidar_data(msg, 'rear', 'right')

        left_detection_msg = Bool()
        right_detection_msg = Bool()
        left_detection_msg.data = left_detection_result
        right_detection_msg.data = right_detection_result
        self.rear_left_publisher.publish(left_detection_msg)
        self.rear_right_publisher.publish(right_detection_msg)

        self.get_logger().info(f'Rear Left Lidar Obstacle detected : {left_detection_result}')
        self.get_logger().info(f'Rear Right Lidar Obstacle detected : {right_detection_result}')


    def process_lidar_data(self, msg, lidar_position, direction):
         
        """
        라이다 데이터를 처리하여 장애물 감지 여부를 반환합니다.
        :param msg: LaserScan 메시지
        :param lidar_position: 'front' 또는 'rear'
        :return: Boolean (True if obstacle detected, else False)
        """
        # 원하는 각도 및 거리 범위 설정 (필요에 따라 수정)
        if lidar_position == 'front':
            if direction == 'right':
                start_angle = 220
                end_angle = 330
                range_min = 0.15
                range_max = 1.0
                detection_checker = self.front_right_detection_checker
            elif direction == 'left':
                start_angle = 30
                end_angle = 140
                range_min = 0.15
                range_max = 1.0
                detection_checker = self.front_left_detection_checker
            else:
                start_angle = 260  # 전방 예시
                end_angle = 100
                range_min = 0.25
                range_max = 2.3
                detection_checker = self.front_detection_checker
        elif lidar_position == 'rear':
            if direction == 'left':
                start_angle = 75 # 후방 예시
                end_angle = 170
                range_min = 0.15
                range_max = 2.0
                detection_checker = self.rear_left_detection_checker
            elif direction == 'right':
                start_angle = 185  # 후방 예시
                end_angle = 285
                range_min = 0.15
                range_max = 2.0
                detection_checker = self.rear_right_detection_checker
            
        else:
            self.get_logger().error('Invalid lidar position specified for processing')
            return False

        ranges = msg.ranges

        detected = LPFL.detect_object(
            ranges=ranges,
            start_angle=start_angle,
            end_angle=end_angle,
            range_min=range_min,
            range_max=range_max
        )

        detection_result = detection_checker.check_consecutive_detections(detected)
        return detection_result

# ranges는 라이다 센서값 입력                                                        

        # 각도 범위 지정
        # 예시 1) 
        # start_angle을 355도로, end_angle을 4도로 설정하면, 
        # 355도에서 4도까지의 모든 각도(355, 356, 357, 358, 359, 0, 1, 2, 3, 4도)가 포함.
        # 
        # 예시 2)
        # start_angle을 0도로, end_angle을 30도로 설정하면, 
        # 0도에서 30도까지의 모든 각도(0, 1, 2, ..., 30도)가 포함.
        # 
        # 예시 3)
        # start_angle을 180도로, end_angle을 190도로 설정하면, 
        # 180도에서 190도까지의 모든 각도(180, 181, 182, ..., 190도)가 포함. 

        # 거리범위 지정 
        # range_min보다 크거나 같고, range_max보다 작거나 같은 거리값을 포함.

        # 각도범위 및 거리범위를 둘 다 만족하는 범위에 라이다 센서값이 존재하면 True, 아니면 False 리턴. 


def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetection()
    rclpy.spin(object_detection_node)
    object_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
