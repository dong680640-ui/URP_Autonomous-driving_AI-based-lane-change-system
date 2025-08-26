import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import CarInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_car_info"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class CarDetector(Node):
    def __init__(self):
        super().__init__('car_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(CarInfo, self.pub_topic, self.qos_profile)
    

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        
        if len(detection_msg.detections) == 0:
            return

        # 'car' 클래스의 객체만 필터링
        car_detections = [
            detection for detection in detection_msg.detections 
            if detection.class_name == 'car'
        ]

        car_info = CarInfo()
        for car in car_detections:
            center = car.bbox.center.position  # Pose2D의 position 필드 사용
            car_center_x = center.x  # 중심점의 x 좌표
            car_center_y = center.y  # 중심점의 y 좌표

            car_info.x.append(center.x)
            car_info.y.append(center.y)

        # 결과를 CarInfo 메시지로 생성 및 발행
        car_info.num_cars = len(car_info.x)
        self.publisher.publish(car_info)

def main(args=None):
    rclpy.init(args=args)
    node = CarDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()