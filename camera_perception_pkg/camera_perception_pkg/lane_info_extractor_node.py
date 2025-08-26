import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

import numpy as np

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

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
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
 
    def get_lane_center(self,cv_image: np.array, detection_height: int, detection_thickness: int, road_gradient: float, lane_width: int) -> int:
        detection_area_upper_bound = detection_height - int(detection_thickness/2)
        detection_area_lower_bound = detection_height + int(detection_thickness/2)

        detected_x_coords = np.sort(np.where(cv_image[detection_area_upper_bound:detection_area_lower_bound,:]!=0)[1])

        if (detected_x_coords.shape[0] < 5):
            line_x_axis_pixel = None
            center_pixel = None
            return None

        cut_outliers_array = detected_x_coords[1:-1]
        difference_array = cut_outliers_array[1:] - cut_outliers_array[:-1]

        max_diff_idx_left = np.argmax(difference_array)
        max_diff_idx_right = np.argmax(difference_array)+1
        left_val = cut_outliers_array[max_diff_idx_left]
        right_val = cut_outliers_array[max_diff_idx_right]

        if abs(left_val - right_val) < (lane_width/3):
            line_x_axis_pixel = cut_outliers_array[round((cut_outliers_array.shape[0])/2)]
            center_pixel = None
        else:
            line_x_axis_pixel = None
            center_pixel = (left_val + right_val)/2

        if center_pixel == None and line_x_axis_pixel == None:
            road_target_point_x = None
        else:
            road_target_point_x = center_pixel
            if road_target_point_x == None and line_x_axis_pixel != None:
                if cut_outliers_array[-1] > 540:
                    road_target_point_x = line_x_axis_pixel + (lane_width/2)
                    if road_target_point_x < (639-lane_width):
                        road_target_point_x = (639-lane_width)
                    elif road_target_point_x > 639:
                        road_target_point_x = 639
                elif cut_outliers_array[0] < 100:
                    road_target_point_x = line_x_axis_pixel - (lane_width/2)
                    if road_target_point_x > (lane_width-1):
                        road_target_point_x = (lane_width-1)
                    elif road_target_point_x < 0:
                        road_target_point_x = 0
                else:
                    road_target_point_x = left_val + int((left_val - right_val) / 2)
                    
        return road_target_point_x

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        
        if len(detection_msg.detections) == 0:
            return

        lane1_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane1', color=255)
        lane2_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane2', color=255)

        (h1, w1) = (lane1_edge_image.shape[0], lane1_edge_image.shape[1]) #(480, 640)
        (h2, w2) = (lane2_edge_image.shape[0], lane2_edge_image.shape[1]) #(480, 640)

        dst_mat = [[round(w1 * 0.3), round(h1 * 0.0)], [round(w1 * 0.7), round(h1 * 0.0)], [round(w1 * 0.7), h1], [round(w1 * 0.3), h1]]
        src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]

        lane1_bird_image = CPFL.bird_convert(lane1_edge_image, srcmat=src_mat, dstmat=dst_mat)
        lane2_bird_image = CPFL.bird_convert(lane2_edge_image, srcmat=src_mat, dstmat=dst_mat)

        roi_image1 = CPFL.roi_rectangle_below(lane1_bird_image, cutting_idx=300)
        roi_image2 = CPFL.roi_rectangle_below(lane2_bird_image, cutting_idx=300)

        if self.show_image:
            cv2.imshow('lane1_edge_image', lane1_edge_image)
            cv2.imshow('lane1_bird_img', lane1_bird_image)
            cv2.imshow('roi_img1', roi_image1)
            
            cv2.imshow('lane2_edge_image', lane2_edge_image)
            cv2.imshow('lane2_bird_img', lane2_bird_image)
            cv2.imshow('roi_img2', roi_image2)

            cv2.waitKey(1)

        grad1 = CPFL.dominant_gradient(roi_image1, theta_limit=70)
        grad2 = CPFL.dominant_gradient(roi_image2, theta_limit=70)

        lane1_point_y = 90
        lane1_point_x = CPFL.get_lane_center(roi_image1, detection_height=lane1_point_y, 
                              detection_thickness=10, road_gradient=grad1, lane_width=300)
        lane2_point_y = 90
        lane2_point_x = self.get_lane_center(roi_image2, detection_height=lane2_point_y, 
                              detection_thickness=10, road_gradient=grad2, lane_width=300)

        lane = LaneInfo()

        lane.slope1 = grad1
        lane.slope2 = grad2
        if lane1_point_x is not None:
            lane.lane1_x = round(lane1_point_x)
        if lane1_point_y is not None:
            lane.lane1_y = round(lane1_point_y)
        if lane2_point_x is not None:
            lane.lane2_x = round(lane2_point_x)
        if lane2_point_y is not None:
            lane.lane2_y = round(lane2_point_y)

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
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