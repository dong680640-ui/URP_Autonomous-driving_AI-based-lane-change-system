import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import CarInfo, LaneInfo, MotionCommand
from .lib import decision_making_func_lib as DMFL

from enum import Enum
import numpy as np


# 변수 설정
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"
SUB_CAR_TOPIC_NAME = "yolov8_car_info"
SUB_FRONT_LIDAR_OBSTACLE_TOPIC_NAME = "front_lidar_obstacle_info"
SUB_FRONT_LEFT_LIDAR_OBSTACLE_TOPIC_NAME = 'front_left_lidar_obstacle_info'
SUB_FRONT_RIGHT_LIDAR_OBSTACLE_TOPIC_NAME = 'front_right_lidar_obstacle_info'
SUB_REAR_LEFT_LIDAR_OBSTACLE_TOPIC_NAME = "rear_left_lidar_obstacle_info"
SUB_REAR_RIGHT_LIDAR_OBSTACLE_TOPIC_NAME = "rear_right_lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

# 모션 플랜 발행 주기 (초) - 소수점 필요 (int형은 반영되지 않음)
TIMER = 0.1

# FSM STATE
class State(Enum):
    NORMAL_DRIVING = 0 # 일반 주행
    PREPARING_TO_CHANGE_LANE = 1 # 차선 변경 준비
    CHANGING_LANE = 2 # 차선 변경
    OVERTAKING = 3 # 추월
    RETURNING_LANE = 4 # 차선 복귀


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')
        ## 토픽 이름 설정
        # 카메라 인식
        self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.sub_car_topic = self.declare_parameter('sub_car_topic', SUB_CAR_TOPIC_NAME).value
        # Lidar 인식
        self.sub_front_lidar_obstacle_topic = self.declare_parameter('sub_front_lidar_obstacle_topic', SUB_FRONT_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.sub_front_left_lidar_obstacle_topic = self.declare_parameter('sub_front_left_lidar_obstacle_topic', SUB_FRONT_LEFT_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.sub_front_right_lidar_obstacle_topic = self.declare_parameter('sub_front_right_lidar_obstacle_topic', SUB_FRONT_RIGHT_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.sub_rear_left_lidar_obstacle_topic = self.declare_parameter('sub_rear_left_lidar_obstacle_topic', SUB_REAR_LEFT_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.sub_rear_right_lidar_obstacle_topic = self.declare_parameter('sub_rear_right_lidar_obstacle_topic', SUB_REAR_RIGHT_LIDAR_OBSTACLE_TOPIC_NAME).value
        # 퍼블리셔
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        # 타이머
        self.timer_period = self.declare_parameter('timer', TIMER).value

        ## QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        ## 변수 초기화
        # self.detection_data = None
        self.lane_data = None
        self.car_data = None
        self.front_lidar_data = None
        self.front_left_lidar_data = None
        self.front_right_lidar_data = None
        self.rear_left_lidar_data = None
        self.rear_right_lidar_data = None
        # 차량의 속도 (-255 ~ +255), 음수면 후진, 양수면 전진
        self.speed = 200
        # 차량의 조향 및 바퀴 속도
        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        # FSM
        self.current_state = State.NORMAL_DRIVING
        self.current_lane = 1 # 0이면 1차선, 1이면 2차선 주행 중
        self.target_lane = 1 # 0이면 1차선, 1이면 2차선 targeting
        # 시간 계산 카운트
        self.count = 0
        # 차선 변경 횟수
        self.num_lane_changes = 0
        # 경로 기울기
        self.slope = None

        ## 서브스크라이버 설정
        self.lane_sub = self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)
        self.car_sub = self.create_subscription(CarInfo, self.sub_car_topic, self.car_callback, self.qos_profile)
        self.front_lidar_sub = self.create_subscription(Bool, self.sub_front_lidar_obstacle_topic, self.front_lidar_callback, self.qos_profile)
        self.front_left_lidar_sub = self.create_subscription(Bool, self.sub_front_left_lidar_obstacle_topic, self.front_left_lidar_callback, self.qos_profile)
        self.front_right_lidar_sub = self.create_subscription(Bool, self.sub_front_right_lidar_obstacle_topic, self.front_right_lidar_callback, self.qos_profile)
        self.rear_left_lidar_sub = self.create_subscription(Bool, self.sub_rear_left_lidar_obstacle_topic, self.rear_left_lidar_callback, self.qos_profile)
        self.rear_right_lidar_sub = self.create_subscription(Bool, self.sub_rear_right_lidar_obstacle_topic, self.rear_right_lidar_callback, self.qos_profile)

        ## 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        ## 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def lane_callback(self, msg: LaneInfo):
        self.lane_data = msg

    def car_callback(self, msg: CarInfo):
        self.car_data = msg

    def front_lidar_callback(self, msg: Bool):
        self.front_lidar_data = msg.data

    def front_left_lidar_callback(self, msg: Bool):
        self.front_left_lidar_data = msg.data

    def front_right_lidar_callback(self, msg: Bool):
        self.front_right_lidar_data = msg.data 
    
    def rear_left_lidar_callback(self, msg: Bool):
        self.rear_left_lidar_data = msg.data

    def rear_right_lidar_callback(self, msg: Bool):
        self.rear_right_lidar_data = msg.data


    def timer_callback(self):
        # 장애물 변수 선언
        front_obs = self.front_lidar_data if self.front_lidar_data is not None else False
        front_left_obs = self.front_left_lidar_data if self.front_left_lidar_data is not None else False
        front_right_obs = self.front_right_lidar_data if self.front_right_lidar_data is not None else False
        rear_left_obs = self.rear_left_lidar_data if self.rear_left_lidar_data is not None else False
        rear_right_obs = self.rear_right_lidar_data if self.rear_right_lidar_data is not None else False
        # 차선 변수 선언
        if self.lane_data is not None:
            lane_x = self.lane_data.lane1_x if self.target_lane == 0 else self.lane_data.lane2_x
        else:
            lane_x = 320

        self.normal_driving()
        print(self.current_state)


        ####################FSM State####################

        ## 일반 주행 State
        if self.current_state == State.NORMAL_DRIVING:
            # 전방 차량 감지 시
            if front_obs and (self.car_data is not None and self.car_data.num_cars != 0):
                current_lane_front_obs = None # 같은 차선 전방 차량만 선택
                for i in range(self.car_data.num_cars):
                    if lane_x - 100 <= self.car_data.x[i] <= lane_x + 30:
                        current_lane_front_obs = int(i)
                # 같은 차선 전방 차량 감지 시, 차선 변경 준비 state로 전이
                if current_lane_front_obs is not None and :
                    self.current_state = State.PREPARING_TO_CHANGE_LANE

        ## 차선 변경 준비 State
        elif self.current_state == State.PREPARING_TO_CHANGE_LANE:
            # 전방 차량 감지 시
            if front_obs and self.car_data.num_cars != 0:
                rear_obs = rear_right_obs if self.current_lane == 0 else rear_left_obs
                front_side_obs = front_right_obs if self.current_lane == 0 else front_left_obs
                side_lane_front_obs = None # 옆 차선 전방 차량만 선택
                for i in range(self.car_data.num_cars):
                    if not (lane_x - 100 <= self.car_data.x[i] <= lane_x + 30):
                        side_lane_front_obs = int(i)
                # 옆 차선 차량 미감지 시, 차선 변경 state로 전이
                if not rear_obs and not front_side_obs and (side_lane_front_obs is None or (side_lane_front_obs is not None and self.car_data.y[side_lane_front_obs] >= 240)):
                    self.target_lane ^= 1
                    self.current_state = State.CHANGING_LANE
            # 전방 차량 미감지 시
            else:
                # 차선 변경을 짝수 번 했을 시, 일반 주행 state로 전이
                if self.num_lane_changes == 0:
                    self.current_state = State.NORMAL_DRIVING
                # 차선 변경을 홀수 번 했을 시, 추월 state로 전이
                else:
                    self.current_state = State.OVERTAKING

        ## 차선 변경 State
        elif self.current_state == State.CHANGING_LANE:
            max_count = self.set_count(self.speed)
            # 차선 변경 완료 시
            if self.count > max_count and 0 <= abs(self.slope) <= 15:
                self.count = 0
                self.current_lane = self.target_lane
                self.num_lane_changes ^= 1
                # 차선 변경을 짝수 번 했을 시, 일반 주행 State로 전이
                if self.num_lane_changes == 0:
                    self.current_state = State.NORMAL_DRIVING
                # 차선 변경을 홀수 번 했을 시, 추월 State로 전이
                else:
                    self.current_state = State.OVERTAKING
            # 차선 변경 진행 중
            else:
                self.count += 1

        ## 추월 State
        elif self.current_state == State.OVERTAKING:
            rear_obs = rear_right_obs if self.current_lane == 0 else rear_left_obs
            front_side_obs = front_right_obs if self.current_lane == 0 else front_left_obs
            current_lane_front_obs = None # 같은 차선 전방 차량만 선택
            side_lane_front_obs = None # 옆 차선 전방 차량만 선택
            for i in range(self.car_data.num_cars):
                if lane_x - 100 <= self.car_data.x[i] <= lane_x + 30:
                    current_lane_front_obs = int(i)
                else:
                    side_lane_front_obs = 1
            # 같은 차선 전방 차량 감지 시, 차선 변경 준비 state로 전이
            if front_obs and (current_lane_front_obs is not None and self.car_data.y[current_lane_front_obs] >= 240):
                self.count = 0
                self.current_state = State.PREPARING_TO_CHANGE_LANE
            # 옆 차선 차량 미감지 시, 차선 복귀 state로 전이
            elif not rear_obs and not front_side_obs and not side_lane_front_obs:
                self.target_lane ^= 1
                self.current_state = State.RETURNING_LANE

        ## 차선 복귀 State
        elif self.current_state == State.RETURNING_LANE:
            max_count = self.set_count(self.speed)
            # 차선 변경 완료 시, 일반 주행 state로 전이
            if self.count > max_count:
                self.count = 0
                self.current_lane = self.target_lane
                self.num_lane_changes = 0
                self.current_state = State.NORMAL_DRIVING
            # 차선 변경 진행 중
            else:
                self.count += 1

        #################################################

        # 모션 명령 메시지 생성 및 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)
        
    def normal_driving(self):
        if self.lane_data is not None:
            lane_x = self.lane_data.lane2_x if self.target_lane == 1 else self.lane_data.lane1_x
            if lane_x is not None:
                ## 같은 차선 전방 차량 감지 시, 감속
                front_obs = self.front_lidar_data if self.front_lidar_data is not None else False
                if self.current_state != State.CHANGING_LANE and front_obs and (self.car_data is not None and self.car_data.num_cars != 0):
                    for i in range(self.car_data.num_cars):
                        if lane_x - 100 <= self.car_data.x[i] <= lane_x + 30:
                            self.speed = self.speed_change(90)
                            break
                else:
                    self.speed = self.speed_change(200)
                
                ## 기본 주행
                # 변수 선언
                target_point = (self.lane_data.lane1_x, self.lane_data.lane1_y) if self.target_lane == 0 else (self.lane_data.lane2_x, self.lane_data.lane2_y)
                self.slope = self.lane_data.slope1 if self.target_lane == 0 else self.lane_data.slope2
                car_center_point = (285, 170) # roi가 잘린 후 차량 앞 범퍼 중앙 위치
                self.target_slope = DMFL.calculate_slope_between_points(target_point, car_center_point)
                steerang = np.rad2deg(np.arctan((2*0.3*np.sin(np.deg2rad(self.target_slope)))/1.38))
                
                if int(steerang*7/20.5) <= -6 or int(steerang*7/20.5) >= 0:
                    self.steering_command = int(steerang*7/20)
                else:
                    self.steering_command = int(steerang*7/20.5)

                # 차선 변경 시 조향 제한
                if np.abs(self.steering_command) > 7:
                    if self.steering_command > 0:
                        self.steering_command = 7
                    else:
                        self.steering_command = -7

                # 조향에 따른 바퀴 속도 설정
                if self.steering_command == -7:
                    self.right_speed_command = self.speed +15 
                    self.left_speed_command = self.speed  -50
                elif self.steering_command ==-6:
                    self.right_speed_command = self.speed  + 6 -15
                    self.left_speed_command = self.speed  -20
                elif self.steering_command ==-5:
                    self.right_speed_command = self.speed  + 3 -8
                    self.left_speed_command = self.speed  -10
                elif self.steering_command ==-4:
                    self.right_speed_command = self.speed  + 1
                    self.left_speed_command = self.speed 
                elif self.steering_command ==-3:
                    self.right_speed_command = self.speed 
                    self.left_speed_command = self.speed 
                elif self.steering_command ==-2:
                    self.right_speed_command = self.speed 
                    self.left_speed_command = self.speed 
                elif self.steering_command ==-1:
                    self.right_speed_command = self.speed 
                    self.left_speed_command = self.speed 
                elif self.steering_command ==0:
                    self.right_speed_command = self.speed  
                    self.left_speed_command = self.speed 
                elif self.steering_command ==1:
                    self.right_speed_command = self.speed  
                    self.left_speed_command = self.speed 
                elif self.steering_command ==2:
                    self.right_speed_command = self.speed 
                    self.left_speed_command = self.speed 
                elif self.steering_command ==3:
                    self.right_speed_command = self.speed 
                    self.left_speed_command = self.speed 
                elif self.steering_command ==4:
                    self.right_speed_command = self.speed  
                    self.left_speed_command = self.speed  + 5
                elif self.steering_command ==5:
                    self.right_speed_command = self.speed  -15
                    self.left_speed_command = self.speed  + 10 -15
                elif self.steering_command ==6:
                    self.right_speed_command = self.speed  -30
                    self.left_speed_command = self.speed  + 15 -30
                else:
                    self.right_speed_command = self.speed  -35 -60
                    self.left_speed_command = self.speed  + 30 -35
            else: # lane이 인식되지 않을 시, 직진
                self.steering_command = 2
        else:     # lane이 인식되지 않을 시, 직진
            self.steering_command = 2

    # 부드럽게 속도 조절
    def speed_change(self, target_speed):
        speed_increment = 5  # 속도 변경 단위
        if self.speed  < target_speed:
            new_speed = self.speed  + speed_increment
            if new_speed > target_speed:
                new_speed = target_speed
        elif self.speed  > target_speed:
            new_speed = self.speed  - speed_increment
            if new_speed < target_speed:
                new_speed = target_speed
        else:
            new_speed = self.speed 

        return new_speed

    # 부드럽게 조향 조절
    def angle_change(self, target_angle):
        steering_increment = 1  # 조향 각도 변경 단위
        if self.steering_command < target_angle:
            new_steering = self.steering_command + steering_increment
            if new_steering > target_angle:
                new_steering = target_angle
        elif self.steering_command > target_angle:
            new_steering = self.steering_command - steering_increment
            if new_steering < target_angle:
                new_steering = target_angle
        else:
            new_steering = self.steering_command

        return new_steering

    # 속도에 따른 차선 변경 및 복귀 시간 설정
    def set_count(self, current_speed):
        if -255 <= current_speed <= 255:
            max_count = int((380 - np.abs(current_speed)) / 2 - 1)
        else:
            max_count = 0
        return max_count

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()