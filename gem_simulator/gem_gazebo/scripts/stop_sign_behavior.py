#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDrive
from vision_msgs.msg import Detection2DArray 
from gazebo_msgs.msg import ModelStates 
import math                             

class AutonomousMux:
    def __init__(self):
        rospy.init_node('autonomous_mux_node')
        
        rospy.loginfo("========================================")
        rospy.loginfo("V2 MUX: STOP SIGNS & PEDESTRIANS ACTIVE!")
        rospy.loginfo("========================================")
        
        # --- Stop Sign Config ---
        self.STOP_AREA_THRESHOLD = 6000  
        self.STOP_DURATION = 3.0          
        self.COOLDOWN_DURATION = 25.0     
        
        # --- Pedestrian Config ---
        self.PED_AREA_THRESHOLD = 5000      
        self.PED_PATIENCE_TIME = 3.0

        # Strike zone covering the lane width
        self.STRIKE_ZONE_LEFT = 400.0       
        self.STRIKE_ZONE_RIGHT = 700.0

        # --- State Tracking ---
        self.state = "DRIVING" # DRIVING, STOPPING, COOLDOWN, YIELDING
        self.state_start_time = 0.0
        self.last_pedestrian_time = 0.0    
        self.current_speed = 1.5
        
        # Latch flag to prevent runaway reverse braking due to simulator noise
        self.has_stopped = False

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.Subscriber('/yolo/detections', Detection2DArray, self.yolo_callback)
        rospy.Subscriber('/dwa_cmd', AckermannDrive, self.dwa_callback)
        
        self.cmd_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=10)
        self.rate = rospy.Rate(100)

    def model_states_callback(self, msg):
        try:
            for i, name in enumerate(msg.name):
                if "gem" in name:
                    vx = msg.twist[i].linear.x
                    vy = msg.twist[i].linear.y
                    self.current_speed = math.hypot(vx, vy)
                    break
        except Exception:
            pass

    def yolo_callback(self, msg):
        if len(msg.detections) == 0:
            return 

        # --- PRIORITY 1: EVALUATE PEDESTRIANS FIRST ---
        pedestrian_detected = False
        for det in msg.detections:
            if len(det.results) > 0:
                detected_id = int(det.results[0].id)
                area = det.bbox.size_x * det.bbox.size_y
                center_x = det.bbox.center.x

                if detected_id == 0 and area > self.PED_AREA_THRESHOLD:
                    if self.STRIKE_ZONE_LEFT < center_x < self.STRIKE_ZONE_RIGHT:
                        self.last_pedestrian_time = rospy.get_time()
                        pedestrian_detected = True
                        
                        if self.state != "YIELDING":
                            rospy.logwarn("*** PEDESTRIAN IN ROAD! Severing DWA & Locking Brakes! ***")
                            self.state = "YIELDING"
                            self.has_stopped = False  # Reset latch for new yield event
                        break 

        # Supreme Override: Ignore stop signs entirely if yielding to a pedestrian
        if pedestrian_detected:
            return

        # --- PRIORITY 2: EVALUATE STOP SIGNS ---
        for det in msg.detections:
            if len(det.results) > 0:
                detected_id = int(det.results[0].id)
                area = det.bbox.size_x * det.bbox.size_y

                if detected_id == 11 and area > self.STOP_AREA_THRESHOLD:
                    if self.state == "DRIVING": 
                        rospy.logwarn(f"*** STOP SIGN! Area: {area:.2f}. BRAKING! ***")
                        self.state = "STOPPING"
                        self.state_start_time = rospy.get_time()
                        self.has_stopped = False  # Reset latch for new stop event
                        break

    def dwa_callback(self, msg):
        if self.state == "DRIVING" or self.state == "COOLDOWN":
            self.cmd_pub.publish(msg)

    def run(self):
        rospy.loginfo("Entering main control loop. Forwarding DWA commands...")
        while not rospy.is_shutdown():
            current_time = rospy.get_time()

            # --- STOP SIGN EXECUTION ---
            if self.state == "STOPPING":
                if not self.has_stopped:
                    if self.current_speed > 0.1:
                        # Keep holding start time current until physical stop is reached
                        self.state_start_time = current_time
                        rospy.loginfo_throttle(0.5, f"[BRAKING] Slowing down... Speed: {self.current_speed:.2f} m/s")
                        self.slam_brakes()
                    else:
                        self.has_stopped = True
                        rospy.loginfo("[BRAKING] Initial stop achieved. Starting 3-second hold.")
                        self.hold_position()
                else:
                    # Latch engaged: smoothly process the remaining hold time
                    elapsed_time = current_time - self.state_start_time
                    if elapsed_time >= self.STOP_DURATION:
                        rospy.loginfo("Done stopping for sign. Entering cooldown...")
                        self.state = "COOLDOWN"
                        self.state_start_time = current_time
                    else:
                        rospy.loginfo_throttle(0.5, f"[BRAKING] Fully stopped! Holding... {elapsed_time:.1f} / {self.STOP_DURATION} sec")
                        self.hold_position()

            # --- STOP SIGN COOLDOWN ---
            elif self.state == "COOLDOWN":
                elapsed_cooldown = current_time - self.state_start_time
                if elapsed_cooldown >= self.COOLDOWN_DURATION:
                    rospy.loginfo("Cooldown complete.")
                    self.state = "DRIVING"

            # --- PEDESTRIAN EXECUTION ---
            elif self.state == "YIELDING":
                time_since_last_seen = current_time - self.last_pedestrian_time
                
                if time_since_last_seen > self.PED_PATIENCE_TIME:
                    rospy.loginfo("Pedestrian cleared the road. Resuming DWA...")
                    self.state = "DRIVING"
                else:
                    if not self.has_stopped:
                        if self.current_speed > 0.1:
                            rospy.loginfo_throttle(0.5, f"[YIELDING] Braking for pedestrian... Speed: {self.current_speed:.2f} m/s")
                            self.slam_brakes()
                        else:
                            self.has_stopped = True
                            rospy.loginfo("[YIELDING] Fully stopped. Holding steady.")
                            self.hold_position()
                    else:
                        # Latch engaged: perfectly silent zero-command hold regardless of jitter
                        # rospy.loginfo_throttle(0.5, "[YIELDING] Fully stopped. Waiting for pedestrian to cross...")
                        self.hold_position()

            self.rate.sleep()

    def slam_brakes(self):
        stop_cmd = AckermannDrive()
        stop_cmd.steering_angle = 0.0
        stop_cmd.speed = -4.0  
        self.cmd_pub.publish(stop_cmd)

    def hold_position(self):
        stop_cmd = AckermannDrive()
        stop_cmd.steering_angle = 0.0
        stop_cmd.speed = 0.0   
        self.cmd_pub.publish(stop_cmd)

if __name__ == '__main__':
    try:
        node = AutonomousMux()
        node.run()
    except rospy.ROSInterruptException:
        pass