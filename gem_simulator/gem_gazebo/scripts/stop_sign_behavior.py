#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDrive
from vision_msgs.msg import Detection2DArray 
from gazebo_msgs.msg import ModelStates # <--- NEW: The Eye of God
import math                             # <--- NEW: For speed calculation

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
        self.PED_AREA_THRESHOLD = 300      
        self.PED_PATIENCE_TIME = 3.0

        self.STRIKE_ZONE_LEFT = 300.0       
        self.STRIKE_ZONE_RIGHT = 980.0

        # --- State Tracking ---
        self.state = "DRIVING" # Can be: DRIVING, STOPPING, COOLDOWN, YIELDING
        self.state_start_time = 0.0
        self.last_pedestrian_time = 0.0    # Tracks the exact moment we last saw a person
        self.current_speed = 1.5

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback) # <--- NEW: The Eye of God

        rospy.Subscriber('/yolo/detections', Detection2DArray, self.yolo_callback)
        rospy.Subscriber('/dwa_cmd', AckermannDrive, self.dwa_callback)
        self.cmd_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=10)
        
        self.rate = rospy.Rate(100)

    def model_states_callback(self, msg):
        try:
            # Search the simulation for the vehicle (usually named 'gem')
            for i, name in enumerate(msg.name):
                if "gem" in name:
                    # Get the raw X and Y velocities
                    vx = msg.twist[i].linear.x
                    vy = msg.twist[i].linear.y
                    
                    # Calculate absolute forward speed (always positive)
                    self.current_speed = math.hypot(vx, vy)
                    break
        except Exception:
            pass

    def yolo_callback(self, msg):
        if len(msg.detections) == 0:
            return 

        for det in msg.detections:
            if len(det.results) > 0:
                detected_id = int(det.results[0].id)
                area = det.bbox.size_x * det.bbox.size_y
                center_x = det.bbox.center.x

                # rospy.loginfo_throttle(0.5, f"[VISION DEBUG] ID: {detected_id} | Area: {area:.1f} | X: {center_x:.1f}")
                
                # ---------------------------------------------------
                # PRIORITY 1: PEDESTRIANS (Supreme Override)
                # We care about humans EVEN IF we are in a Stop Sign Cooldown!
                # ---------------------------------------------------
                if detected_id == 0 and area > self.PED_AREA_THRESHOLD:
                    if self.STRIKE_ZONE_LEFT < center_x < self.STRIKE_ZONE_RIGHT:
                        # Reset the patience timer
                        self.last_pedestrian_time = rospy.get_time()
                        
                        # Instantly override any other state (DRIVING or COOLDOWN)
                        if self.state != "YIELDING":
                            rospy.logwarn(f"*** PEDESTRIAN IN ROAD! Severing DWA! ***")
                            self.state = "YIELDING"
                        
                        # Skip reading any other objects (like signs) until the human is safe
                        return 

                # ---------------------------------------------------
                # PRIORITY 2: STOP SIGNS (Low Priority)
                # ---------------------------------------------------
                elif detected_id == 11 and area > self.STOP_AREA_THRESHOLD:
                    # ONLY trigger a stop sign if we are in normal DRIVING mode.
                    # If we are in COOLDOWN, we ignore it.
                    if self.state == "DRIVING": 
                        rospy.logwarn(f"*** STOP SIGN! Area: {area:.2f}. BRAKING! ***")
                        self.state = "STOPPING"
                        self.state_start_time = rospy.get_time()

    def dwa_callback(self, msg):
        # MUX LOGIC: Only pass DWA commands if it is perfectly safe
        if self.state == "DRIVING" or self.state == "COOLDOWN":
            self.cmd_pub.publish(msg)

    def run(self):
        rospy.loginfo("Entering main control loop. Forwarding DWA commands...")
        while not rospy.is_shutdown():
            current_time = rospy.get_time()

            # --- STOP SIGN EXECUTION ---
            if self.state == "STOPPING":
                # Check if we are physically stopped (velocity near 0)
                if abs(self.current_speed) > 0.1:
                    # CLOSED-LOOP: Still rolling! Keep resetting the clock so the 3s timer never starts.
                    self.state_start_time = current_time
                    rospy.loginfo_throttle(0.5, f"[BRAKING] Slowing down... Speed: {self.current_speed:.2f} m/s")
                    self.slam_brakes()
                else:
                    # WE HAVE FULLY STOPPED. NOW we start the 3-second countdown.
                    elapsed_time = current_time - self.state_start_time
                    if elapsed_time >= self.STOP_DURATION:
                        rospy.loginfo("Done stopping for sign. Entering cooldown...")
                        self.state = "COOLDOWN"
                        self.state_start_time = current_time
                    else:
                        rospy.loginfo_throttle(0.5, f"[BRAKING] Fully stopped! Holding... {elapsed_time:.1f} / {self.STOP_DURATION} sec")
                        self.slam_brakes()

            # --- STOP SIGN COOLDOWN ---
            elif self.state == "COOLDOWN":
                elapsed_cooldown = current_time - self.state_start_time
                if elapsed_cooldown >= self.COOLDOWN_DURATION:
                    rospy.loginfo("Cooldown complete.")
                    self.state = "DRIVING"

            # --- PEDESTRIAN EXECUTION ---
            elif self.state == "YIELDING":
                time_since_last_seen = current_time - self.last_pedestrian_time
                
                # The Patience Timer: Have they been gone for a full second?
                if time_since_last_seen > self.PED_PATIENCE_TIME:
                    rospy.loginfo("Pedestrian cleared the road. Resuming DWA...")
                    self.state = "DRIVING"
                else:
                    rospy.loginfo_throttle(0.5, "[YIELDING] Waiting for pedestrian to cross...")
                    self.slam_brakes()

            self.rate.sleep()

    def slam_brakes(self):
        stop_cmd = AckermannDrive()
        stop_cmd.steering_angle = 0.0
        
        if self.current_speed > 0.15:
            # We are rolling forward. Hit the brakes!
            stop_cmd.speed = -4.0  
        else:
            # We hit 0.0 m/s. INSTANTLY let off the brakes to prevent going into reverse!
            stop_cmd.speed = 0.0   
            
        self.cmd_pub.publish(stop_cmd)

if __name__ == '__main__':
    try:
        node = AutonomousMux()
        node.run()
    except rospy.ROSInterruptException:
        pass