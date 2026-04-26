#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDrive
from vision_msgs.msg import Detection2DArray 

class StopSignBehavior:
    def __init__(self):
        rospy.init_node('stop_sign_behavior_node')
        
        rospy.loginfo("========================================")
        rospy.loginfo("AUTONOMOUS MULTIPLEXER (MUX) INITIALIZED!")
        rospy.loginfo("========================================")
        
        self.STOP_AREA_THRESHOLD = 5000  
        self.STOP_DURATION = 3.0          
        self.COOLDOWN_DURATION = 10.0     
        
        self.state = "DRIVING"
        self.state_start_time = 0.0

        # 1. Listen to YOLO
        rospy.Subscriber('/yolo/detections', Detection2DArray, self.yolo_callback)
        
        # 2. Listen to DWA (The remapped private channel)
        rospy.Subscriber('/dwa_cmd', AckermannDrive, self.dwa_callback)
        
        # 3. The ONLY node allowed to talk to the car's wheels
        self.cmd_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=10)
        
        self.rate = rospy.Rate(50)

    def yolo_callback(self, msg):
        if self.state != "DRIVING":
            return

        if len(msg.detections) == 0:
            return 

        for det in msg.detections:
            if len(det.results) > 0:
                detected_id = int(det.results[0].id)
                area = det.bbox.size_x * det.bbox.size_y
                
                if detected_id == 11 and area > self.STOP_AREA_THRESHOLD:
                    rospy.logwarn(f"*** STOP SIGN CLOSE! Area: {area:.2f}. SEVERING DWA CONTROL! ***")
                    self.state = "STOPPING"
                    self.state_start_time = rospy.get_time()
                    return 

    def dwa_callback(self, msg):
        # MUX LOGIC: If we are not stopping, forward DWA's driving commands directly to the car
        if self.state == "DRIVING" or self.state == "COOLDOWN":
            self.cmd_pub.publish(msg)

    def run(self):
        rospy.loginfo("Entering main control loop. Forwarding DWA commands...")
        while not rospy.is_shutdown():
            current_time = rospy.get_time()

            if self.state == "STOPPING":
                elapsed_time = current_time - self.state_start_time
                if elapsed_time >= self.STOP_DURATION:
                    rospy.loginfo("Done stopping. Reconnecting DWA control. Entering cooldown...")
                    self.state = "COOLDOWN"
                    self.state_start_time = current_time
                else:
                    rospy.loginfo_throttle(0.5, f"[BRAKING] Holding brakes... {elapsed_time:.1f} / {self.STOP_DURATION} sec")
                    
                    stop_cmd = AckermannDrive()
                    stop_cmd.speed = 0.0
                    stop_cmd.steering_angle = 0.0
                    stop_cmd.acceleration = -10.0  # <--- NEW: Aggressive physical braking force!
                    stop_cmd.jerk = -10.0          # <--- NEW: Apply it instantly!
                    self.cmd_pub.publish(stop_cmd)

            elif self.state == "COOLDOWN":
                elapsed_cooldown = current_time - self.state_start_time
                if elapsed_cooldown >= self.COOLDOWN_DURATION:
                    rospy.loginfo("Cooldown complete. Looking for stop signs again.")
                    self.state = "DRIVING"

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = StopSignBehavior()
        node.run()
    except rospy.ROSInterruptException:
        pass