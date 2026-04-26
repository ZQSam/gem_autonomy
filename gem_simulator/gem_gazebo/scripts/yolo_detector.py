#!/usr/bin/env python3

#================================================================
# File name: yolo_detector.py
# Description: Lightweight YOLOv8 nano object detection node.
#              Subscribes to a camera topic, runs inference, and
#              publishes an annotated image and overlay text for
#              RViz visualization.
# Author: Henry Che
# Usage: rosrun gem_gazebo yolo_detector.py
#        roslaunch gem_gazebo yolo_detector.launch
# Python version: 3.8
#================================================================

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from jsk_rviz_plugins.msg import OverlayText
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Optional: vision_msgs for structured detection output
try:
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False


class YoloDetectorNode:

    def __init__(self):

        if not YOLO_AVAILABLE:
            rospy.logfatal("ultralytics is not installed. Run: pip install ultralytics")
            raise SystemExit(1)

        # Parameters
        self.model_name  = rospy.get_param('~model',       'yolov8n.pt')
        self.conf_thresh = rospy.get_param('~confidence',  0.4)
        self.img_size    = rospy.get_param('~img_size',    320)   # smaller = faster
        self.device      = rospy.get_param('~device',      'cpu')
        self.image_topic = rospy.get_param('~image_topic', '/oak/rgb/image_raw')

        rospy.loginfo(f"Loading YOLO model: {self.model_name}  (img_size={self.img_size}, device={self.device})")
        self.model = YOLO(self.model_name)
        # Warm-up pass to avoid latency on first real frame
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.img_size, device=self.device, verbose=False)
        rospy.loginfo("YOLO model ready.")

        self.bridge = CvBridge()

        # Publishers
        self.pub_image = rospy.Publisher('/yolo/image_annotated', Image, queue_size=1)
        self.pub_overlay = rospy.Publisher('/yolo/detection_info', OverlayText, queue_size=1)

        if VISION_MSGS_AVAILABLE:
            self.pub_detections = rospy.Publisher('/yolo/detections', Detection2DArray, queue_size=1)
            rospy.loginfo("vision_msgs available — publishing Detection2DArray on /yolo/detections")
        else:
            self.pub_detections = None
            rospy.logwarn("vision_msgs not found; skipping /yolo/detections. "
                          "Install with: sudo apt install ros-noetic-vision-msgs")

        # Subscribe — queue_size=1 drops stale frames so we always process the latest
        self.sub = rospy.Subscriber(
            self.image_topic, Image,
            self.image_callback,
            queue_size=1, buff_size=2 ** 24
        )

        rospy.loginfo(f"Subscribed to {self.image_topic}")
        rospy.loginfo("Add Image display in RViz pointing to /yolo/image_annotated")
        rospy.loginfo("Add OverlayText display in RViz pointing to /yolo/detection_info")

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # Run inference
        results = self.model(
            cv_image,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        boxes = results[0].boxes
        n_detections = len(boxes) if boxes is not None else 0

        # ---- Annotated image ----------------------------------------
        annotated = results[0].plot()  # draws boxes + labels + conf
        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header = msg.header
            self.pub_image.publish(ann_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish annotated image: {e}")

        # ---- OverlayText summary ------------------------------------
        overlay = self._build_overlay(msg.header.stamp, n_detections, boxes, results[0].names)
        self.pub_overlay.publish(overlay)

        # ---- vision_msgs/Detection2DArray (optional) ----------------
        if self.pub_detections is not None and boxes is not None:
            det_array = Detection2DArray()
            det_array.header = msg.header
            for box in boxes:
                det = Detection2D()
                det.header = msg.header

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det.bbox.center.x = (x1 + x2) / 2.0
                det.bbox.center.y = (y1 + y2) / 2.0
                det.bbox.size_x   = x2 - x1
                det.bbox.size_y   = y2 - y1

                hyp       = ObjectHypothesisWithPose()
                # hyp.id    = str(int(box.cls[0].item()))
                hyp.id    = int(box.cls[0].item())
                hyp.score = float(box.conf[0].item())
                det.results.append(hyp)

                det_array.detections.append(det)
            self.pub_detections.publish(det_array)

        rospy.loginfo_throttle(5.0, f"[YOLO] {n_detections} object(s) detected")

    # ------------------------------------------------------------------
    def _build_overlay(self, stamp, n_detections, boxes, names):
        text = OverlayText()
        text.width      = 280
        text.height     = 400
        text.left       = 10
        text.top        = 10
        text.text_size  = 11
        text.line_width = 2
        text.font       = "DejaVu Sans Mono"
        text.fg_color   = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0)
        text.bg_color   = ColorRGBA(0.0, 0.0, 0.0, 0.3)

        lines = [
            "---- YOLO Detections ----",
            f"Model : {self.model_name}",
            f"Count : {n_detections}",
            "-------------------------",
        ]

        if boxes is not None and n_detections > 0:
            for box in boxes:
                cls_id  = int(box.cls[0].item())
                label   = names.get(cls_id, str(cls_id))
                conf    = float(box.conf[0].item())
                lines.append(f"  {label:<15} {conf:.2f}")
        else:
            lines.append("  (none)")

        text.text = "\n".join(lines)
        return text


# ----------------------------------------------------------------------
def main():
    rospy.init_node('yolo_detector', anonymous=True)
    try:
        node = YoloDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
