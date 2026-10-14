"""Detect only the COCO computer-mouse class in TurtleBot camera images."""

import os
from threading import Lock

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


MOUSE_CLASS_ID = 64  # COCO's "mouse" means a computer mouse.


def image_to_bgr(msg):
    """Convert common raw ROS image encodings without cv_bridge."""
    conversions = {
        'bgr8': (3, None),
        'rgb8': (3, cv2.COLOR_RGB2BGR),
        'bgra8': (4, cv2.COLOR_BGRA2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
        'mono8': (1, cv2.COLOR_GRAY2BGR),
    }
    encoding = msg.encoding.lower()
    if encoding not in conversions:
        raise ValueError(f'Unsupported camera encoding: {msg.encoding}')
    channels, conversion = conversions[encoding]
    row_bytes = msg.width * channels
    if msg.height == 0 or msg.width == 0 or msg.step < row_bytes:
        raise ValueError('Invalid camera image dimensions or row stride')
    byte_count = msg.height * msg.step
    if len(msg.data) < byte_count:
        raise ValueError('Camera image data is shorter than its dimensions')
    data = np.frombuffer(msg.data, dtype=np.uint8, count=byte_count)
    pixels = data.reshape(msg.height, msg.step)[:, :row_bytes]
    shape = (msg.height, msg.width, channels)
    pixels = np.ascontiguousarray(pixels.reshape(shape))
    if channels == 1:
        pixels = pixels[:, :, 0]
    return cv2.cvtColor(pixels, conversion) if conversion else pixels


class MouseDetector(Node):
    """Publish whether the latest camera frame contains a computer mouse."""

    def __init__(self):
        super().__init__('mouse_detector_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path',
                               '/home/kim/yolo_webcam/yolov8n.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('process_period', 0.05)
        self.declare_parameter('image_size', 416)

        model_path = os.path.expanduser(
            self.get_parameter('model_path').value)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'YOLO model not found: {model_path}')
        # Import here so the driving node needs no YOLO installation.
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        if self.model.names.get(MOUSE_CLASS_ID) != 'mouse':
            raise ValueError('Model class 64 must be the COCO mouse class')

        self.latest_image = None
        self.image_lock = Lock()
        self.detected = False
        self.missed_frames = 0
        self.camera_group = MutuallyExclusiveCallbackGroup()
        self.inference_group = MutuallyExclusiveCallbackGroup()
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(
            Image, image_topic, self.image_callback, camera_qos,
            callback_group=self.camera_group)
        self.status_pub = self.create_publisher(Bool, '/mouse_detected', 10)
        period = max(0.05, float(self.get_parameter('process_period').value))
        self.create_timer(
            period, self.process_latest_image,
            callback_group=self.inference_group)
        self.get_logger().info(
            f'Detecting COCO mouse (class 64) on {image_topic}')

    def image_callback(self, msg):
        """Keep only the newest camera frame for the next inference."""
        with self.image_lock:
            self.latest_image = msg

    def process_latest_image(self):
        """Run YOLO on one frame and publish a smoothed detection state."""
        with self.image_lock:
            msg = self.latest_image
            self.latest_image = None
        if msg is None:
            return
        try:
            frame = image_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        confidence = float(self.get_parameter('confidence').value)
        image_size = int(self.get_parameter('image_size').value)
        results = self.model.predict(
            frame, classes=[MOUSE_CLASS_ID], conf=confidence,
            imgsz=image_size,
            verbose=False)
        found = bool(results and len(results[0].boxes) > 0)
        if found:
            self.missed_frames = 0
            if not self.detected:
                self.get_logger().warn('Computer mouse detected')
            self.detected = True
        else:
            self.missed_frames += 1
            if self.detected and self.missed_frames >= 3:
                self.get_logger().info('Computer mouse no longer detected')
                self.detected = False
        self.status_pub.publish(Bool(data=self.detected))


def main(args=None):
    """Run the camera detector with the existing YOLO virtual environment."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MouseDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
