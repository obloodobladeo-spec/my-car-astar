"""Follow an A* route while checking the live laser scan before driving."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from astar_pursuit import AStarPursuit, world_to_cell


def scan_sectors(scan):
    """Find nearest valid front, left, and right rays by their angles."""
    nearest = [math.inf, math.inf, math.inf]
    front_angle = None
    for index, measured in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if abs(angle) <= math.radians(30):
            sector = 0
        elif math.radians(30) < angle <= math.radians(100):
            sector = 1
        elif -math.radians(100) <= angle < -math.radians(30):
            sector = 2
        else:
            continue
        if math.isinf(measured) and measured > 0:
            measured = scan.range_max
        if (math.isfinite(measured) and
                scan.range_min <= measured <= scan.range_max and
                measured < nearest[sector]):
            nearest[sector] = measured
            if sector == 0:
                front_angle = angle
    return tuple(None if math.isinf(value) else value for value in nearest), \
        front_angle


def front_distance(scan):
    """Return the nearest valid range within 30 degrees of straight ahead."""
    sectors, _ = scan_sectors(scan)
    return sectors[0]


def add_obstacle_disk(blocked, map_info, x, y, radius):
    """Block cells around a laser obstacle in a copy of the map."""
    row, col = world_to_cell(x, y, *map_info)
    height, width = blocked.shape
    if not (0 <= row < height and 0 <= col < width):
        return
    radius_cells = math.ceil(radius / map_info[3])
    row0 = max(0, row - radius_cells)
    row1 = min(height, row + radius_cells + 1)
    col0 = max(0, col - radius_cells)
    col1 = min(width, col + radius_cells + 1)
    rows, cols = np.ogrid[row0:row1, col0:col1]
    disk = ((rows - row) ** 2 + (cols - col) ** 2) * map_info[3] ** 2
    blocked[row0:row1, col0:col1] |= disk <= radius ** 2


class BestDriver(AStarPursuit):
    """Turn toward open space, then replan with the sensed obstacle."""

    def __init__(self):
        super().__init__(node_name='best_driver_node')
        self.declare_parameter('front_stop_distance', 0.32)
        self.declare_parameter('front_resume_distance', 0.45)
        self.declare_parameter('avoid_turn_speed', 0.7)
        self.declare_parameter('scan_timeout', 1.0)
        self.front_range = None
        self.front_angle = None
        self.left_range = None
        self.right_range = None
        self.scan_received_at = None
        self.last_warning_at = 0.0
        self.avoiding = False
        self.avoid_turn_sign = 0
        self.clear_scan_count = 0
        self.static_blocked = None
        self.obstacle_points = []
        self.mouse_detected = False
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(
            Bool, '/mouse_detected', self.mouse_callback, 10)
        self.get_logger().info(
            'A* + steering + /scan safety ready; goal topic: /astar_goal')

    def scan_callback(self, msg):
        sectors, self.front_angle = scan_sectors(msg)
        self.front_range, self.left_range, self.right_range = sectors
        self.scan_received_at = time.monotonic()
        resume_distance = max(
            float(self.get_parameter('front_resume_distance').value),
            float(self.get_parameter('front_stop_distance').value) + 0.05)
        if self.avoiding and self.front_range is not None:
            if self.front_range >= resume_distance:
                self.clear_scan_count += 1
            else:
                self.clear_scan_count = 0

    def mouse_callback(self, msg):
        """Stop immediately when the camera detector sees a mouse."""
        was_detected = self.mouse_detected
        self.mouse_detected = bool(msg.data)
        if self.mouse_detected:
            self.stop_robot()
            if not was_detected:
                self.get_logger().warn('Mouse detected; robot stopped')
        elif was_detected:
            self.get_logger().info('Mouse cleared; route following resumed')

    def map_callback(self, msg):
        old_map = self.map_data
        super().map_callback(msg)
        if self.map_data is not old_map:
            self.static_blocked = self.blocked.copy()
            self.obstacle_points = []

    def goal_callback(self, msg):
        if msg.header.frame_id != 'map':
            super().goal_callback(msg)
            return
        super().goal_callback(msg)
        self.avoiding = False
        self.clear_scan_count = 0
        self.obstacle_points = []
        if self.static_blocked is not None:
            self.blocked = self.static_blocked.copy()

    def scan_is_fresh(self):
        """Require recent scan data before allowing autonomous movement."""
        timeout = max(0.1, float(self.get_parameter('scan_timeout').value))
        return (self.scan_received_at is not None and
                self.front_range is not None and
                time.monotonic() - self.scan_received_at <= timeout)

    def control_loop(self):
        if self.goal is None:
            return
        if self.mouse_detected:
            self.stop_robot()
            return
        if not self.scan_is_fresh():
            self.stop_robot()
            self.warn_sparingly('Waiting for a fresh, valid /scan')
            return
        if self.avoiding:
            self.avoidance_step()
            return
        super().control_loop()

    def publish_drive_command(self, cmd):
        """Enter the turning state instead of repeatedly stopping."""
        if self.mouse_detected:
            self.stop_robot()
            return
        stop_distance = max(
            0.0, float(self.get_parameter('front_stop_distance').value))
        if cmd.linear.x > 0.0 and self.front_range < stop_distance:
            self.start_avoidance()
            self.warn_sparingly(
                f'Obstacle {self.front_range:.2f} m ahead; '
                'turning toward open space')
            self.avoidance_step()
            return
        self.cmd_pub.publish(cmd)

    def start_avoidance(self):
        """Remember the laser hit so A* can avoid it after turning."""
        self.avoiding = True
        self.clear_scan_count = 0
        left = self.left_range if self.left_range is not None else -1.0
        right = self.right_range if self.right_range is not None else -1.0
        self.avoid_turn_sign = 1 if left >= right else -1
        pose = self.current_pose()
        if pose is not None and self.front_angle is not None:
            angle = pose[2] + self.front_angle
            hit_x = pose[0] + self.front_range * math.cos(angle)
            hit_y = pose[1] + self.front_range * math.sin(angle)
            self.obstacle_points.append((hit_x, hit_y))
            self.obstacle_points = self.obstacle_points[-10:]

    def avoidance_step(self):
        """Rotate until several scans show an open forward direction."""
        if self.clear_scan_count >= 3:
            self.finish_avoidance()
            return
        side = (self.left_range if self.avoid_turn_sign > 0
                else self.right_range)
        clearance = max(
            0.0, float(self.get_parameter('robot_clearance').value))
        if side is None or side < clearance:
            self.stop_robot()
            self.warn_sparingly('No safe side reading for avoidance turn')
            return
        cmd = Twist()
        cmd.angular.z = self.avoid_turn_sign * max(
            0.0, float(self.get_parameter('avoid_turn_speed').value))
        self.cmd_pub.publish(cmd)

    def finish_avoidance(self):
        """Plan again with laser hits added to the map."""
        self.stop_robot()
        self.avoiding = False
        self.clear_scan_count = 0
        if self.static_blocked is not None:
            self.blocked = self.static_blocked.copy()
            radius = max(
                0.0, float(self.get_parameter('robot_clearance').value))
            for x, y in self.obstacle_points:
                add_obstacle_disk(self.blocked, self.map_info, x, y, radius)
        self.path = []
        self.path_index = 0
        self.plan_needed = True
        self.route_blocked = False
        self.replan_attempted = False
        self.replan_pose = None
        self.waiting_for_clear_pose = False
        self.clear_pose_count = 0
        self.get_logger().info('Front clear; replanning from current pose')

    def warn_sparingly(self, message):
        now = time.monotonic()
        if now - self.last_warning_at >= 2.0:
            self.get_logger().warn(message)
            self.last_warning_at = now


def main(args=None):
    """Run the separate best-driver executable."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = BestDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
