"""Plan on an occupancy grid with A* and follow it with Pure Pursuit."""

import heapq
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def world_to_cell(x, y, origin_x, origin_y, origin_yaw, resolution):
    """Convert a map-frame point to a (row, column) occupancy-grid cell."""
    dx, dy = x - origin_x, y - origin_y
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return (math.floor((-s * dx + c * dy) / resolution),
            math.floor((c * dx + s * dy) / resolution))


def cell_to_world(row, col, origin_x, origin_y, origin_yaw, resolution):
    """Return the map-frame coordinates of the center of a grid cell."""
    gx, gy = (col + 0.5) * resolution, (row + 0.5) * resolution
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return origin_x + c * gx - s * gy, origin_y + s * gx + c * gy


def inflate_obstacles(data, resolution, radius, occupied_threshold=50):
    """Mark unknown and occupied cells plus the robot's required clearance."""
    blocked = (data < 0) | (data >= occupied_threshold)
    radius_cells = math.ceil(radius / resolution)
    if radius_cells == 0:
        return blocked
    height, width = blocked.shape
    padded = np.pad(blocked, radius_cells, constant_values=True)
    inflated = np.zeros_like(blocked)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if math.hypot(dx, dy) * resolution > radius:
                continue
            y0, x0 = radius_cells + dy, radius_cells + dx
            inflated |= padded[y0:y0 + height, x0:x0 + width]
    return inflated


def astar(blocked, start, goal):
    """Find an eight-connected grid route without cutting blocked corners."""
    height, width = blocked.shape
    for row, col in (start, goal):
        if not (0 <= row < height and 0 <= col < width):
            return None
        if blocked[row, col]:
            return None

    def heuristic(cell):
        dy, dx = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return dy + dx + (math.sqrt(2) - 2) * min(dy, dx)

    frontier = [(heuristic(start), 0, 0.0, start)]
    best_cost = {start: 0.0}
    parent = {}
    serial = 0
    while frontier:
        _, _, cost, current = heapq.heappop(frontier)
        if cost > best_cost[current] + 1e-9:
            continue
        if current == goal:
            route = [current]
            while current in parent:
                current = parent[current]
                route.append(current)
            return route[::-1]

        row, col = current
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nr, nc = row + dy, col + dx
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if blocked[nr, nc]:
                    continue
                if dx and dy and (blocked[row, nc] or blocked[nr, col]):
                    continue
                neighbor = (nr, nc)
                new_cost = cost + (math.sqrt(2) if dx and dy else 1.0)
                if new_cost >= best_cost.get(neighbor, math.inf):
                    continue
                best_cost[neighbor] = new_cost
                parent[neighbor] = current
                serial += 1
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic(neighbor), serial, new_cost,
                     neighbor))
    return None


def trajectory_is_clear(blocked, map_info, pose, linear, angular, distance):
    """Check the robot center along the arc produced by a Twist command."""
    x, y, yaw = pose
    step = map_info[3] / 10.0
    samples = max(1, math.ceil(distance / step))
    curvature = angular / linear if linear > 0 else 0.0
    for i in range(samples + 1):
        traveled = distance * i / samples
        if abs(curvature) < 1e-9:
            px = x + traveled * math.cos(yaw)
            py = y + traveled * math.sin(yaw)
        else:
            angle = yaw + curvature * traveled
            px = x + (math.sin(angle) - math.sin(yaw)) / curvature
            py = y + (math.cos(yaw) - math.cos(angle)) / curvature
        row, col = world_to_cell(px, py, *map_info)
        if not (0 <= row < blocked.shape[0] and
                0 <= col < blocked.shape[1] and
                not blocked[row, col]):
            return False
    return True


class AStarPursuit(Node):
    """Turn a map and a goal into a safe grid route and velocity commands."""

    def __init__(self, node_name='astar_pursuit'):
        super().__init__(node_name)
        self.declare_parameter('linear_velocity', 0.08)
        self.declare_parameter('lookahead_distance', 0.35)
        # Waffle Pi base footprint reaches about 0.24 m from base_link.
        self.declare_parameter('robot_clearance', 0.24)
        self.declare_parameter('goal_tolerance', 0.15)

        self.map_data = None
        self.blocked = None
        self.map_info = None
        self.goal = None
        self.path = []
        self.path_index = 0
        self.plan_needed = False
        self.pose_seen = False
        self.finished = False
        self.route_blocked = False
        self.replan_attempted = False
        self.replan_pose = None
        self.waiting_for_clear_pose = False
        self.clear_pose_count = 0

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback,
            map_qos)
        # Nav2 already listens on /goal_pose. A separate topic avoids two
        # controllers issuing commands when a Nav2 goal is selected in RViz.
        self.create_subscription(
            PoseStamped, '/astar_goal', self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Waiting for /map, /amcl_pose, and /astar_goal')

    def map_callback(self, msg):
        if (msg.info.width == 0 or msg.info.height == 0 or
                msg.info.resolution <= 0):
            self.get_logger().error(
                'Received invalid map dimensions or resolution')
            return
        data = np.asarray(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        q = msg.info.origin.orientation
        origin_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))
        info = (msg.info.origin.position.x, msg.info.origin.position.y,
                origin_yaw, msg.info.resolution)
        if (self.map_info == info and self.map_data is not None and
                np.array_equal(self.map_data, data)):
            return
        self.map_info = info
        self.map_data = data
        self.blocked = inflate_obstacles(
            data, msg.info.resolution,
            max(0.0, self.get_parameter('robot_clearance').value))
        self.path = []
        self.plan_needed = self.goal is not None
        self.route_blocked = False
        self.replan_attempted = False
        self.replan_pose = None
        self.waiting_for_clear_pose = False
        self.clear_pose_count = 0
        self.stop_robot()
        self.get_logger().info(
            f'Map received: {msg.info.width} x {msg.info.height}')

    def pose_callback(self, _msg):
        self.pose_seen = True

    def goal_callback(self, msg):
        if msg.header.frame_id != 'map':
            self.get_logger().error('Goal frame must be map')
            return
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.path = []
        self.path_index = 0
        self.plan_needed = True
        self.finished = False
        self.route_blocked = False
        self.replan_attempted = False
        self.replan_pose = None
        self.waiting_for_clear_pose = False
        self.clear_pose_count = 0
        self.stop_robot()
        self.get_logger().info(f'New goal: {self.goal}')

    def current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time())
        except TransformException:
            return None
        p, q = transform.transform.translation, transform.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        return p.x, p.y, yaw

    def plan(self, pose):
        self.plan_needed = False
        self.path = []
        start = world_to_cell(pose[0], pose[1], *self.map_info)
        end = world_to_cell(self.goal[0], self.goal[1], *self.map_info)
        cells = astar(self.blocked, start, end)
        if cells is None:
            self.get_logger().error(
                'No route: goal/start is blocked or unreachable')
            self.publish_path()
            self.route_blocked = True
            return
        self.path = [cell_to_world(row, col, *self.map_info)
                     for row, col in cells]
        self.path.append(self.goal)
        self.path_index = 0
        self.route_blocked = False
        self.publish_path()
        self.get_logger().info(f'A* route: {len(cells)} grid cells')

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            point = PoseStamped()
            point.header = msg.header
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.orientation.w = 1.0
            msg.poses.append(point)
        self.path_pub.publish(msg)

    def line_is_clear(self, start, end):
        # A coarse sample can skip a blocked cell touched near its corner.
        step = self.map_info[3] / 10
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, math.ceil(distance / step))
        for i in range(samples + 1):
            ratio = i / samples
            row, col = world_to_cell(
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]), *self.map_info)
            if not (0 <= row < self.blocked.shape[0] and
                    0 <= col < self.blocked.shape[1]):
                return False
            if self.blocked[row, col]:
                return False
        return True

    def control_loop(self):
        if self.goal is None or self.map_data is None or not self.pose_seen:
            return
        pose = self.current_pose()
        if pose is None:
            self.stop_robot()
            return
        if self.plan_needed:
            self.stop_robot()
            self.plan(pose)
        if self.waiting_for_clear_pose:
            row, col = world_to_cell(pose[0], pose[1], *self.map_info)
            if (0 <= row < self.blocked.shape[0] and
                    0 <= col < self.blocked.shape[1] and
                    not self.blocked[row, col]):
                self.clear_pose_count += 1
            else:
                self.clear_pose_count = 0
            if self.clear_pose_count >= 3:
                self.waiting_for_clear_pose = False
                self.replan_attempted = False
                self.plan(pose)
            else:
                return
        if not self.path or self.finished or self.route_blocked:
            return

        x, y, yaw = pose
        if (self.replan_pose is not None and
                math.hypot(x - self.replan_pose[0],
                           y - self.replan_pose[1]) > 0.2):
            self.replan_attempted = False
            self.replan_pose = None
        goal_distance = math.hypot(self.goal[0] - x, self.goal[1] - y)
        if goal_distance < self.get_parameter('goal_tolerance').value:
            self.stop_robot()
            self.finished = True
            self.get_logger().info('Goal reached')
            return

        # Advance along the route, then choose a visible point ahead.
        end_index = min(len(self.path), self.path_index + 25)
        self.path_index = min(
            range(self.path_index, end_index),
            key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))
        # First return to the center of the current route cell. Jumping
        # straight to the next cell can cut a corner from an offset pose.
        target_index = self.path_index
        if (target_index < len(self.path) - 1 and
                math.hypot(self.path[target_index][0] - x,
                           self.path[target_index][1] - y) <
                self.map_info[3] / 2):
            target_index += 1
        lookahead = max(0.05, self.get_parameter('lookahead_distance').value)
        for i in range(target_index, len(self.path)):
            if not self.line_is_clear((x, y), self.path[i]):
                break
            target_index = i
            distance = math.hypot(
                self.path[i][0] - x, self.path[i][1] - y)
            if distance >= lookahead:
                break
        target = self.path[target_index]
        if not self.line_is_clear((x, y), target):
            self.handle_blocked_route(pose, 'target line')
            return

        dx, dy = target[0] - x, target[1] - y
        target_distance = math.hypot(dx, dy)
        alpha = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                           math.cos(math.atan2(dy, dx) - yaw))
        cmd = Twist()
        if abs(alpha) > 0.45:
            cmd.angular.z = max(-0.45, min(0.45, 2.0 * alpha))
        else:
            speed = max(0.0, self.get_parameter('linear_velocity').value)
            cmd.linear.x = min(speed * max(0.35, math.cos(alpha)),
                               max(0.03, 0.5 * goal_distance))
            curvature = 2 * math.sin(alpha) / max(target_distance, 0.05)
            cmd.angular.z = max(-0.8, min(0.8, cmd.linear.x * curvature))
            # The straight sight line can be free while the turning arc
            # clips a corner. Check the actual command before publishing it.
            # Commands are refreshed every 0.1 s. Check about one second of
            # travel, stopping at the current waypoint; extending beyond it
            # wrongly rejects a route that turns at that waypoint.
            horizon = min(target_distance, max(0.08, cmd.linear.x))
            if not trajectory_is_clear(
                    self.blocked, self.map_info, pose, cmd.linear.x,
                    cmd.angular.z, horizon):
                if abs(alpha) > 0.003:
                    cmd = Twist()
                    cmd.angular.z = max(-0.45, min(0.45, 2.0 * alpha))
                else:
                    self.handle_blocked_route(pose, 'command arc')
                    return
        self.publish_drive_command(cmd)

    def publish_drive_command(self, cmd):
        """Publish a drive command; subclasses may add safety checks."""
        self.cmd_pub.publish(cmd)

    def handle_blocked_route(self, pose, reason):
        """Stop, then replan once if the robot is still in free space."""
        self.stop_robot()
        row, col = world_to_cell(pose[0], pose[1], *self.map_info)
        in_free_space = (0 <= row < self.blocked.shape[0] and
                         0 <= col < self.blocked.shape[1] and
                         not self.blocked[row, col])
        if in_free_space and not self.replan_attempted:
            self.replan_attempted = True
            self.replan_pose = pose[:2]
            self.route_blocked = True
            self.get_logger().warn(
                f'{reason} blocked at ({pose[0]:.2f}, {pose[1]:.2f}); '
                'replanning')
            self.plan(pose)
        elif not in_free_space and not self.route_blocked:
            self.route_blocked = True
            self.waiting_for_clear_pose = True
            self.clear_pose_count = 0
            self.get_logger().error(
                f'Estimated pose ({pose[0]:.2f}, {pose[1]:.2f}) is inside '
                'the inflated obstacle area; waiting for a stable free pose')
        elif not self.route_blocked:
            self.route_blocked = True
            self.get_logger().error(
                f'{reason} remains blocked after replanning; robot stopped')

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    """Run the A* and Pure Pursuit ROS node."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AStarPursuit()
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
