#!/usr/bin/env python3
import math
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geographic_msgs.msg import GeoPoseStamped
from cv_bridge import CvBridge, CvBridgeError

class CardinalDirectionNode(Node):
    def __init__(self):
        super().__init__('cardinal_direction_node')

        # ROS2 Subscriptions
        self.create_subscription(GeoPoseStamped, '/geo_pose', self.geo_pose_callback, 10)
        self.create_subscription(Image, '/left/image_rect_color', self.image_callback, 10)

        # ROS2 Publisher
        self.image_pub = self.create_publisher(Image, '/cardinal_direction_image', 10)

        # Internal Variables
        self.bridge = CvBridge()
        self.latest_yaw = None
        self.latest_image = None
        self.published = False  # Ensures publishing happens only once

        self.get_logger().info("Cardinal Direction Node Initialized.")

    def geo_pose_callback(self, msg: GeoPoseStamped):
        if self.published:
            return

        q = msg.pose.orientation
        self.latest_yaw = self.quaternion_to_yaw(q)
        self.get_logger().info(f"Received GeoPose. Yaw: {math.degrees(self.latest_yaw):.2f}")
        self.try_process_and_publish()

    def image_callback(self, msg: Image):
        if self.published:
            return

        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.get_logger().info("Received image from camera.")
            self.try_process_and_publish()
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def try_process_and_publish(self):
        if self.published:
            return

        if self.latest_image is None or self.latest_yaw is None:
            self.get_logger().warn("Waiting for both image and yaw data...")
            return

        img = self.latest_image.copy()
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        red = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10

        # --- Draw the top yaw text ---
        current_heading = math.degrees(self.latest_yaw) % 360
        yaw_text = f"Yaw: {current_heading:.0f}"
        fs_top, thickness_top = 1.0, 2
        (yaw_w, yaw_h), _ = cv2.getTextSize(yaw_text, font, fs_top, thickness_top)
        yaw_x = (w - yaw_w) // 2
        yaw_y = yaw_h + margin  # baseline for yaw text
        cv2.putText(img, yaw_text, (yaw_x, yaw_y), font, fs_top, red, thickness_top)

        # Reserve a top region that includes the yaw text
        extra_margin = 10
        reserved_top = yaw_y + extra_margin

        # --- Draw dynamic cardinal markers ---
        fs_marker, thickness_marker = 1.2, 2
        cardinals = [
            ("N", 0), ("E", 90), ("S", 180), ("W", 270),
            ("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)
        ]

        for label, abs_bearing in cardinals:
            rel_angle = self._normalize_angle(abs_bearing - current_heading)
            rad = math.radians(rel_angle)
            dx, dy = math.sin(rad), -math.cos(rad)
            ix, iy = self._compute_intersection(center[0], center[1], dx, dy, w, h)
            (tw, th), _ = cv2.getTextSize(label, font, fs_marker, thickness_marker)
            pos_x = max(margin, min(ix - tw // 2, w - margin - tw))
            pos_y = iy + th // 2

            if pos_y - th < reserved_top:
                pos_y = reserved_top + th
            if pos_y > h - margin:
                pos_y = h - margin
            cv2.putText(img, label, (pos_x, pos_y), font, fs_marker, red, thickness_marker)

        cv2.destroyAllWindows()

        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.image_pub.publish(ros_img)
            self.get_logger().info("Published dynamic compass overlay image.")
            self.published = True
            self.get_logger().info("Shutting down node after publishing.")
            rclpy.shutdown()
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error when publishing: {e}")

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        while angle_deg <= -180:
            angle_deg += 360
        while angle_deg > 180:
            angle_deg -= 360
        return angle_deg

    @staticmethod
    def _compute_intersection(cx: int, cy: int, dx: float, dy: float, w: int, h: int) -> (int, int):
        t_candidates = []
        if dx > 0:
            t_candidates.append((w - cx) / dx)
        elif dx < 0:
            t_candidates.append((0 - cx) / dx)
        if dy > 0:
            t_candidates.append((h - cy) / dy)
        elif dy < 0:
            t_candidates.append((0 - cy) / dy)
        t = min(val for val in t_candidates if val > 0) if t_candidates else 1
        return int(cx + dx * t), int(cy + dy * t)

    @staticmethod
    def quaternion_to_yaw(q) -> float:
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = CardinalDirectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

