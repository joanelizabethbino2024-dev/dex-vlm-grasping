import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
import tf2_ros
from ultralytics import YOLO


class YoloGripperTargetNode(Node):

  def __init__(self):
    super().__init__('yolo_gripper_target_node')
    self.bridge = CvBridge()

    model_path = '/home/tt501/runs/detect/train-11/weights/best.pt'
    self.get_logger().info(f'Loading YOLO model from: {model_path}')
    self.model = YOLO(model_path)

    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    self.create_subscription(
        Image,
        '/camera/image_raw',
        self.rgb_callback,
        qos_profile_sensor_data,
    )
    self.create_subscription(
        Image,
        '/camera/depth/image_raw',
        self.depth_callback,
        qos_profile_sensor_data,
    )
    self.create_subscription(
        CameraInfo,
        '/camera/camera_info',
        self.info_callback,
        qos_profile_sensor_data,
    )

    self.target_pub = self.create_publisher(
        PoseStamped, '/gripper_target_pose', 10
    )

    self.depth_image = None
    self.fx = self.fy = self.cx = self.cy = None
    self.last_log_time = self.get_clock().now()

    self.get_logger().info('YOLO Node Active! Waiting for streams...')

  def info_callback(self, msg):
    if self.fx is None:
      self.fx = msg.k[0]
      self.cx = msg.k[2]
      self.fy = msg.k[4]
      self.cy = msg.k[5]
      self.get_logger().info('Received Camera Intrinsic Info!')

  def depth_callback(self, msg):
    self.depth_image = self.bridge.imgmsg_to_cv2(
        msg, desired_encoding='passthrough'
    )

  def rgb_callback(self, msg):
    if self.fx is None or self.depth_image is None:
      return

    bgr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Extremely low confidence threshold (0.05) to catch everything
    results = self.model(bgr_frame,imgsz=640, conf=0.01, verbose=False)[0]

    # Render YOLO detections directly on the frame window
    annotated_frame = results.plot()
    cv2.imshow('YOLO Live Detection', annotated_frame)
    cv2.waitKey(1)

    detected_items = []
    frame_name = msg.header.frame_id

    for box in results.boxes:
      cls_id = int(box.cls[0])
      class_name = self.model.names[cls_id]

      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
      u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

      h, w = self.depth_image.shape
      v_min, v_max = max(0, v - 2), min(h, v + 3)
      u_min, u_max = max(0, u - 2), min(w, u + 3)
      depth_patch = self.depth_image[v_min:v_max, u_min:u_max]

      z_depth = float(np.nanmedian(depth_patch))
      if np.isnan(z_depth) or z_depth <= 0.0:
        continue

      x_cam = (u - self.cx) * z_depth / self.fx
      y_cam = (v - self.cy) * z_depth / self.fy

      pose_msg = PoseStamped()
      pose_msg.header.stamp = self.get_clock().now().to_msg()
      pose_msg.header.frame_id = msg.header.frame_id
      pose_msg.pose.position.x = float(x_cam)
      pose_msg.pose.position.y = float(y_cam)
      pose_msg.pose.position.z = float(z_depth)
      pose_msg.pose.orientation.w = 1.0

      try:
        transform = self.tf_buffer.lookup_transform(
            'base_link', msg.header.frame_id, rclpy.time.Time()
        )
        pose_msg = tf2_geometry_msgs.do_transform_pose_stamped(
            pose_msg, transform
        )
        frame_name = 'base_link'
      except Exception:
        pass

      self.target_pub.publish(pose_msg)
      pos = pose_msg.pose.position
      detected_items.append(
          f'{class_name}: [{pos.x:.2f}m, {pos.y:.2f}m, {pos.z:.2f}m]'
      )

    # Print summary every 1 second
    now = self.get_clock().now()
    if (now - self.last_log_time).nanoseconds / 1e9 >= 1.0:
      if detected_items:
        summary = ' | '.join(detected_items)
        self.get_logger().info(f'({frame_name}) -> {summary}')
      else:
        self.get_logger().info(
            f'YOLO active, but detected 0 objects (Raw Boxes:'
            f' {len(results.boxes)})'
        )
      self.last_log_time = now


def main(args=None):
  rclpy.init(args=args)
  node = YoloGripperTargetNode()
  try:
    rclpy.spin(node)
  except (KeyboardInterrupt, SystemExit):
    pass
  finally:
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()
