#!/usr/bin/env python3
import threading
import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from navigate_ros1 import trajs_msg

# 复用项目里的转换函数（避免 cv_bridge 兼容问题）
from utils import msg_to_pil

IMAGE_TOPIC = "/camera/color/image_raw"     #sub
WAYPOINT_TOPIC = "/waypoint"                #sub            
TRAJS_TOPIC = "/candidate_trajs"            #sub
OVERLAY_TOPIC = "/overlay_image"            #pub


class NavVisualizerNode:
    def __init__(self):
        # ===== 参数 =====
        self.image_topic = rospy.get_param("~image_topic", IMAGE_TOPIC)
        self.waypoint_topic = rospy.get_param("~waypoint_topic", WAYPOINT_TOPIC)
        self.candidate_trajs_topic = rospy.get_param("~candidate_trajs_topic", TRAJS_TOPIC)
        self.overlay_topic = rospy.get_param("~overlay_topic", OVERLAY_TOPIC)

        self.render_hz = float(rospy.get_param("~render_hz", 15.0))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.window_name = rospy.get_param("~window_name", "nav_overlay")

        # 轨迹向量维度（2: x,y；4: x,y,sin,cos）
        self.traj_dim = int(rospy.get_param("~traj_dim", 2))

        # 简化投影参数（机器人局部坐标 -> 图像像素）
        # x: 前进方向(米)；y: 左右方向(米)
        self.origin_u = float(rospy.get_param("~origin_u", 320.0))  # 图像底部中心附近
        self.origin_v = float(rospy.get_param("~origin_v", 440.0))
        self.scale = float(rospy.get_param("~meter_to_pixel", 120.0))  # 1m -> 120px

        # 可选：单应矩阵（长度9，优先于简化投影）
        H_list = rospy.get_param("~homography", [])
        self.H = None
        if isinstance(H_list, list) and len(H_list) == 9:
            self.H = np.array(H_list, dtype=np.float32).reshape(3, 3)
            rospy.loginfo("Using homography projection.")
        else:
            rospy.loginfo("Using simple linear projection.")

        # ===== 状态 =====
        self.lock = threading.Lock()
        self.latest_bgr = None               # np.ndarray (H,W,3)
        self.latest_waypoint = None          # np.ndarray (>=2,)
        self.latest_traj = None              # np.ndarray (N,2)

        # ===== ROS IO =====
        self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.waypoint_topic, Float32MultiArray, self.cb_waypoint, queue_size=1)
        rospy.Subscriber(self.candidate_trajs_topic, Float32MultiArray, self.cb_pred_traj, queue_size=1)

        # 定时渲染
        period = 1.0 / max(self.render_hz, 1.0)
        self.timer = rospy.Timer(rospy.Duration(period), self.on_timer)

        rospy.loginfo("NavVisualizerNode started.")

    def cb_image(self, msg: Image):
        try:
            pil = msg_to_pil(msg)  # PIL RGB
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            with self.lock:
                self.latest_bgr = bgr
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"cb_image error: {e}")

    def cb_waypoint(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        if arr.size >= 2:
            with self.lock:
                self.latest_waypoint = arr

    def cb_pred_traj(self, msg: Float32MultiArray):
        arr = np.array(msg.data, dtype=np.float32)
        if arr.size < self.traj_dim:
            return
        n = arr.size // self.traj_dim
        arr = arr[: n * self.traj_dim].reshape(n, self.traj_dim)
        traj_xy = arr[:, :2]  # 只取 x,y
        with self.lock:
            self.latest_traj = traj_xy

    def project_xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        """
        xy: (N,2), 机器人局部坐标（米）
        return: uv: (N,2) 图像像素
        """
        if xy is None or len(xy) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        if self.H is not None:
            pts = xy.reshape(-1, 1, 2).astype(np.float32)
            uv = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
            return np.round(uv).astype(np.int32)

        # 简化投影（无标定时先用）
        x = xy[:, 0]
        y = xy[:, 1]
        u = self.origin_u + y * self.scale
        v = self.origin_v - x * self.scale
        uv = np.stack([u, v], axis=1)
        return np.round(uv).astype(np.int32)

    @staticmethod
    def cv2_to_imgmsg(bgr: np.ndarray, stamp=None, frame_id="camera"):
        msg = Image()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = bgr.tobytes()
        return msg

    def on_timer(self, _event):
        with self.lock:
            if self.latest_bgr is None:
                return
            frame = self.latest_bgr.copy()
            wp = None if self.latest_waypoint is None else self.latest_waypoint.copy()
            traj = None if self.latest_traj is None else self.latest_traj.copy()

        # 画预测轨迹（绿线）
        if traj is not None and len(traj) > 0:
            traj_uv = self.project_xy_to_uv(traj)
            h, w = frame.shape[:2]
            valid = (
                (traj_uv[:, 0] >= 0) & (traj_uv[:, 0] < w) &
                (traj_uv[:, 1] >= 0) & (traj_uv[:, 1] < h)
            )
            pts = traj_uv[valid]
            if len(pts) >= 2:
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, tuple(p), 2, (0, 200, 0), -1)

        # 画选中 waypoint（红点）
        if wp is not None and wp.size >= 2:
            wp_uv = self.project_xy_to_uv(wp[:2].reshape(1, 2))[0]
            cv2.circle(frame, tuple(wp_uv), 6, (0, 0, 255), -1)

        # HUD 文本
        cv2.putText(frame, "Green: predicted traj | Red: chosen waypoint",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 发布 overlay
        self.pub_overlay.publish(self.cv2_to_imgmsg(frame))

        # 可选窗口
        if self.show_window:
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)


def main():
    rospy.init_node("nav_visualizer", anonymous=False)
    _ = NavVisualizerNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()