#!/usr/bin/env python3
import threading
import numpy as np
import cv2
import rospy
import os,yaml
import torch

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from navigate_ros1 import trajs_msg


# 复用项目里的转换函数（避免 cv_bridge 兼容问题）
from utils import msg_to_pil

IMAGE_TOPIC = "/camera/color/image_raw"     #sub
WAYPOINT_TOPIC = "/waypoint"                #sub            
TRAJS_TOPIC = "/candidate_trajs"            #sub
OVERLAY_TOPIC = "/overlay_image"            #pub

#从flow_correct.py复用过来的投影类，因为这个分支没有reward所以不能import
class TrajectoryProjector:
    """Handles action space conversions and camera projection.

    Loads action stats and camera intrinsics once, then provides methods to
    convert between normalized deltas, cumulative actions, and pixel coords.
    """

    base_dir = os.path.dirname(__file__)
    default_action_config = os.path.join(base_dir, "../../../flownav/data/data_config.yaml")        #注意这里的路径跟flowcorrect中的不同
    default_camera_config = os.path.join(
        base_dir, "../../../deployment/config/camera.yaml"
    )

    def __init__(self, dataset_name="deploy", image_size=(640, 480),
                 action_config_path=default_action_config,
                 camera_config_path=default_camera_config,
                 camera_params=None):
        """
        Args:
            dataset_name: dataset key in camera config yaml.
            image_size: (width, height) for pixel clipping.
            action_config_path: path to action stats yaml.
            camera_config_path: path to camera metrics yaml.
            camera_params: dict to directly specify camera intrinsics, e.g.:
                {
                    "camera_height": 0.95,
                    "camera_x_offset": 0.45,
                    "fx": 272.5, "fy": 266.4, "cx": 320.0, "cy": 220.0,
                    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
                }
                When provided, camera_config_path and dataset_name are ignored
                for camera loading.
        """
        with open(action_config_path, "r") as f:
            action_config = yaml.safe_load(f)
        self.action_stats = {k: np.array(v) for k, v in action_config["action_stats"].items()}

        if camera_params is not None:
            self._init_camera_from_dict(camera_params)
        else:
            with open(camera_config_path, "r") as f:
                camera_config = yaml.safe_load(f)
            cam = camera_config[dataset_name]["camera_metrics"]
            self._init_camera_from_yaml(cam)
        self.image_size = image_size

    def _init_camera_from_yaml(self, cam):
        """Initialize from yaml camera_metrics nested dict."""
        cm = cam["camera_matrix"]
        dc = cam["dist_coeffs"]
        self._init_camera_from_dict({
            "camera_height": cam["camera_height"],
            "camera_x_offset": cam.get("camera_x_offset", 0.0),
            "fx": cm["fx"], "fy": cm["fy"], "cx": cm["cx"], "cy": cm["cy"],
            "k1": dc["k1"], "k2": dc["k2"], "p1": dc["p1"], "p2": dc["p2"], "k3": dc["k3"],
        })

    def _init_camera_from_dict(self, p):
        self.camera_height = p["camera_height"]
        self.camera_x_offset = p.get("camera_x_offset", 0.0)
        self.camera_matrix = np.array([
            [p["fx"], 0.0, p["cx"]],
            [0.0, p["fy"], p["cy"]],
            [0.0, 0.0, 1.0],
        ])
        # 畸变矫正参数，k1,k2,k3为径向畸变，p1,p2为切向畸变
        self.dist_coeffs = np.array([
            p.get("k1", 0.0), p.get("k2", 0.0),
            p.get("p1", 0.0), p.get("p2", 0.0),
            p.get("k3", 0.0), 0.0, 0.0, 0.0,
        ])

    def ndeltas_to_actions(self, ndeltas):
        """Normalized deltas (B, T, 2) tensor → cumulative actions (B, T, 2) tensor."""
        ndeltas_np = ndeltas.detach().cpu().numpy().reshape(ndeltas.shape[0], -1, 2)
        ndeltas_np = (ndeltas_np + 1) / 2 * (self.action_stats["max"] - self.action_stats["min"]) + self.action_stats["min"]
        actions = np.cumsum(ndeltas_np, axis=1)
        return torch.from_numpy(actions).float().to(ndeltas.device)

    def project_points(self, xy):
        """Local (x, y) waypoints (B, T, 2) np → pixel (u, v) (B, T, 2) np.

        Reused from visualnav-transformer/train/vint_train/visualizing/action_utils.py.
        """
        batch_size, horizon, _ = xy.shape
        xyz = np.concatenate(
            [xy, -self.camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
        )
        rvec = tvec = (0, 0, 0)
        xyz[..., 0] += self.camera_x_offset
        xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
        uv, _ = cv2.projectPoints(
            xyz_cv.reshape(batch_size * horizon, 3).astype(np.float64),
            rvec, tvec, self.camera_matrix, self.dist_coeffs,
        )
        uv = uv.reshape(batch_size, horizon, 2)
        return uv

    def actions_to_pixels(self, actions_np):
        """Cumulative actions (B, T, 2) np → clipped pixel coords (B, T, 2) np."""
        w, h = self.image_size
        uv = self.project_points(actions_np)
        uv[..., 0] = w - uv[..., 0]
        uv[..., 0] = np.clip(uv[..., 0], 0, w)
        uv[..., 1] = np.clip(uv[..., 1], 0, h)
        return uv




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

        # 投影类实例
        self.traj_projector = TrajectoryProjector()

        # ===== 状态 =====
        self.lock = threading.Lock()
        self.latest_bgr = None               # np.ndarray (H,W,3)
        self.latest_waypoint = None          # np.ndarray (>=2,)
        self.latest_traj = None              # np.ndarray (N,2)

        # ===== ROS IO =====
        self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.waypoint_topic, Float32MultiArray, self.cb_waypoint, queue_size=1)
        # rospy.Subscriber(self.candidate_trajs_topic, Float32MultiArray, self.cb_pred_traj, queue_size=1)
        rospy.Subscriber(self.candidate_trajs_topic, trajs_msg, self.cb_pred_traj, queue_size=1)

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
        trajs = arr.reshape(msg.layout.dim[0].size,     
                            msg.layout.dim[1].size,
                            msg.layout.dim[2].size)
        with self.lock:
            self.latest_traj = trajs

    def project_xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        """
        xy: (B,T,2), 机器人局部坐标（米）
        return: uv: (B,T,2) 图像像素
        """
        uv = self.traj_projector.project_points(xy)
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