import rospy
import numpy as np
import copy

class ROSData:
    def __init__(self, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.timout = timeout
        self.last_time_received = float("-inf")
        self.queue_size = queue_size
        self.data = None
        self.lastdata = None
        self.name = name
        self.phantom = False
        self.last_exec_index = -1
        self.current_waypoint_index = 0
        self.Nfree=3
        
    
    def get(self):
        return self.data

    def pop_head(self):
        if self.queue_size == 1:
            data = self.data
            self.data = None
            self.current_waypoint_index = 0
            return data
        if self.data is None or len(self.data) == 0:
            return None
        return self.data.pop(0)

    def _to_waypoint_list(self, data):
        """Convert flattened waypoint array/list to [[x,y], ...] or [[x,y,hx,hy], ...]."""
        arr = np.asarray(data, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []
        if arr.size % 2 == 0:
            return arr.reshape(-1, 2).tolist()
        # fallback: keep as a single waypoint-like entry
        return [arr.tolist()]

    def _smooth_new_waypoints(self, new_points):
        """
        Smooth transition between old and new blocks.
        Smooth target:
        - first Nfree points of new_points
        - lastdata[current_waypoint_index-Nfree : current_waypoint_index] window
        """
        if not new_points or self.lastdata is None:
            return new_points

        prev = np.asarray(self.lastdata, dtype=np.float32)
        if prev.ndim != 2 or prev.shape[0] == 0:
            return new_points

        # window in previous data: [current_idx-Nfree, current_idx] (inclusive on idx)
        cur_idx = int(self.current_waypoint_index)
        start = max(0, cur_idx - int(self.Nfree))
        end = min(prev.shape[0], cur_idx + 1)
        hist = prev[start:end]
        if hist.shape[0] == 0:
            return new_points

        new_arr = np.asarray(new_points, dtype=np.float32)
        if new_arr.ndim != 2 or new_arr.shape[0] == 0:
            return new_points

        # blend first k points of new with tail k points of history
        k = min(int(self.Nfree), new_arr.shape[0], hist.shape[0])
        if k <= 0:
            return new_points

        old_tail = hist[-k:]
        # 使用带边界条件的三次多项式最小二乘拟合：
        # 约束 p(0) = last_old_point, p'(0) = v0 （从历史末尾估计），最小化 sum||p(t_i)-new_i||^2
        # t_i 取 1..k
        t = np.arange(1, k + 1, dtype=np.float32)
        T2 = (t ** 2).reshape(-1, 1)
        T3 = (t ** 3).reshape(-1, 1)
        A = np.hstack([T2, T3])  # 用于求解 c,d

        # a = p(0) = old_tail[-1]
        a = old_tail[-1]
        # 估计 p'(0)=b，使用历史末尾两点差分（若可用），否则用 new 的第一个差分近似
        if hist.shape[0] >= 2:
            b = old_tail[-1] - hist[-2]
        else:
            # fallback: estimate from new_arr first two points if possible
            if new_arr.shape[0] >= 2:
                b = new_arr[1] - new_arr[0]
            else:
                b = np.zeros_like(a)

        # 为 x, y（或维度）分别求解 c,d
        rhs = new_arr[:k] - (a + (b * t.reshape(-1, 1)))
        try:
            # least squares 求解 A @ [c; d] = rhs  （对每维独立）
            params, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            # params 形状 (2, dim)
            c = params[0]
            d = params[1]
            # 生成平滑后的前 k 点
            fitted = (a + b * t.reshape(-1, 1)) + (c * (t ** 2).reshape(-1, 1)) + (d * (t ** 3).reshape(-1, 1))
            new_arr[:k] = fitted
        except Exception:
            # 若最小二乘失败，回退到线性融合
            for i in range(k):
                alpha = float(i + 1) / float(k + 1)
                new_arr[i] = (1.0 - alpha) * old_tail[i] + alpha * new_arr[i]

        return new_arr.tolist()
    
    def set(self, data):        #self.queue_size=8!!!
        time_waited = rospy.get_time() - self.last_time_received
        self.lastdata = copy.deepcopy(self.data)
        if self.queue_size == 1:
            self.data = data
            # reset index when a new whole-trajectory message arrives
            self.current_waypoint_index = 0
        else:
            if self.data is None or time_waited > self.timout: # reset queue if timeout
                self.data = []
            # 扁平化数据 -> waypoint 点列表（如 8*xy）
            points = self._to_waypoint_list(data)
            # 对新传入前 Nfree 点做平滑，参考 lastdata 的 [idx-Nfree, idx] 段
            points = self._smooth_new_waypoints(points)

            for pt in points:
                if len(self.data) == self.queue_size:
                    self.data.pop(0)
                self.data.append(pt)

        self.last_time_received = rospy.get_time()
        self.last_exec_index = self.current_waypoint_index 
        
        
    def is_valid(self, verbose: bool = False):
        time_waited = rospy.get_time() - self.last_time_received
        valid =  time_waited < self.timout
        if self.queue_size > 1: #self.queue_size=8!!!
            valid = valid and self.data is not None and len(self.data) > 0 and self.current_waypoint_index < len(self.data)
        else:   #self.queue_size=8!!!
            valid = valid and (self.data is not None)
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {time_waited} seconds (timeout: {self.timout} seconds)")
        return valid
