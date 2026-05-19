import numpy as np
import os
import click
import pickle
import yaml
from typing import Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset

from flownav.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)


class ViNT_Dataset(Dataset):
    """Dataset for ViNT-style goal-conditioned trajectory learning.

    This dataset produces one training sample containing:
    - Observation context images (stacked along channel dimension)
    - Goal image
    - Future action trajectory in local frame
    - Distance category and action supervision mask

    The implementation also supports negative goal mining and LMDB image caching
    to speed up repeated training runs.
    """

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
    ):
        # Root directory that contains trajectories.
        self.data_folder = data_folder
        # Split directory that contains traj_names.txt and cache files.
        self.data_split_folder = data_split_folder
        # Dataset key used to lookup normalization/stat settings in data_config.yaml.
        self.dataset_name = dataset_name

        # Read trajectory names used by this split.
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        # Remove trailing empty line if present.
        if "" in self.traj_names:
            self.traj_names.remove("")

        # Per-sample image size for loading and transforms.
        self.image_size = image_size
        # Step interval used for sampling waypoints in time.
        self.waypoint_spacing = waypoint_spacing

        # Distance categories are quantized by waypoint spacing.
        self.distance_categories = list(        #离散化dist的预测目标，将回归问题转化为分类问题，更容易收敛
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]

        # Whether to include negative goals sampled from other trajectories.
        self.negative_mining = negative_mining
        if self.negative_mining:
            # Reserve a special category for negatives.
            self.distance_categories.append(-1)     #类似于一个错误的、不相关的目标来作为对比学习信号

        # Number of future waypoints predicted by the model.
        self.len_traj_pred = len_traj_pred
        # Whether action labels contain orientation information.
        self.learn_angle = learn_angle
        # Action loss valid range in distance categories.
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        # Number of historical frames used as observation context.
        self.context_size = context_size

        # Support only known context sampling strategies.
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type

        # Number of frames to skip at trajectory end when constructing index.
        self.end_slack = end_slack
        # Number of goals potentially sampled per observation (used by configs/indexing logic).
        self.goals_per_obs = goals_per_obs
        # Whether to normalize action and goal position by metric waypoint spacing.
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # Load dataset-specific config (e.g., metric_waypoint_spacing).
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config, (      #一种数据集用一个类实例
            f"Dataset {self.dataset_name} not found in data_config.yaml"
        )
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()

        # Stable dataset index used by multi-dataset training.
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        # In-memory trajectory metadata cache to reduce repeated file IO.
        self.trajectory_cache = {}
        # Load or build sample index and goal index.  先建立索引 再建立缓存
        self._load_index()
        # Build/open LMDB cache for image reads.
        self._build_caches()

        # Action label dimensionality depends on whether yaw is learned.
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        # Required for dataloader workers pickling.
        # LMDB environment cannot be safely pickled, so drop it.
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        # Restore state and reopen LMDB in each worker process.
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        # Cache file is split- and dataset-specific.
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )
        # Prime trajectory metadata cache.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        # Build LMDB only once. Subsequent runs reuse it.
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}",
                colour="cyan",
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        # Image key is its resolved path string.
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Open LMDB in read-only mode for fast concurrent reads.
        self._image_cache: lmdb.Environment = lmdb.open(
            cache_filename, readonly=True, max_readers=256
        )

    def _build_index(self, use_tqdm: bool = True):
        # (traj_name, curr_time, max_goal_distance)
        samples_index = []
        # (traj_name, goal_time), used for negative sampling.
        goals_index = []

        for traj_name in tqdm.tqdm(
            self.traj_names, disable=not use_tqdm, dynamic_ncols=True, colour="yellow"
        ):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            # Every frame can be a potential goal frame.
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            # Need enough history for context.
            begin_time = self.context_size * self.waypoint_spacing
            # Need enough future for predicted trajectory horizon.
            end_time = (
                traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            )
            for curr_time in range(begin_time, end_time):
                # Goal distance cannot exceed configured max or remaining traj length.
                max_goal_distance = min(
                    self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1
                )
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        # Uniformly sample offset in [0, max_goal_dist].
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            # Offset 0 is mapped to negative goal mining.
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            # Positive goal sampled from the same trajectory in the future.
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        # Sample random (trajectory, time) from global goal pool.
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        # Cache file name includes major index-affecting params.
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # Load prebuilt index if available.
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except Exception:
            # Build and persist index on first run.
            click.echo(
                click.style(f"!! Creating index for {self.dataset_name}", fg="yellow")
            )
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        # Convert logical index to absolute image path key.
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            # Wrap bytes into file-like object for common image loader.
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            # Usually means cache miss for this key.
            click.echo(click.style(f"Failed to load image {image_path}", fg="red"))

    def _compute_actions(self, traj_data, curr_time, goal_time):
        # Build future window [curr_time, curr_time + horizon] with stride waypoint_spacing.
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index : end_index : self.waypoint_spacing]
        positions = traj_data["position"][
            start_index : end_index : self.waypoint_spacing
        ]
        # Clamp goal index for safety.
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        # Normalize yaw shape to 1D if data is stored as Nx1.
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        # If near the end, pad with the last valid value.
        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0
            )

        assert yaw.shape == (self.len_traj_pred + 1,), (
            f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        )
        assert positions.shape == (self.len_traj_pred + 1, 2), (
            f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        )

        # Convert all waypoints and goal to local frame centered at current pose.
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), (
            f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        )

        if self.learn_angle:
            # Relative yaw against current orientation.
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            # Position-only actions.
            actions = waypoints[1:]

        if self.normalize:
            # Normalize by metric spacing and temporal spacing to keep scales stable.
            actions[:, :2] /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )
            goal_pos /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )

        assert actions.shape == (self.len_traj_pred, self.num_action_params), (
            f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        )

        return actions, goal_pos

    def _get_trajectory(self, trajectory_name):
        # Hot path: return cached metadata if available.
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            # Load traj_data.pkl once and cache it.
            with open(
                os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb"
            ) as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        # Number of valid current-time anchors in this split.
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        # Get current anchor (trajectory, time, max sampled goal distance).
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]

        # Sample goal frame and whether it is negative.
        f_goal, goal_time, goal_is_negative = self._sample_goal(
            f_curr, curr_time, max_goal_dist
        )

        # Build observation context frames.
        context = []
        if self.context_type == "temporal":
            # Temporal context uses fixed stride backward from current time.
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        # Stack context images along channel dimension.
        obs_image = torch.cat([self._load_image(f, t) for f, t in context])

        # Goal image is a single frame.
        goal_image = self._load_image(f_goal, goal_time)

        # Load trajectory metadata for assertions and supervision.
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute local-frame action trajectory and goal position.
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Distance label: negatives mapped to max category, positives by frame gap.
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, (
                f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
            )

        # Convert numpy actions to float tensor.
        actions_torch = torch.as_tensor(
            actions.astype(dtype=np.float32), dtype=torch.float32
        )
        if self.learn_angle:
            # Convert angle channel to sin/cos representation.
            actions_torch = calculate_sin_cos(actions_torch)

        # Action loss is only applied to valid positive distances.
        action_mask = (
            (distance < self.max_action_distance)
            and (distance > self.min_action_distance)
            and (not goal_is_negative)
        )

        # Return format consumed by training loop.
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos.astype(dtype=np.float32), dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
