"""
Real-time visualization of annotated observation and target goal side-by-side.
Saves frame pairs to disk and displays via OpenCV.
"""

import os
import numpy as np
import cv2
import torch
from pathlib import Path
from threading import Thread, Lock
import time
from queue import Queue, Empty

class GoalVisualizer:
    """Visualizes observation + goal images in real-time."""
    
    def __init__(self, output_dir="vis_frames", display=True, save=False):
        self.output_dir = output_dir
        self.display = display
        self.save = save
        self.frame_queue = Queue(maxsize=2)
        self.lock = Lock()
        self.running = True
        self.frame_idx = 0
        
        if self.save:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Start display thread
        if self.display:
            self.display_thread = Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
    
    def update(self, obs_img: np.ndarray, goal_img: torch.Tensor):
        """
        Update visualization with observation and goal images.
        
        Args:
            obs_img: Observation image (H, W, 3) uint8 in RGB
            goal_img: Goal image tensor (3, H, W) in range [0,1], normalized
        """
        # Convert goal tensor to numpy
        if isinstance(goal_img, torch.Tensor):
            goal_np = (goal_img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        else:
            goal_np = goal_img
        
        # Ensure same height
        h_obs, w_obs = obs_img.shape[:2]
        h_goal, w_goal = goal_np.shape[:2]
        
        if h_obs != h_goal:
            goal_np = cv2.resize(goal_np, (w_goal, h_obs))
        
        # Horizontal stack
        combined = np.hstack([obs_img, goal_np])
        
        # Add label text
        cv2.putText(combined, "Observation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Goal", (w_obs + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Put to queue (drop if full)
        try:
            self.frame_queue.put_nowait((combined, obs_img, goal_np))
        except:
            pass  # Queue full, skip
        
        # Save if enabled
        if self.save:
            save_path = os.path.join(self.output_dir, f"frame_{self.frame_idx:06d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            self.frame_idx += 1
    
    def _display_loop(self):
        """Background thread for displaying frames."""
        while self.running:
            try:
                combined, _, _ = self.frame_queue.get(timeout=0.1)
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.imshow("Observation (LEFT) vs Goal (RIGHT)", bgr)
                cv2.waitKey(1)
            except Empty:
                pass
    
    def close(self):
        """Clean up."""
        self.running = False
        cv2.destroyAllWindows()


# Example usage in navigate_ros1.py:
# 
# Add to imports:
#   from visualize_goals import GoalVisualizer
# 
# Add in main() after line 122 (after Scorer init):
#   visualizer = GoalVisualizer(display=True, save=False)
# 
# Add in main loop after line 192 (after annotated_image_msg = msg_from_numpy(...)):
#   visualizer.update(annotated_np, goal_images[sg_idx])
# 
# Add before rospy.spin() or at end of main():
#   visualizer.close()

if __name__ == "__main__":
    # Demo usage
    print("GoalVisualizer ready. Import in navigate_ros1.py and call:")
    print("  visualizer = GoalVisualizer(display=True, save=False)")
    print("  visualizer.update(obs_img, goal_img_tensor)")
