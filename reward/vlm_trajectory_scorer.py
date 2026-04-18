import os
import re
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv

from reward.prompt import TRAJECTORY_SCORE_PROMPT

# Load .env from reward/ directory
load_dotenv(Path(__file__).parent / ".env")

# 5 distinct colors for trajectory rendering (RGB tuples)
TRAJ_COLORS = [
    (255, 0, 0),      # red
    (0, 200, 0),      # green
    (0, 100, 255),    # blue
    (255, 165, 0),    # orange
    (200, 0, 255),    # purple
]


class VLMTrajectoryScorer:
    """Score candidate navigation trajectories using Doubao VLM.

    Renders trajectories onto the observation image, sends to Doubao via
    Ark SDK, and returns per-trajectory scores for GRPO.
    """

    def __init__(
        self,
        base_url=None,
        model=None,
        api_key=None,
        num_trajectories=5,
    ):
        self.client = Ark(
            base_url=base_url or os.getenv("ARK_BASE_URL"),
            api_key=api_key or os.getenv("ARK_API_KEY"),
        )
        self.model = model or os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215")
        self.num_trajectories = num_trajectories

    def render_trajectories(self, obs_image, trajectories):
        """Draw numbered, colored trajectories onto the observation image.

        Args:
            obs_image: PIL Image or numpy array (H, W, 3) uint8.
            trajectories: list of N arrays, each shape (T, 2) in pixel coords.

        Returns:
            PIL Image with trajectories drawn on it.
        """
        if isinstance(obs_image, np.ndarray):
            obs_image = Image.fromarray(obs_image)
        img = obs_image.copy().convert("RGB")

        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for idx, traj in enumerate(trajectories):
            color = TRAJ_COLORS[idx % len(TRAJ_COLORS)]
            points = [(float(x), float(y)) for x, y in traj]

            # Draw trajectory line
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)

            # Draw waypoint dots
            for pt in points:
                r = 3
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)

            # Draw trajectory number label at the endpoint
            if len(points) > 0:
                end = points[-1]
                label = str(idx + 1)
                draw.text(
                    (end[0] + 5, end[1] - 7),
                    label,
                    fill=color,
                    font=font,
                )

        return img

    @staticmethod
    def _image_to_base64(image):
        """Convert PIL Image to base64 data URI."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _build_input(self, image_base64, task_description=""):
        """Build Doubao Ark responses API input."""
        prompt_text = TRAJECTORY_SCORE_PROMPT.format(
            num_trajectories=self.num_trajectories,
            task_description=task_description if task_description else "Navigate safely and efficiently.",
        )
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_base64,
                    },
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    },
                ],
            }
        ]

    def score(self, obs_image, trajectories, task_description=""):
        """Score candidate trajectories using the VLM.

        Args:
            obs_image: PIL Image or numpy array (H, W, 3).
            trajectories: list of N arrays, each shape (T, 2) in pixel coords.
            task_description: optional task context string.

        Returns:
            dict with "scores" (list of floats) and "raw_output" (str).
        """
        annotated = self.render_trajectories(obs_image, trajectories)
        image_b64 = self._image_to_base64(annotated)
        input_messages = self._build_input(image_b64, task_description)

        response = self.client.responses.create(
            model=self.model,
            input=input_messages,
        )
        # Extract text from the response output messages
        raw_output = ""
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        raw_output += block.text

        scores = self._parse_scores(raw_output, len(trajectories))
        return {
            "scores": scores,
            "raw_output": raw_output,
        }

    def _parse_scores(self, text, num_trajs):
        """Parse <Scores>[s1, s2, ...]</Scores> from VLM output."""
        m = re.search(r"<Scores>\s*\[([^\]]+)\]\s*</Scores>", text)
        if m:
            parts = m.group(1).split(",")
            try:
                scores = [float(s.strip()) for s in parts[:num_trajs]]
                if len(scores) == num_trajs:
                    return scores
            except ValueError:
                pass
        # Fallback: equal scores
        return [5.0] * num_trajs

    @staticmethod
    def scores_to_rewards(scores):
        """Normalize scores within the group for GRPO advantages.

        Returns (score - mean) / (std + eps) so that the group mean is ~0.
        """
        arr = np.array(scores, dtype=np.float32)
        mean = arr.mean()
        std = arr.std()
        eps = 1e-6
        rewards = (arr - mean) / (std + eps)
        return rewards.tolist()

if __name__ == "__main__":
    # Example usage
    scorer = VLMTrajectoryScorer(num_trajectories=3)
    obs_img = np.zeros((480, 640, 3), dtype=np.uint8)  # dummy black image
    trajs = [
        np.array([[100, 400], [150, 350], [200, 300]]),
        np.array([[100, 400], [120, 380], [140, 360]]),
        np.array([[100, 400], [80, 420], [60, 440]]),
    ]
    result = scorer.score(obs_img, trajs)
    print("Scores:", result["scores"])
    print("Raw output:", result["raw_output"])