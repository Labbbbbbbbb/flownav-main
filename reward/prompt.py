TRAJECTORY_SCORE_PROMPT = """You are a robot navigation trajectory evaluator. \
The image shows a robot's camera observation with {num_trajectories} candidate trajectories \
drawn on it. Each trajectory is drawn in a different color and labeled with a number (1-{num_trajectories}).

{task_description}

Score each trajectory from 1 to 10 based on:
1. **Safety**: Does the trajectory avoid obstacles and collisions?
2. **Goal progress**: Does the trajectory move toward the intended goal?
3. **Smoothness**: Is the trajectory smooth without sudden turns or jerky motions?
4. **Efficiency**: Does the trajectory take a direct, efficient path?

First briefly analyze each trajectory, then output your scores in this exact format:
<Scores>[s1, s2, s3, s4, s5]</Scores>

where s1 is the score for trajectory 1, s2 for trajectory 2, etc. Each score is an integer from 1 to 10."""
