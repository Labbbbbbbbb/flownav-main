TRAJECTORY_SCORE_PROMPT = """You are a robot navigation trajectory evaluator. \
The image shows a robot's camera observation with {num_trajectories} candidate trajectories \
drawn on it. 
Each trajectory is drawn by a different color in turns of (255, 0, 0)red,(0, 200, 0)green,(0, 100, 255)blue,(255, 165, 0)orange,(200, 0, 255)purple.

{task_description}

Score each trajectory in different colors based on:
1. **Safety**: Does the trajectory avoid obstacles and collisions?
2. **Goal progress**: Does the trajectory move toward the intended goal?
3. **Smoothness**: Is the trajectory smooth without sudden turns or jerky motions?
4. **Efficiency**: Does the trajectory take a direct, efficient path?

First briefly analyze each trajectory, then output your scores in this exact format:
<Scores>[s1, s2, s3, s4, s5]</Scores>

where s1 is the score for red, s2 for green, s3 for blue, s4 for orange, and s5 for purple. Each score is an integer from 1 to 10."""
