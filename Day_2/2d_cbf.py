import numpy as np
import matplotlib.pyplot as plt

# Simulation params
dt = 0.1
T = 30
steps = int(T / dt)

# Robot state
x = np.array([0.0, 0.0])  # Initial position
goal = np.array([10.0, 10.0])
goal_thresh = 0.2

# Obstacle
obs = np.array([5.0, 5.0])
obs_radius = 1.5

# Logs
trajectory = [x.copy()]

def cbf_control(x, goal, obs, r_safe):
    v_desired = goal - x
    dist_to_goal = np.linalg.norm(v_desired)
    if dist_to_goal < goal_thresh:
        return np.zeros_like(v_desired)
    v_desired = v_desired / dist_to_goal

    # CBF safety constraint
    d = x - obs
    normal_dist = np.linalg.norm(d)
    h = normal_dist - r_safe

    # Project only when inside the danger zone
    if h < 0:
        grad_h = d / normal_dist
        v_proj = v_desired - np.dot(v_desired, grad_h) * grad_h
        if np.linalg.norm(v_proj) < 1e-2:
            # Fallback: take an orthogonal detour if projection collapses
            v_proj = np.array([-v_desired[1], v_desired[0]])
        v_proj = v_proj / np.linalg.norm(v_proj)
        return v_proj
    else:
        return v_desired

step = 0
while np.linalg.norm(x - goal) > goal_thresh and step < steps:
    u = cbf_control(x, goal, obs, obs_radius + 0.3)
    x = x + dt * u
    trajectory.append(x.copy())
    step += 1

trajectory = np.array(trajectory)

# Plot
plt.figure(figsize=(6,6))
plt.plot(trajectory[:,0], trajectory[:,1], 'b-', label="Robot Path")
plt.scatter(*goal, c='g', label='Goal')
plt.scatter(*obs, c='r', label='Obstacle')
circle = plt.Circle(obs, obs_radius, color='r', alpha=0.3)
plt.gca().add_patch(circle)
plt.grid()
plt.legend()
plt.title("2D CBF Obstacle Avoidance")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.savefig('cbf_trajectory.png')

if np.linalg.norm(x - goal) <= goal_thresh:
    print("✅ Goal reached successfully!")
else:
    print("❌ Robot failed to reach the goal in time.")