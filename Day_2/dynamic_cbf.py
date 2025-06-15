import numpy as np
import matplotlib.pyplot as plt

# Time setup
dt = 0.1
T = 30
steps = int(T / dt)

# Robot setup
x = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])

# Moving obstacle setup
obs = np.array([5.0, 0.0])        # Starting position
obs_velocity = np.array([0.0, 0.1])  # Moves upward
obs_radius = 1.5

# Log trajectories
robot_trajectory = [x.copy()]
obs_trajectory = [obs.copy()]
hx_log = []

def cbf_control(x, goal, obs, r_safe):
    v_desired = goal - x
    v_desired = v_desired / np.linalg.norm(v_desired)

    d = x - obs
    dist = np.linalg.norm(d)
    h = dist - r_safe
    hx_log.append(h)

    if h <= 0.5:  # Danger zone
        grad_h = d / dist
        v_proj = v_desired - np.dot(v_desired, grad_h) * grad_h
        v_proj = v_proj / np.linalg.norm(v_proj)
        return v_proj
    else:
        return v_desired

for _ in range(steps):
    # Move obstacle
    obs = obs + dt * obs_velocity

    # CBF control
    u = cbf_control(x, goal, obs, obs_radius + 0.5)
    x = x + dt * u

    # Logging
    robot_trajectory.append(x.copy())
    obs_trajectory.append(obs.copy())

# Convert to array
robot_trajectory = np.array(robot_trajectory)
obs_trajectory = np.array(obs_trajectory)

# Plot
plt.figure(figsize=(7,7))
plt.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'b-', label="Robot Path")
plt.plot(obs_trajectory[:,0], obs_trajectory[:,1], 'r--', label="Obstacle Path")
plt.scatter(*goal, c='g', label='Goal')
plt.grid()
plt.legend()
plt.title("CBF with Moving Obstacle")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Plot h(x) over time
plt.figure()
plt.plot(hx_log)
plt.title("Safety Margin h(x) Over Time")
plt.xlabel("Timestep")
plt.ylabel("h(x)")
plt.axhline(0, color='r', linestyle='--')
plt.grid()
plt.savefig('cbf_hx_log.png')