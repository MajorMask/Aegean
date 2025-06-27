import numpy as np
import matplotlib.pyplot as plt

# Setup
dt = 0.1
T = 60  # Extended simulation time
steps = int(T / dt)
goal_tolerance = 0.3

# Initial robot state
x = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])
path = [x.copy()]

# Obstacles (true position and velocity)
obstacles = [
    {"pos": np.array([5.0, 5.0]), "vel": np.array([-0.02, 0.01]), "radius": 1.5},
    {"pos": np.array([8.0, 2.0]), "vel": np.array([-0.015, 0.02]), "radius": 1.0},
    {"pos": np.array([2.0, 9.0]), "vel": np.array([0.01, -0.01]), "radius": 1.2},
]

# Fake CV: adds Gaussian noise to detected obstacle positions
def fake_cv_detection(obs):
    noisy_obs = []
    for o in obs:
        pos_noisy = o["pos"] + np.random.normal(0, 0.1, 2)
        noisy_obs.append({"pos": pos_noisy, "radius": o["radius"]})
    return noisy_obs

# Control Barrier Function: generates safe direction
def control_barrier_fn(x, goal, perceived_obstacles):
    u_nominal = goal - x
    u_nominal = u_nominal / (np.linalg.norm(u_nominal) + 1e-6)
    u = u_nominal * 0.3

    for obs in perceived_obstacles:
        diff = x - obs["pos"]
        dist = np.linalg.norm(diff)
        safety_margin = obs["radius"] + 1.0

        if dist < safety_margin:
            repulsion_dir = diff / (dist + 1e-6)
            tangent_dir = np.array([-repulsion_dir[1], repulsion_dir[0]])  # 90-degree rotation
            blend_factor = (safety_margin - dist) / safety_margin
            repulsion = blend_factor * repulsion_dir * 0.5 + (1 - blend_factor) * tangent_dir * 0.3
            u += repulsion

    max_speed = 0.5
    speed = np.linalg.norm(u)
    if speed > max_speed:
        u = u / speed * max_speed
    return u

# Simulation loop
max_iters = 2000
for t in range(max_iters):
    if np.linalg.norm(x - goal) < goal_tolerance:
        print(f"Reached goal at step {t}")
        break

    for o in obstacles:
        o["pos"] += o["vel"] * dt

    cv_obs = fake_cv_detection(obstacles)
    u = control_barrier_fn(x, goal, cv_obs)

    x += u * dt
    path.append(x.copy())
else:
    print("Did not reach the goal within maximum iterations.")

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], 'b-', label='Robot Path')
ax.plot(goal[0], goal[1], 'go', label='Goal')

for o in obstacles:
    ax.plot(o["pos"][0], o["pos"][1], 'ro')
    circ = plt.Circle(o["pos"], o["radius"], color='r', alpha=0.2)
    ax.add_patch(circ)

ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_title("2D CBF with Moving Obstacles & Fake CV")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid()
plt.gca().set_aspect('equal')

plt.savefig('multiple_obstacles.png')