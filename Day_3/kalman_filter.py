# Ensure required packages are imported
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("output_kf_visuals", exist_ok=True)

# Simulation Parameters
dt = 0.1
T = 20
steps = int(T / dt)

# True Trajectory
def true_trajectory(t):
    return 0.5 * t * dt, 0.5 * t * dt

true_pos = np.array([true_trajectory(t) for t in range(steps)])

# Simulated Noisy GPS Measurements
np.random.seed(42)
measured_pos = true_pos + np.random.normal(0, 0.3, true_pos.shape)

# Kalman Filter Initialization
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.eye(4) * 0.01
R = np.eye(2) * 0.09
P = np.eye(4)
x_est = np.array([0, 0, 0, 0])

kf_estimates = []
cbf_estimates = []

# Multiple Moving Obstacles with circular paths
obstacles = [
    {
        'center_fn': lambda t, i=i: [5 + np.sin(0.1 * t + i), 5 + np.cos(0.1 * t + i)],
        'radius': 0.8
    }
    for i in range(3)
]

# Kalman Filter with CBF Loop
for t in range(steps):
    z = measured_pos[t]

    # Kalman Prediction
    x_pred = A @ x_est
    P_pred = A @ P @ A.T + Q

    # Kalman Update
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred

    estimate = x_est[:2]
    kf_estimates.append(estimate)

    # CBF Correction for multiple obstacles
    corrected_estimate = estimate.copy()
    for obs in obstacles:
        obs_center = np.array(obs['center_fn'](t))
        obs_radius = obs['radius']
        dist = np.linalg.norm(corrected_estimate - obs_center)
        if dist < obs_radius + 0.3:
            direction = (corrected_estimate - obs_center) / (dist + 1e-5)
            corrected_estimate = obs_center + (obs_radius + 0.3) * direction

    cbf_estimates.append(corrected_estimate)

kf_estimates = np.array(kf_estimates)
cbf_estimates = np.array(cbf_estimates)

# Plot 1: True Path vs Noisy GPS
plt.figure(figsize=(6, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label='True Path')
plt.plot(measured_pos[:, 0], measured_pos[:, 1], 'rx', label='Noisy GPS')
plt.title("True Path vs Noisy GPS")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.axis('equal')
plt.savefig("output_kf_visuals/true_vs_gps.png")

# Plot 2: KF Estimate vs True
plt.figure(figsize=(6, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label='True Path')
plt.plot(kf_estimates[:, 0], kf_estimates[:, 1], 'b--', label='KF Estimate')
plt.title("Kalman Filter Estimate vs True Path")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.axis('equal')
plt.savefig("output_kf_visuals/kf_vs_true.png")

# Plot 3: CBF-Corrected Path vs Moving Obstacles
plt.figure(figsize=(6, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label='True Path')
plt.plot(cbf_estimates[:, 0], cbf_estimates[:, 1], 'm-.', label='CBF Corrected Path')
for t in range(0, steps, 5):
    for obs in obstacles:
        obs_center = obs['center_fn'](t)
        circle = plt.Circle(obs_center, obs['radius'], color='red', alpha=0.2)
        plt.gca().add_patch(circle)
plt.title("CBF-Corrected Path with Moving Obstacles")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.axis('equal')
plt.savefig("output_kf_visuals/cbf_vs_obstacles.png")

# Plot 4: All Combined
plt.figure(figsize=(8, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label='True Path')
plt.plot(measured_pos[:, 0], measured_pos[:, 1], 'rx', label='Noisy GPS')
plt.plot(kf_estimates[:, 0], kf_estimates[:, 1], 'b--', label='KF Estimate')
plt.plot(cbf_estimates[:, 0], cbf_estimates[:, 1], 'm-.', label='CBF Corrected')
for t in range(0, steps, 5):
    for obs in obstacles:
        obs_center = obs['center_fn'](t)
        circle = plt.Circle(obs_center, obs['radius'], color='red', alpha=0.15)
        plt.gca().add_patch(circle)
plt.title("KF, CBF and GPS Comparison with Moving Obstacles")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend(loc='best')
plt.axis('equal')
plt.savefig("output_kf_visuals/full_comparison.png")
plt.show()

print("✅ Visuals saved in 'output_kf_visuals' folder. Check: true_vs_gps.png, kf_vs_true.png, cbf_vs_obstacles.png, full_comparison.png")
