import numpy as np
import matplotlib.pyplot as plt

# Known parameters
x1, y1 = 100, 100  # Fixed point (injunction)

# Initial approximations for the coordinates
x_init = np.array([200.0, 200.0, 100.0, 300.0, 50.0])  # x2, x3, x4, x5, x6
y_init = np.array([65.0, 200.0, 200.0, 150.0, 150.0])   # y2, y3, y4, y5, y6

# Observed distances
observed_distances = np.array([104.4226, 141.4264, 100.0186, 206.1519, 70.6993, 130.0119, 164.0010,
                               128.0688, 169.9940, 100.0009, 111.7834, 158.1090, 206.1490, 70.6874, 250.0094])

# Observation pairs (from-to)
obs_pairs = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
             (2, 3), (2, 4), (2, 5), (2, 6),
             (3, 4), (3, 5), (3, 6),
             (4, 5), (4, 6),
             (5, 6)]

# Observation precision (standard deviation)
sigma_obs = 0.010  # 10mm as stated in the problem
weight_matrix = np.eye(len(observed_distances)) / (sigma_obs**2)  # Weight matrix (P)

# Distance function for two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to compute all distances for the network
def compute_all_distances(params):
    xs = np.zeros(6)
    ys = np.zeros(6)
    xs[0], ys[0] = x1, y1  # P1 is fixed
    xs[1:] = params[:5]
    ys[1:] = params[5:]

    calc_distances = np.zeros(len(obs_pairs))
    for i, (from_idx, to_idx) in enumerate(obs_pairs):
        from_idx -= 1  # Convert to 0-based index
        to_idx -= 1
        calc_distances[i] = distance((xs[from_idx], ys[from_idx]), (xs[to_idx], ys[to_idx]))
    return calc_distances

# Function to compute residuals
def compute_residuals(params):
    calc_distances = compute_all_distances(params)
    residuals = observed_distances - calc_distances
    return residuals

# Jacobian of the residual function
def compute_jacobian(params):
    xs = np.zeros(6)
    ys = np.zeros(6)
    xs[0], ys[0] = x1, y1  # P1 is fixed
    xs[1:] = params[:5]
    ys[1:] = params[5:]

    J = np.zeros((len(observed_distances), len(params)))
    for i, (from_idx, to_idx) in enumerate(obs_pairs):
        from_idx -= 1  # Convert to 0-based index
        to_idx -= 1
        dx = xs[to_idx] - xs[from_idx]
        dy = ys[to_idx] - ys[from_idx]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist != 0:
            J[i, from_idx] = -dx / dist
            J[i, to_idx] = dx / dist
            J[i, 5 + from_idx] = -dy / dist
            J[i, 5 + to_idx] = dy / dist
    return J

# Newton's method for adjustment
def newton_adjustment(initial_params, max_iter=100, tol=1e-8):
    params = np.copy(initial_params)
    
    for iteration in range(max_iter):
        residuals = compute_residuals(params)
        J = compute_jacobian(params)
        
        # Compute the normal equations: J^T P J delta = J^T P r
        JT_P = np.dot(J.T, weight_matrix)
        JT_P_J = np.dot(JT_P, J)
        JT_P_r = np.dot(JT_P, residuals)
        
        # Solve for delta (parameter update)
        delta = np.linalg.solve(JT_P_J, JT_P_r)
        
        # Update the parameters
        params += delta
        
        # Check for convergence (using the norm of the update)
        if np.linalg.norm(delta) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
    
    return params

# Initial parameter vector
initial_params = np.concatenate([x_init, y_init])

# Solve using Newton's method
adjusted_params = newton_adjustment(initial_params)

# Extract the results
x_final = adjusted_params[:5]
y_final = adjusted_params[5:]

# Calculate the final distances and residuals
final_distances = compute_all_distances(adjusted_params)
residuals = observed_distances - final_distances

# Calculate the variance factor
v_T_P_v = np.sum(residuals**2 * weight_matrix.diagonal())
n_obs = len(observed_distances)
n_params = len(initial_params)
degrees_of_freedom = n_obs - n_params
sigma0_squared = v_T_P_v / degrees_of_freedom

# Print results
print("Trilateration Network Adjustment Results")
print("=" * 60)
print(f"Final Adjusted Coordinates:")
for i, (point, x, y) in enumerate(zip(["P2", "P3", "P4", "P5", "P6"], x_final, y_final)):
    print(f"{point}: X = {x:.4f}, Y = {y:.4f}")
print()

# Display residuals
print("Observation Residuals:")
for i, (from_point, to_point) in enumerate(obs_pairs):
    observed = observed_distances[i]
    calculated = final_distances[i]
    residual = (observed - calculated) * 1000  # Convert to mm
    print(f"{from_point}-{to_point}      {observed:.4f}        {calculated:.4f}        {residual:.2f}")
print()

# Statistical Summary
print("Statistical Summary:")
print(f"A posteriori variance factor (σ₀²): {sigma0_squared:.8f}")
print(f"A posteriori standard deviation (σ₀): {np.sqrt(sigma0_squared):.8f}")
print()

# Plot the network
plt.figure(figsize=(10, 8))

# Plot the fixed point P1
plt.scatter(x1, y1, color='red', s=100, zorder=5)
plt.annotate(f"P1 ({x1}, {y1})", (x1, y1), xytext=(5, 5), textcoords='offset points')

# Plot initial and final positions
for i, (point, x_i, y_i, x_f, y_f) in enumerate(zip(["P2", "P3", "P4", "P5", "P6"], x_init, y_init, x_final, y_final)):
    plt.scatter(x_i, y_i, color='blue', alpha=0.5, s=80, zorder=4)
    plt.annotate(f"{point} init", (x_i, y_i), xytext=(5, 5), textcoords='offset points', alpha=0.5)

    plt.scatter(x_f, y_f, color='green', s=80, zorder=5)
    plt.annotate(f"{point} ({x_f:.2f}, {y_f:.2f})", (x_f, y_f), xytext=(5, 5), textcoords='offset points')

# Plot the network connections
xs = np.concatenate(([x1], x_final))
ys = np.concatenate(([y1], y_final))

for i, (from_point, to_point) in enumerate(obs_pairs):
    from_idx = from_point - 1
    to_idx = to_point - 1

    plt.plot([xs[from_idx], xs[to_idx]], [ys[from_idx], ys[to_idx]], 'k--', alpha=0.5, zorder=1)

    mid_x = (xs[from_idx] + xs[to_idx]) / 2
    mid_y = (ys[from_idx] + ys[to_idx]) / 2
    calculated_dist = final_distances[i]
    plt.annotate(f"{calculated_dist:.2f}", (mid_x, mid_y),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                 ha='center', va='center')

plt.title("Trilateration Network Adjustment")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
