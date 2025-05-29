import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper Functions
# -----------------------------
def compute_line(p1, p2):
    """
    Given two points p1, p2 on a line in 2D, compute the normal vector w
    and offset b for the line defined by: w^T x + b = 0.
    
    Returns:
        w: numpy array of shape (2,)
        b: scalar
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    # Direction vector of the line
    d = p2 - p1
    
    # A normal to (dx, dy) can be (dy, -dx)
    w = np.array([d[1], -d[0]], dtype=float)
    
    # Compute offset: from 0 = w^T p1 + b, we get b = -w^T p1
    b = - w.dot(p1)
    return w, b

def classify_point(x, w, b):
    """
    Classify a 2D point x using the line w^T x + b = 0.
    Returns the signed distance (w^T x + b).
    """
    return w.dot(x) + b

def side_label(val):
    """Return a text label based on the sign of the value."""
    if np.isclose(val, 0.0):
        return "on the line"
    elif val > 0:
        return "positive side"
    else:
        return "negative side"

# -----------------------------
# Define Points for Figure 2.7(a) and (b)
# -----------------------------
# Figure 2.7(a) points
v1 = [1.25, 4.1]
v2 = [5.0, 7.4]

# Figure 2.7(b) points
z1 = [2.6, 4.25]
z2 = [2.0, 7.0]

# Points to classify
a = np.array([1.4, 5.1])
b_pt = np.array([4.7, 7.0])  # renamed to avoid conflict with offset b

# -----------------------------
# Compute Line Parameters for Both Figures
# -----------------------------
# Figure 2.7(a)
w_a, b_a = compute_line(v1, v2)
print("Figure 2.7(a):")
print("  Points:", v1, "and", v2)
print(f"  w = {w_a},  b = {b_a:.4f}")
print(f"  Line equation: {w_a[0]:.2f} x1 + {w_a[1]:.2f} x2 + {b_a:.2f} = 0")
print()

# Figure 2.7(b)
w_b, b_b = compute_line(z1, z2)
print("Figure 2.7(b):")
print("  Points:", z1, "and", z2)
print(f"  w = {w_b},  b = {b_b:.4f}")
print(f"  Line equation: {w_b[0]:.2f} x1 + {w_b[1]:.2f} x2 + {b_b:.2f} = 0")
print()

# -----------------------------
# Classify Points a and b for Both Lines
# -----------------------------
val_a_a = classify_point(a, w_a, b_a)
val_a_b = classify_point(b_pt, w_a, b_a)
print("Classification using Figure 2.7(a) line:")
print(f"  w^T a + b = {val_a_a:.3f}  =>  a is on the {side_label(val_a_a)}")
print(f"  w^T b + b = {val_a_b:.3f}  =>  b is on the {side_label(val_a_b)}")
print()

val_b_a = classify_point(a, w_b, b_b)
val_b_b = classify_point(b_pt, w_b, b_b)
print("Classification using Figure 2.7(b) line:")
print(f"  w^T a + b = {val_b_a:.3f}  =>  a is on the {side_label(val_b_a)}")
print(f"  w^T b + b = {val_b_b:.3f}  =>  b is on the {side_label(val_b_b)}")
print()

# -----------------------------
# Plotting the Results
# -----------------------------
# For plotting lines, we'll solve for y given x: y = (-w[0]*x - b)/w[1]
# Choose an x-range that covers all points.
all_x = np.array([v1[0], v2[0], z1[0], z2[0], a[0], b_pt[0]])
x_min = all_x.min() - 2
x_max = all_x.max() + 2
x_vals = np.linspace(x_min, x_max, 400)

# Compute y-values for both lines
y_vals_a = (-w_a[0] * x_vals - b_a) / w_a[1]
y_vals_b = (-w_b[0] * x_vals - b_b) / w_b[1]

# Create subplots: one for each figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# -----------------------------
# Plot for Figure 2.7(a)
# -----------------------------
ax1.plot(x_vals, y_vals_a, 'k-', label='Line (Fig 2.7(a))')
ax1.scatter(v1[0], v1[1], color='red', marker='o', s=100, label='v1')
ax1.scatter(v2[0], v2[1], color='red', marker='o', s=100, label='v2')
ax1.scatter(a[0], a[1], color='blue', marker='^', s=100, label='Point a')
ax1.scatter(b_pt[0], b_pt[1], color='green', marker='s', s=100, label='Point b')

# Plot the normal vector for Fig 2.7(a)
mid_a = (np.array(v1) + np.array(v2)) / 2.0
w_unit_a = w_a / np.linalg.norm(w_a)
arrow_scale = 1.0  # adjust length of arrow if desired
ax1.arrow(mid_a[0], mid_a[1], w_unit_a[0]*arrow_scale, w_unit_a[1]*arrow_scale, 
          color='m', head_width=0.2, head_length=0.3, label='Normal vector')
ax1.text(mid_a[0] + w_unit_a[0]*arrow_scale*1.1, mid_a[1] + w_unit_a[1]*arrow_scale*1.1, 
         'w', color='m', fontsize=12)

ax1.set_title('Figure 2.7(a)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend()
ax1.set_aspect('equal', adjustable='box')

# -----------------------------
# Plot for Figure 2.7(b)
# -----------------------------
ax2.plot(x_vals, y_vals_b, 'k-', label='Line (Fig 2.7(b))')
ax2.scatter(z1[0], z1[1], color='red', marker='o', s=100, label='z1')
ax2.scatter(z2[0], z2[1], color='red', marker='o', s=100, label='z2')
ax2.scatter(a[0], a[1], color='blue', marker='^', s=100, label='Point a')
ax2.scatter(b_pt[0], b_pt[1], color='green', marker='s', s=100, label='Point b')

# Plot the normal vector for Fig 2.7(b)
mid_b = (np.array(z1) + np.array(z2)) / 2.0
w_unit_b = w_b / np.linalg.norm(w_b)
ax2.arrow(mid_b[0], mid_b[1], w_unit_b[0]*arrow_scale, w_unit_b[1]*arrow_scale, 
          color='m', head_width=0.2, head_length=0.3, label='Normal vector')
ax2.text(mid_b[0] + w_unit_b[0]*arrow_scale*1.1, mid_b[1] + w_unit_b[1]*arrow_scale*1.1, 
         'w', color='m', fontsize=12)

ax2.set_title('Figure 2.7(b)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()
ax2.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()