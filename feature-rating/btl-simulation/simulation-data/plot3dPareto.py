import matplotlib
from tkinter import *
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define Pareto front points
pareto_points = [(1, 1, 1), (2, 0.5, 0.6), (3, 0.2, 0.5), (4, 0.1, 0.4), (5, 0.05, 0.3), (6, 0.025, 0.2)]

# Sort points by first objective function in ascending order
pareto_points.sort(key=lambda x: x[0])

# Initialize the Pareto front with the first point
pareto_front = [pareto_points[0]]

# Iterate through points and add to Pareto front if it has better values for both the second and third objective functions
best_2nd_obj = pareto_points[0][1]
best_3rd_obj = pareto_points[0][2]
for point in pareto_points[1:]:
    if point[1] < best_2nd_obj and point[2] < best_3rd_obj:
        pareto_front.append(point)
        best_2nd_obj = point[1]
        best_3rd_obj = point[2]

# Plot the Pareto front
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_values = [point[0] for point in pareto_front]
y_values = [point[1] for point in pareto_front]
z_values = [point[2] for point in pareto_front]

ax.scatter(x_values, y_values, z_values)

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.set_title('Pareto Front')

plt.show()

import numpy as np

def pareto_frontier(X):
    """
    Returns the Pareto frontier of a set of points with multiple objectives.

    Parameters:
        X (ndarray): A 2D numpy array where each row represents a point in n-dimensional space.

    Returns:
        ndarray: A 2D numpy array where each row represents a point on the Pareto frontier.
    """
    # Convert the input to a numpy array
    X = np.array(X)

    # Calculate the number of objectives
    num_objs = X.shape[1]

    # Sort the points by the first objective
    X_sorted = X[X[:, 0].argsort()]

    # Initialize the Pareto frontier
    frontier = [X_sorted[0]]

    # Loop over the remaining points
    for i in range(1, len(X_sorted)):
        point = X_sorted[i]

        # Check if the point dominates any of the existing frontier points
        dominated = False
        to_remove = []
        for j in range(len(frontier)):
            if all(point <= frontier[j]) and any(point < frontier[j]):
                # The point is dominated by an existing frontier point
                dominated = True
                break
            elif all(point >= frontier[j]) and any(point > frontier[j]):
                # The point dominates an existing frontier point
                to_remove.append(j)

        # Remove dominated frontier points
        if to_remove:
            to_remove.reverse()
            for j in to_remove:
                frontier.pop(j)

        # Add the non-dominated point to the frontier
        if not dominated:
            frontier.append(point)

    return np.array(frontier)

# Generate some sample data
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(10*np.pi*X) + np.cos(10*np.pi*Y)
Z2 = np.sin(20*np.pi*X) + np.cos(20*np.pi*Y)
Z3 = np.sin(30*np.pi*X) + np.cos(30*np.pi*Y)
Z4 = np.sin(40*np.pi*X) + np.cos(40*np.pi*Y)
Z5 = np.sin(50*np.pi*X) + np.cos(50*np.pi*Y)

# Compute the Pareto front
data = np.array([Z1.ravel(), Z2.ravel(), Z3.ravel(), Z4.ravel(), Z5.ravel()]).T
front = pareto_frontier(data)

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(front[:,0], front[:,1], front[:,2], linewidth=0.2, antialiased=True)
plt.show()

