# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph again
G = nx.DiGraph()

# Define nodes again
nodes = {
    "User Perspective Data": "User Perspective Data Collection",
    "UI Generation": "UI Generation & Display distortion",
    "Projection Plane": "Projection Plane Calculation",
    "Dynamic Adjustment": "Dynamic Adjustment Based on User View",
    "Projected Image": "Projected Image Rendering",
    "Final Alignment": "Final User View Alignment",
}

# Define edges again
edges = [
    ("User Perspective Data", "Projection Plane"),
    ("User Perspective Data", "Dynamic Adjustment"),
    ("UI Generation", "Projection Plane"),
    ("Projection Plane", "Dynamic Adjustment"),
    ("Dynamic Adjustment", "Projected Image"),
    ("Projected Image", "Final Alignment"),
]

# Add nodes and edges to graph
for key, label in nodes.items():
    G.add_node(key, label=label)

G.add_edges_from(edges)

# Adjusted layout to prevent overlapping text
pos = {
    "User Perspective Data": (-2, 2),
    "UI Generation": (2, 2),
    "Projection Plane": (0, 1),
    "Dynamic Adjustment": (0, 0),
    "Projected Image": (0, -1),
    "Final Alignment": (0, -2),
}

# Create figure
plt.figure(figsize=(12, 7))

# Draw graph with new layout
nx.draw(G, pos, with_labels=False, node_size=3500, node_color="lightblue", edge_color="black", font_size=10)

# Add labels separately to prevent overlap
for node, (x, y) in pos.items():
    plt.text(x, y, nodes[node], fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Show improved diagram
plt.title("Improved Block Diagram of UI Projection and Adjustment Process", fontsize=12)
plt.show()
