import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes with labels
nodes = {
    "User Perspective Data": "User Perspective Data Collection",
    "UI Generation": "UI Generation & Display Distortion",
    "Projection Plane": "Projection Plane Calculation",
    "Dynamic Adjustment": "Dynamic Adjustment Based on User View",
    "Projected Image": "Projected Image Rendering",
    "Final Alignment": "Final User View Alignment",
}

# Define edges
edges = [
    ("User Perspective Data", "Projection Plane"),
    ("User Perspective Data", "Dynamic Adjustment"),
    ("UI Generation", "Projection Plane"),
    ("Projection Plane", "Dynamic Adjustment"),
    ("Dynamic Adjustment", "Projected Image"),
    ("Projected Image", "Final Alignment"),
]

# Add nodes and edges to the graph
for key, label in nodes.items():
    G.add_node(key, label=label)

G.add_edges_from(edges)

# Ultra-compact zigzag layout
zigzag_pos = {
    "User Perspective Data": (-1, 0.5),
    "UI Generation": (-0.5, -0.5),
    "Projection Plane": (0, 0.5),
    "Dynamic Adjustment": (0.5, -0.5),
    "Projected Image": (1, 0.5),
    "Final Alignment": (1.5, -0.5),
}

# Create figure with a small aspect ratio
plt.figure(figsize=(4, 2))

# Draw graph with compact zigzag layout
nx.draw(G, zigzag_pos, with_labels=False, node_size=1500, node_color="lightblue", edge_color="black", font_size=6)

# Add labels with compact spacing
for node, (x, y) in zigzag_pos.items():
    plt.text(x, y, nodes[node], fontsize=6, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

# Show final ultra-compact diagram
plt.title("Ultra-Compact Zig-Zag Block Diagram of UI Projection and Adjustment Process", fontsize=8)
plt.show()
