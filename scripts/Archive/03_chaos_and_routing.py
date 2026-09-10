import torch
import torch.nn.functional as F
import networkx as nx
import folium
import numpy as np
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.dataset import METRLADatasetLoader

print("--- PHASE 3: Chaos Engineering & Dynamic Routing ---")

# 1. Redefine Architecture to Load Weights
class EmergencyRoutingSTGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dimensions):
        super(EmergencyRoutingSTGCN, self).__init__()
        self.tgcn = TGCN(in_channels=node_features, out_channels=hidden_dimensions)
        self.linear = torch.nn.Linear(hidden_dimensions, 1)

    def forward(self, x, edge_index, edge_weight):
        h = None
        for t in range(x.size(2)):
            h = self.tgcn(x[:, :, t], edge_index, edge_weight, h)
        return self.linear(F.relu(h))

# 2. Load the Frozen Brain and Data
model = EmergencyRoutingSTGCN(node_features=2, hidden_dimensions=32)
model.load_state_dict(torch.load('output/stgcn_la_weights.pt', weights_only=True))
model.eval()

loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
snapshots = list(dataset)
snapshot = snapshots[0] # Selecting one test scenario

# 3. The Chaos Function (Matrix Surgery)
def sever_road_connection(edge_index, edge_weight, origin_node, dest_node):
    mask = ~((edge_index[0] == origin_node) & (edge_index[1] == dest_node))
    return edge_index[:, mask], edge_weight[mask]

# Simulating a massive crash closing the route from Sensor 0 to Sensor 13
broken_edge_index, broken_edge_weight = sever_road_connection(
    snapshot.edge_index, snapshot.edge_attr, origin_node=0, dest_node=13
)

print(f"Original active connections: {snapshot.edge_index.shape[1]}")
print(f"Active connections after failure: {broken_edge_index.shape[1]}")

# 4. Generate AI Crisis Prediction
with torch.no_grad():
    crisis_prediction = model(snapshot.x, broken_edge_index, broken_edge_weight)

# 5. Wardrop's Translation & NetworkX Graph Construction
city_graph = nx.DiGraph()
predicted_speeds = crisis_prediction.detach().cpu().numpy().flatten()
edges_origin = broken_edge_index[0].cpu().numpy()
edges_destination = broken_edge_index[1].cpu().numpy()

for i in range(len(edges_origin)):
    origin = edges_origin[i]
    destination = edges_destination[i]
    speed = predicted_speeds[destination]
    
    if speed <= 0: speed = 1.0 # Safety constraint
        
    time_penalty = 1.0 / speed # Wardrop Math
    city_graph.add_edge(origin, destination, weight=time_penalty)

# 6. Dijkstra's Emergency Dispatch
start_node = 0
end_node = 115

try:
    emergency_route = nx.shortest_path(city_graph, source=start_node, target=end_node, weight='weight')
    route_cost = nx.path_weight(city_graph, emergency_route, weight='weight')
    print(f"\nOptimal Path to bypass congestion: {emergency_route}")
    print(f"Total Artificial Time Penalty: {route_cost:.4f}")
except nx.NetworkXNoPath:
    print(f"\nCRITICAL FAILURE: No viable path exists.")

# 7. Geospatial Visualization (Folium)
sensor_coordinates = {
    0: (34.0522, -118.2437),   # Start
    13: (34.0450, -118.2500),  # Destroyed Node
    116: (34.0450, -118.2550), # Arterial Detour
    115: (34.0300, -118.2700)  # Crisis Zone
}

route_gps = [sensor_coordinates[int(node)] for node in emergency_route if int(node) in sensor_coordinates]

# Fallback coordinates if Dijkstra finds a path outside our 4 mock GPS points
if not route_gps: route_gps = [sensor_coordinates[0], sensor_coordinates[116], sensor_coordinates[115]]

start_lat, start_lon = route_gps[0]
la_map = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles='CartoDB dark_matter')

# Draw severed road
folium.PolyLine([sensor_coordinates[0], sensor_coordinates[13]], color='red', weight=8, dash_array='10, 10', tooltip='SEVERED LINK').add_to(la_map)
# Draw AI Detour
folium.PolyLine(route_gps, color='cyan', weight=5, tooltip='AI Optimal Reroute').add_to(la_map)
# Add Waypoints
folium.Marker(route_gps[0], popup='START', icon=folium.Icon(color='green')).add_to(la_map)
folium.Marker(route_gps[-1], popup='END', icon=folium.Icon(color='orange')).add_to(la_map)

# Save the map to the output folder
map_path = "output/Los_Angeles_Dynamic_Dispatch.html"
la_map.save(map_path)
print(f"\nMap successfully exported to '{map_path}'.")