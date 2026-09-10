import pandas as pd
from torch_geometric_temporal.dataset import METRLADatasetLoader

print("--- PHASE 1: Data Pipeline & Feature Scaling ---")

# 1. Fetching the dataset (Automatically applies Z-Score Feature Scaling, and fills NaN values  )
# We set inputs and outputs to 12 steps (60 minutes past -> 60 minutes future)
loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

# 2. Extracting snapshots to verify the Temporal Windows
snapshots = list(dataset)
print(f"Total temporal snapshots generated: {len(snapshots)}")

# 3. Data Verification (Exploratory Data Analysis)
first_snapshot = snapshots[0]
print("\n--- Verifying Snapshot Data Structures ---")
print(f"Node Features (X) shape: {first_snapshot.x.shape}")
print(f"Edge Index (Topology) shape: {first_snapshot.edge_index.shape}")
print(f"Target (Y) shape: {first_snapshot.y.shape}") 

# 4. Verifying the automatic scaling on Sensor 0
sensor_0_history = [snapshot.y[0, 0].item() for snapshot in snapshots[:5]]
df = pd.DataFrame(sensor_0_history, columns=["Normalized Speed (Z-Scores)"])
print("\n--- Sample of Scaled Data (First 5 Timesteps) ---")
print(df)

print("\nData Pipeline verified. Ready for STGCN Training.") 