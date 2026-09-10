# import torch
# import torch.nn.functional as F
# from torch_geometric_temporal.nn.recurrent import TGCN
# from torch_geometric_temporal.dataset import METRLADatasetLoader
# from torch_geometric_temporal.signal import temporal_signal_split

# print("--- PHASE 2: STGCN Architecture & Chronological Training ---")

# # 1. Define the Neural Network Architecture
# class EmergencyRoutingSTGCN(torch.nn.Module):
#     def __init__(self, node_features, hidden_dimensions):
#         super(EmergencyRoutingSTGCN, self).__init__()
#         # Spatio-Temporal Engine
#         self.tgcn = TGCN(in_channels=node_features, out_channels=hidden_dimensions)
#         # Output Predictor
#         self.linear = torch.nn.Linear(hidden_dimensions, 1)

#     def forward(self, x, edge_index, edge_weight):
#         h = None
#         for t in range(x.size(2)):
#             x_t = x[:, :, t]
#             h = self.tgcn(x_t, edge_index, edge_weight, h)
#         h = F.relu(h)
#         return self.linear(h)

# # 2. Load Data and Enforce the Chronological Cut
# loader = METRLADatasetLoader()
# dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# print(f"Training snapshots: {len(list(train_dataset))}")
# print(f"Testing snapshots: {len(list(test_dataset))}")

# # 3. Initialize Model, Optimizer, and Loss
# model = EmergencyRoutingSTGCN(node_features=2, hidden_dimensions=32)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.MSELoss()

# # 4. The Training Loop (Calibration)
# print("\n--- Commencing Calibration ---")
# model.train()
# for epoch in range(3): # Set to 3 for quick testing, increase for final portfolio run
#     epoch_loss = 0
#     step_count = 0
#     for snapshot in train_dataset:
#         optimizer.zero_grad()
#         predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         target = snapshot.y[:, 0].unsqueeze(1)
#         loss = loss_fn(predictions, target)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         step_count += 1
    
#     avg_epoch_loss = epoch_loss / step_count
#     print(f"Epoch {epoch+1} | Average Training Loss (MSE): {avg_epoch_loss:.4f}")

# # 5. Evaluation on Unseen Future Data
# print("\n--- Commencing Evaluation on Unseen Future Data ---")
# model.eval()
# test_loss_mse = 0
# test_loss_mae = 0
# step_count = 0

# with torch.no_grad():
#     for snapshot in test_dataset:
#         predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         target = snapshot.y[:, 0].unsqueeze(1)
#         test_loss_mse += torch.nn.MSELoss()(predictions, target).item()
#         test_loss_mae += torch.nn.L1Loss()(predictions, target).item()
#         step_count += 1

# print("\n--- Final Metrics ---")
# print(f"Average MSE on Test Data: {test_loss_mse / step_count:.4f}")
# print(f"Average MAE on Test Data: {test_loss_mae / step_count:.4f} (Scaled)")

# # 6. Freeze and Save the Brain
# torch.save(model.state_dict(), 'output/stgcn_la_weights.pt')
# print("\n--- Success! Model Weights Saved to 'output/stgcn_la_weights.pt' ---")



#Epoch 50

# import torch
# import torch.nn.functional as F
# import numpy as np
# import random
# from torch_geometric_temporal.nn.recurrent import TGCN
# from torch_geometric_temporal.dataset import METRLADatasetLoader
# from torch_geometric_temporal.signal import temporal_signal_split

# print("--- PHASE 2: STGCN Architecture & Chronological Training ---")

# # 0. Reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

# # 1. Define the Neural Network Architecture
# class EmergencyRoutingSTGCN(torch.nn.Module):
#     def __init__(self, node_features, hidden_dimensions):
#         super(EmergencyRoutingSTGCN, self).__init__()
#         self.tgcn = TGCN(in_channels=node_features, out_channels=hidden_dimensions)
#         self.linear = torch.nn.Linear(hidden_dimensions, 1)

#     def forward(self, x, edge_index, edge_weight):
#         h = None
#         for t in range(x.size(2)):
#             x_t = x[:, :, t]
#             h = self.tgcn(x_t, edge_index, edge_weight, h)
#         h = F.relu(h)
#         return self.linear(h)

# # 2. Load Data and Enforce Chronological Train/Val/Test Split
# loader = METRLADatasetLoader()
# dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)

# # First split off test set (80/20), then split train into train/val (85/15)
# train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
# train_val_list = list(train_val_dataset)
# split_idx = int(len(train_val_list) * 0.85)
# train_dataset = train_val_list[:split_idx]
# val_dataset = train_val_list[split_idx:]
# test_dataset = list(test_dataset)

# print(f"Training snapshots: {len(train_dataset)}")
# print(f"Validation snapshots: {len(val_dataset)}")
# print(f"Testing snapshots: {len(test_dataset)}")

# # 3. Initialize Model, Optimizer, Scheduler, Loss
# model = EmergencyRoutingSTGCN(node_features=2, hidden_dimensions=32)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# loss_fn = torch.nn.MSELoss()

# # 4. Training Loop with Validation and Best-Checkpoint Saving
# NUM_EPOCHS = 50
# best_val_loss = float('inf')

# print("\n--- Commencing Calibration ---")
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     epoch_loss = 0
#     step_count = 0
#     for snapshot in train_dataset:
#         optimizer.zero_grad()
#         predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         target = snapshot.y[:, 0].unsqueeze(1)
#         loss = loss_fn(predictions, target)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         step_count += 1
#     avg_train_loss = epoch_loss / step_count

#     # Validation pass
#     model.eval()
#     val_loss = 0
#     val_steps = 0
#     with torch.no_grad():
#         for snapshot in val_dataset:
#             predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#             target = snapshot.y[:, 0].unsqueeze(1)
#             val_loss += loss_fn(predictions, target).item()
#             val_steps += 1
#     avg_val_loss = val_loss / val_steps

#     scheduler.step(avg_val_loss)

#     print(f"Epoch {epoch+1:02d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(model.state_dict(), 'output/stgcn_la_weights.pt')
#         print(f"  -> New best val loss. Checkpoint saved.")

# print(f"\nBest validation MSE achieved: {best_val_loss:.4f}")

# # 5. Final Evaluation on Test Set (using BEST checkpoint, not last epoch)
# print("\n--- Commencing Evaluation on Unseen Test Data (best checkpoint) ---")
# model.load_state_dict(torch.load('output/stgcn_la_weights.pt', weights_only=True))
# model.eval()

# test_loss_mse = 0
# test_loss_mae = 0
# step_count = 0
# with torch.no_grad():
#     for snapshot in test_dataset:
#         predictions = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         target = snapshot.y[:, 0].unsqueeze(1)
#         test_loss_mse += torch.nn.MSELoss()(predictions, target).item()
#         test_loss_mae += torch.nn.L1Loss()(predictions, target).item()
#         step_count += 1

# final_mse = test_loss_mse / step_count
# final_mae = test_loss_mae / step_count

# print("\n--- Final Metrics (Scaled, Z-score space) ---")
# print(f"Test MSE: {final_mse:.4f}")
# print(f"Test MAE: {final_mae:.4f}")
# print("\nNote: These are in normalized (Z-score) space. For the paper, convert back")
# print("to real mph/kph using the dataset's stored mean/std before comparing to")
# print("published METR-LA benchmarks (DCRNN, Graph WaveNet, etc.).")



# import torch
# import numpy as np
# import random
# from torch_geometric_temporal.dataset import METRLADatasetLoader
# from torch_geometric_temporal.signal import temporal_signal_split

# # CHANGED: import shared model + config instead of redefining the class
# # and hardcoding numbers in this file
# from model import EmergencyRoutingSTGCN
# import config

# print("--- PHASE 2: STGCN Architecture & Chronological Training ---")

# # CHANGED: detect device -- uses GPU automatically if available (e.g. on Colab),
# # falls back to CPU with a warning otherwise. Everything below moves model +
# # data onto this device.
# device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# if device.type == "cpu":
#     print("WARNING: No GPU detected. Training will be slow. "
#           "Consider running this on Google Colab with a T4 GPU instead.")

# # 0. Reproducibility
# torch.manual_seed(config.SEED)
# np.random.seed(config.SEED)
# random.seed(config.SEED)

# # 1. Load Data and Enforce Chronological Train/Val/Test Split
# loader = METRLADatasetLoader()
# dataset = loader.get_dataset(
#     num_timesteps_in=config.NUM_TIMESTEPS_IN,
#     num_timesteps_out=config.NUM_TIMESTEPS_OUT
# )

# train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
# train_val_list = list(train_val_dataset)
# split_idx = int(len(train_val_list) * config.TRAIN_VAL_SPLIT)
# train_dataset = train_val_list[:split_idx]
# val_dataset = train_val_list[split_idx:]
# test_dataset = list(test_dataset)

# # CHANGED: optional subset for fast local iteration/debugging. Set
# # config.SUBSET_FRACTION < 1.0 to test the pipeline quickly before committing
# # to a full run on Colab.
# if config.SUBSET_FRACTION < 1.0:
#     train_dataset = train_dataset[:int(len(train_dataset) * config.SUBSET_FRACTION)]
#     val_dataset = val_dataset[:int(len(val_dataset) * config.SUBSET_FRACTION)]
#     print(f"NOTE: Using SUBSET_FRACTION={config.SUBSET_FRACTION} for fast iteration. "
#           f"Set to 1.0 in config.py for the real full-dataset run.")

# print(f"Training snapshots: {len(train_dataset)}")
# print(f"Validation snapshots: {len(val_dataset)}")
# print(f"Testing snapshots: {len(test_dataset)}")

# # 2. Initialize Model, Optimizer, Scheduler, Loss
# model = EmergencyRoutingSTGCN(
#     node_features=config.NODE_FEATURES,
#     hidden_dimensions=config.HIDDEN_DIMENSIONS
# ).to(device)  # CHANGED: move model to GPU/CPU device
# optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE
# )
# loss_fn = torch.nn.MSELoss()

# best_val_loss = float('inf')
# epochs_without_improvement = 0

# print("\n--- Commencing Calibration ---")
# for epoch in range(config.MAX_EPOCHS):
#     model.train()
#     epoch_loss = 0
#     step_count = 0
#     optimizer.zero_grad()

#     for i, snapshot in enumerate(train_dataset):
#         # CHANGED: move each snapshot's tensors to the training device
#         x = snapshot.x.to(device)
#         edge_index = snapshot.edge_index.to(device)
#         edge_attr = snapshot.edge_attr.to(device)
#         y = snapshot.y.to(device)

#         predictions = model(x, edge_index, edge_attr)
#         target = y[:, 0].unsqueeze(1)

#         loss = loss_fn(predictions, target) / config.ACCUMULATION_STEPS
#         loss.backward()

#         if (i + 1) % config.ACCUMULATION_STEPS == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         epoch_loss += loss.item() * config.ACCUMULATION_STEPS
#         step_count += 1

#     if step_count % config.ACCUMULATION_STEPS != 0:
#         optimizer.step()
#         optimizer.zero_grad()

#     avg_train_loss = epoch_loss / step_count

#     # Validation pass
#     model.eval()
#     val_loss = 0
#     val_steps = 0
#     with torch.no_grad():
#         for snapshot in val_dataset:
#             x = snapshot.x.to(device)
#             edge_index = snapshot.edge_index.to(device)
#             edge_attr = snapshot.edge_attr.to(device)
#             y = snapshot.y.to(device)

#             predictions = model(x, edge_index, edge_attr)
#             target = y[:, 0].unsqueeze(1)
#             val_loss += loss_fn(predictions, target).item()
#             val_steps += 1
#     avg_val_loss = val_loss / val_steps

#     scheduler.step(avg_val_loss)

#     print(f"Epoch {epoch+1:03d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}", end="")

#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         epochs_without_improvement = 0
#         torch.save(model.state_dict(), config.CHECKPOINT_PATH)
#         print("  -> New best val loss. Checkpoint saved.")
#     else:
#         epochs_without_improvement += 1
#         print(f"  -> No improvement ({epochs_without_improvement}/{config.EARLY_STOP_PATIENCE})")

#     if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
#         print(f"\nEarly stopping triggered at epoch {epoch+1}. "
#               f"Val loss hasn't improved in {config.EARLY_STOP_PATIENCE} epochs.")
#         break

# print(f"\nBest validation MSE achieved: {best_val_loss:.4f}")

# # 3. Final Evaluation on Test Set (using BEST checkpoint, not last epoch)
# print("\n--- Commencing Evaluation on Unseen Test Data (best checkpoint) ---")
# model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True))
# model.eval()

# test_loss_mse = 0
# test_loss_mae = 0
# step_count = 0
# with torch.no_grad():
#     for snapshot in test_dataset:
#         x = snapshot.x.to(device)
#         edge_index = snapshot.edge_index.to(device)
#         edge_attr = snapshot.edge_attr.to(device)
#         y = snapshot.y.to(device)

#         predictions = model(x, edge_index, edge_attr)
#         target = y[:, 0].unsqueeze(1)
#         test_loss_mse += torch.nn.MSELoss()(predictions, target).item()
#         test_loss_mae += torch.nn.L1Loss()(predictions, target).item()
#         step_count += 1

# final_mse = test_loss_mse / step_count
# final_mae = test_loss_mae / step_count

# print("\n--- Final Metrics (Scaled, Z-score space) ---")
# print(f"Test MSE: {final_mse:.4f}")
# print(f"Test MAE: {final_mae:.4f}")
# print("\nNote: These are in normalized (Z-score) space. For the paper, convert back")
# print("to real mph/kph using the dataset's stored mean/std before comparing to")
# print("published METR-LA benchmarks (DCRNN, Graph WaveNet, etc.).")



import torch
import numpy as np
import random
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

# CHANGED: import shared model + config instead of redefining the class
# and hardcoding numbers in this file
from model import EmergencyRoutingSTGCN
import config

print("--- PHASE 2: STGCN Architecture & Chronological Training ---")
print(f"Training to predict {config.PREDICTION_HORIZON_INDEX * 5 + 5} minutes ahead "
      f"(horizon index {config.PREDICTION_HORIZON_INDEX})")

# CHANGED: detect device -- uses GPU automatically if available (e.g. on Colab),
# falls back to CPU with a warning otherwise. Everything below moves model +
# data onto this device.
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: No GPU detected. Training will be slow. "
          "Consider running this on Google Colab with a T4 GPU instead.")

# 0. Reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

# 1. Load Data and Enforce Chronological Train/Val/Test Split
loader = METRLADatasetLoader()
dataset = loader.get_dataset(
    num_timesteps_in=config.NUM_TIMESTEPS_IN,
    num_timesteps_out=config.NUM_TIMESTEPS_OUT
)

train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
train_val_list = list(train_val_dataset)
split_idx = int(len(train_val_list) * config.TRAIN_VAL_SPLIT)
train_dataset = train_val_list[:split_idx]
val_dataset = train_val_list[split_idx:]
test_dataset = list(test_dataset)

# CHANGED: optional subset for fast local iteration/debugging. Set
# config.SUBSET_FRACTION < 1.0 to test the pipeline quickly before committing
# to a full run on Colab.
if config.SUBSET_FRACTION < 1.0:
    train_dataset = train_dataset[:int(len(train_dataset) * config.SUBSET_FRACTION)]
    val_dataset = val_dataset[:int(len(val_dataset) * config.SUBSET_FRACTION)]
    print(f"NOTE: Using SUBSET_FRACTION={config.SUBSET_FRACTION} for fast iteration. "
          f"Set to 1.0 in config.py for the real full-dataset run.")

print(f"Training snapshots: {len(train_dataset)}")
print(f"Validation snapshots: {len(val_dataset)}")
print(f"Testing snapshots: {len(test_dataset)}")

# 2. Initialize Model, Optimizer, Scheduler, Loss
model = EmergencyRoutingSTGCN(
    node_features=config.NODE_FEATURES,
    hidden_dimensions=config.HIDDEN_DIMENSIONS
).to(device)  # CHANGED: move model to GPU/CPU device
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE
)
loss_fn = torch.nn.MSELoss()

best_val_loss = float('inf')
epochs_without_improvement = 0

print("\n--- Commencing Calibration ---")
for epoch in range(config.MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    step_count = 0
    optimizer.zero_grad()

    for i, snapshot in enumerate(train_dataset):
        # CHANGED: move each snapshot's tensors to the training device
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        y = snapshot.y.to(device)

        predictions = model(x, edge_index, edge_attr)
        target = y[:, config.PREDICTION_HORIZON_INDEX].unsqueeze(1)

        loss = loss_fn(predictions, target) / config.ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % config.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * config.ACCUMULATION_STEPS
        step_count += 1

    if step_count % config.ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = epoch_loss / step_count

    # Validation pass
    model.eval()
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for snapshot in val_dataset:
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)

            predictions = model(x, edge_index, edge_attr)
            target = y[:, config.PREDICTION_HORIZON_INDEX].unsqueeze(1)
            val_loss += loss_fn(predictions, target).item()
            val_steps += 1
    avg_val_loss = val_loss / val_steps

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1:03d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}", end="")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)
        print("  -> New best val loss. Checkpoint saved.")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement ({epochs_without_improvement}/{config.EARLY_STOP_PATIENCE})")

    if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}. "
              f"Val loss hasn't improved in {config.EARLY_STOP_PATIENCE} epochs.")
        break

print(f"\nBest validation MSE achieved: {best_val_loss:.4f}")

# 3. Final Evaluation on Test Set (using BEST checkpoint, not last epoch)
print("\n--- Commencing Evaluation on Unseen Test Data (best checkpoint) ---")
model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()

test_loss_mse = 0
test_loss_mae = 0
step_count = 0
with torch.no_grad():
    for snapshot in test_dataset:
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        y = snapshot.y.to(device)

        predictions = model(x, edge_index, edge_attr)
        target = y[:, config.PREDICTION_HORIZON_INDEX].unsqueeze(1)
        test_loss_mse += torch.nn.MSELoss()(predictions, target).item()
        test_loss_mae += torch.nn.L1Loss()(predictions, target).item()
        step_count += 1

final_mse = test_loss_mse / step_count
final_mae = test_loss_mae / step_count

print("\n--- Final Metrics (Scaled, Z-score space) ---")
print(f"Test MSE: {final_mse:.4f}")
print(f"Test MAE: {final_mae:.4f}")
print("\nNote: These are in normalized (Z-score) space. For the paper, convert back")
print("to real mph/kph using the dataset's stored mean/std before comparing to")
print("published METR-LA benchmarks (DCRNN, Graph WaveNet, etc.).")