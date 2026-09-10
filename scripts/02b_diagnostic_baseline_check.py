# # """
# # Diagnostic: Persistence Baseline vs. Trained Model
# # Run this AFTER 02_stgcn_training.py to check whether the model is actually
# # learning traffic dynamics, or just approximating a naive "predict last known
# # speed" baseline.
# # """
# # import torch
# # from torch_geometric_temporal.dataset import METRLADatasetLoader
# # from torch_geometric_temporal.signal import temporal_signal_split

# # from model import EmergencyRoutingSTGCN
# # import config

# # print("--- DIAGNOSTIC: Persistence Baseline vs. Trained Model ---")

# # loader = METRLADatasetLoader()
# # dataset = loader.get_dataset(
# #     num_timesteps_in=config.NUM_TIMESTEPS_IN,
# #     num_timesteps_out=config.NUM_TIMESTEPS_OUT
# # )
# # train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
# # test_dataset = list(test_dataset)

# # # Load trained model (same architecture params as training, pulled from config
# # # so there's no risk of a mismatch between how it was trained and how it's loaded)
# # model = EmergencyRoutingSTGCN(
# #     node_features=config.NODE_FEATURES,
# #     hidden_dimensions=config.HIDDEN_DIMENSIONS
# # )
# # model.load_state_dict(torch.load(config.CHECKPOINT_PATH, weights_only=True))
# # model.eval()

# # model_mae_total = 0.0
# # persistence_mae_total = 0.0
# # step_count = 0

# # with torch.no_grad():
# #     for snapshot in test_dataset:
# #         target = snapshot.y[:, 0].unsqueeze(1)           # true next value
# #         last_known = snapshot.x[:, 0, -1].unsqueeze(1)   # most recent observed value

# #         prediction = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

# #         model_mae_total += torch.nn.L1Loss()(prediction, target).item()
# #         persistence_mae_total += torch.nn.L1Loss()(last_known, target).item()
# #         step_count += 1

# # model_mae = model_mae_total / step_count
# # persistence_mae = persistence_mae_total / step_count

# # print(f"\nModel Test MAE (scaled):           {model_mae:.4f}")
# # print(f"Persistence Baseline MAE (scaled): {persistence_mae:.4f}")

# # if model_mae < persistence_mae:
# #     improvement = ((persistence_mae - model_mae) / persistence_mae) * 100
# #     print(f"\n✅ Model beats persistence baseline by {improvement:.1f}%. This is a real, defensible result.")
# # else:
# #     print("\n⚠️  Model does NOT beat the naive persistence baseline.")
# #     print("This means the model likely isn't learning meaningful traffic dynamics yet.")
# #     print("Do not report this as a result without investigating further.")

# # print("\n--- Notes ---")
# # print("- 'Scaled' means Z-score normalized space. Multiply both MAE values by the")
# # print("  dataset's speed standard deviation to get real mph MAE for comparison")
# # print("  against published benchmarks (DCRNN, Graph WaveNet, etc. -- verify current")
# # print("  published numbers before citing in your paper).")







# """
# Diagnostic: Persistence Baseline vs. Trained Model
# Run this AFTER 02_stgcn_training.py to check whether the model is actually
# learning traffic dynamics, or just approximating a naive "predict last known
# speed" baseline.
# """
# import torch
# from torch_geometric_temporal.dataset import METRLADatasetLoader
# from torch_geometric_temporal.signal import temporal_signal_split

# from model import EmergencyRoutingSTGCN
# import config

# print("--- DIAGNOSTIC: Persistence Baseline vs. Trained Model ---")

# device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# loader = METRLADatasetLoader()
# dataset = loader.get_dataset(
#     num_timesteps_in=config.NUM_TIMESTEPS_IN,
#     num_timesteps_out=config.NUM_TIMESTEPS_OUT
# )
# train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
# test_dataset = list(test_dataset)

# # Load trained model (same architecture params as training, pulled from config
# # so there's no risk of a mismatch between how it was trained and how it's loaded)
# model = EmergencyRoutingSTGCN(
#     node_features=config.NODE_FEATURES,
#     hidden_dimensions=config.HIDDEN_DIMENSIONS
# ).to(device)
# model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True))
# model.eval()

# model_mae_total = 0.0
# persistence_mae_total = 0.0
# step_count = 0

# # CHANGED: also track a longer horizon (60 min ahead = index 11) since
# # persistence is expected to dominate at the 5-min horizon (index 0) almost
# # by construction -- the real test of whether the model learned anything is
# # whether it starts beating persistence at longer horizons.
# model_mae_total_long = 0.0
# persistence_mae_total_long = 0.0

# with torch.no_grad():
#     for snapshot in test_dataset:
#         x = snapshot.x.to(device)
#         edge_index = snapshot.edge_index.to(device)
#         edge_attr = snapshot.edge_attr.to(device)
#         y = snapshot.y.to(device)

#         target = y[:, 0].unsqueeze(1)           # true value 5 min ahead
#         target_long = y[:, -1].unsqueeze(1)      # true value 60 min ahead (last of 12 steps)
#         last_known = x[:, 0, -1].unsqueeze(1)   # most recent observed value

#         prediction = model(x, edge_index, edge_attr)

#         model_mae_total += torch.nn.L1Loss()(prediction, target).item()
#         persistence_mae_total += torch.nn.L1Loss()(last_known, target).item()

#         # NOTE: current model architecture only outputs a single-step prediction,
#         # so we compare that same single-step prediction against the 60-min-ahead
#         # true value too -- this isn't a fair long-horizon forecast yet (the model
#         # was never trained to predict 60 min out), but it tells us how much
#         # persistence itself degrades at longer horizons, which is the key
#         # missing piece of context.
#         model_mae_total_long += torch.nn.L1Loss()(prediction, target_long).item()
#         persistence_mae_total_long += torch.nn.L1Loss()(last_known, target_long).item()

#         step_count += 1

# model_mae = model_mae_total / step_count
# persistence_mae = persistence_mae_total / step_count
# model_mae_long = model_mae_total_long / step_count
# persistence_mae_long = persistence_mae_total_long / step_count

# print(f"\n--- 5-min horizon (what you trained on) ---")
# print(f"Model Test MAE (scaled):           {model_mae:.4f}")
# print(f"Persistence Baseline MAE (scaled): {persistence_mae:.4f}")

# if model_mae < persistence_mae:
#     improvement = ((persistence_mae - model_mae) / persistence_mae) * 100
#     print(f"✅ Model beats persistence baseline by {improvement:.1f}%.")
# else:
#     print("⚠️  Model does NOT beat persistence at this horizon.")

# print(f"\n--- 60-min horizon (persistence baseline only, for context) ---")
# print(f"Persistence Baseline MAE at 60 min (scaled): {persistence_mae_long:.4f}")
# print(f"(Model's single-step output compared against 60-min target, NOT a real")
# print(f" 60-min forecast -- model was never trained for this horizon. Shown only")
# print(f" to illustrate how much persistence degrades over a longer horizon:")
# print(f" {persistence_mae:.4f} at 5 min -> {persistence_mae_long:.4f} at 60 min)")

# print("\n--- Notes ---")
# print("- 'Scaled' means Z-score normalized space. Multiply both MAE values by the")
# print("  dataset's speed standard deviation to get real mph MAE for comparison")
# print("  against published benchmarks (DCRNN, Graph WaveNet, etc. -- verify current")
# print("  published numbers before citing in your paper).")






"""
Diagnostic: Persistence Baseline vs. Trained Model
Run this AFTER 02_stgcn_training.py to check whether the model is actually
learning traffic dynamics, or just approximating a naive "predict last known
speed" baseline.
"""
import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from model import EmergencyRoutingSTGCN
import config

print("--- DIAGNOSTIC: Persistence Baseline vs. Trained Model ---")

device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loader = METRLADatasetLoader()
dataset = loader.get_dataset(
    num_timesteps_in=config.NUM_TIMESTEPS_IN,
    num_timesteps_out=config.NUM_TIMESTEPS_OUT
)
train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=config.TRAIN_RATIO)
test_dataset = list(test_dataset)

# Load trained model (same architecture params as training, pulled from config
# so there's no risk of a mismatch between how it was trained and how it's loaded)
model = EmergencyRoutingSTGCN(
    node_features=config.NODE_FEATURES,
    hidden_dimensions=config.HIDDEN_DIMENSIONS
).to(device)
model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=True))
model.eval()

model_mae_total = 0.0
persistence_mae_total = 0.0
step_count = 0

with torch.no_grad():
    for snapshot in test_dataset:
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        y = snapshot.y.to(device)

        # CHANGED: compare at the SAME horizon the model was trained on
        # (config.PREDICTION_HORIZON_INDEX), not a fixed 5-min target.
        # This is now a fair, apples-to-apples comparison.
        target = y[:, config.PREDICTION_HORIZON_INDEX].unsqueeze(1)
        last_known = x[:, 0, -1].unsqueeze(1)   # most recent observed value

        prediction = model(x, edge_index, edge_attr)

        model_mae_total += torch.nn.L1Loss()(prediction, target).item()
        persistence_mae_total += torch.nn.L1Loss()(last_known, target).item()
        step_count += 1

model_mae = model_mae_total / step_count
persistence_mae = persistence_mae_total / step_count

horizon_minutes = config.PREDICTION_HORIZON_INDEX * 5 + 5
print(f"\n--- {horizon_minutes}-min horizon (what you trained on) ---")
print(f"Model Test MAE (scaled):           {model_mae:.4f}")
print(f"Persistence Baseline MAE (scaled): {persistence_mae:.4f}")

if model_mae < persistence_mae:
    improvement = ((persistence_mae - model_mae) / persistence_mae) * 100
    print(f"\n✅ Model beats persistence baseline by {improvement:.1f}%. This is a real, defensible result.")
else:
    print("\n⚠️  Model does NOT beat the naive persistence baseline.")
    print("Do not report this as a result without investigating further.")

print("\n--- Notes ---")
print("- 'Scaled' means Z-score normalized space. Multiply both MAE values by the")
print("  dataset's speed standard deviation to get real mph MAE for comparison")
print("  against published benchmarks (DCRNN, Graph WaveNet, etc. -- verify current")
print("  published numbers before citing in your paper).")