# nrs_act

Refactored ACT-based imitation learning codebase for robotic polishing / manipulation experiments.  
This repository is organized around a modular `source/` layout and now supports a **split observation encoder** with **force history** input.

---

## 1. Overview

`nrs_act` is an imitation learning project built on a customized ACT codebase and later refactored for maintainability and future research patches.

Current baseline characteristics:
- ACT-based behavior cloning / imitation learning
- Observation = **position/orientation + force + multi-camera RGB**
- Action = **position/orientation + force**
- Modular structure: `common / data / models / training`
- Main entrypoint kept at `scripts/act/train_act.py`
- Force-history-aware encoder support added without changing raw `.hdf5` demo files

This project is designed so that future patches can be added mainly under `source/` while keeping `scripts/act/train_act.py` as an orchestration entrypoint.

---

## 2. Credits / Origin / Upstream

This repository is **not a from-scratch implementation**. It is a refactored research codebase derived from a customized ACT implementation and upstream ACT/DETR components.

### Original customized ACT codebase
- **Chemin Ahn**
- Homepage: `https://chemx3937.github.io/`
- GitHub: `https://github.com/Chemx3937`

### Upstream references
1. **ACT: Action Chunking with Transformers**
   - Tony Z. Zhao
   - Project page: `https://tonyzhaozh.github.io/aloha/`

2. **DETR**
   - Facebook Research
   - GitHub: `https://github.com/facebookresearch/detr`

### Attribution rule
When sharing, patching, or redistributing this repository:
- keep credit to **Chemin Ahn**
- keep attribution to **ACT**
- keep attribution to **DETR**
- do not remove license / attribution files

---

## 3. License

The root `LICENSE` keeps integrated upstream notices.

Included upstream licenses:
- **ACT**: MIT License  
  Copyright (c) 2023 Tony Z. Zhao
- **DETR**: Apache License 2.0  
  Copyright 2020-present, Facebook, Inc.

Notes:
- root `LICENSE` must be preserved
- `README.md` and attribution notes should continue to mention the original customized code origin
- if new external code is added later, its license notice must also be preserved

---

## 4. Current Project Structure

```text
nrs_act/
├── LICENSE
├── README.md
├── checkpoints/
├── datasets/
├── scripts/
│   └── act/
│       └── train_act.py
└── source/
    ├── common/
    │   ├── fs.py
    │   └── utils.py
    ├── custom/
    │   ├── check_cam_serial.py
    │   ├── custom_constants.py
    │   ├── custom_real_env.py
    │   ├── custom_robot_utils.py
    │   └── demo_data_act_form.py
    ├── data/
    │   ├── dataset.py
    │   ├── loader.py
    │   └── normalization.py
    ├── models/
    │   ├── __init__.py
    │   ├── act_core.py
    │   ├── backbone.py
    │   ├── encoder.py
    │   ├── policy.py
    │   └── transformer.py
    └── training/
        ├── debug.py
        ├── engine.py
        └── plotting.py
```

### Key idea of the refactor
Previous monolithic logic was split by responsibility:
- `data/` → dataset / normalization / dataloader
- `models/` → ACT core / policy / encoder / backbone / transformer
- `training/` → train loop / debug / plotting
- `common/` → general shared utilities

---

## 5. What Each Folder Does

### `scripts/act/`
Training / evaluation entrypoint.

Main file:
- `train_act.py`

Responsibilities:
- parse CLI arguments
- assemble policy config
- call `load_data(...)`
- call `train_bc(...)`
- handle evaluation mode
- save checkpoint directory and dataset stats

Design rule:
- keep this file as thin as possible
- future algorithmic patches should mostly go into `source/`

---

### `source/common/`
General utilities.

Files:
- `fs.py` → checkpoint folder lookup helpers such as latest timestamped subdir search
- `utils.py` → common helpers like seeding and dictionary utilities

---

### `source/data/`
Dataset and dataloader logic.

Files:
- `dataset.py`
- `loader.py`
- `normalization.py`

Responsibilities:
- read `episode_*.hdf5`
- sample episode start timesteps
- build current observation and action chunk
- normalize qpos/action with per-dimension min-max
- optionally build **force history** on-the-fly from raw episode force trajectory
- create train / val `DataLoader`

This folder is the main patch point for:
- contact labels
- phase labels
- onset weighting
- previous-action history
- force-history generation
- normalization changes

---

### `source/models/`
Model definition and wrappers.

Files:
- `encoder.py` → **new split observation encoders**
- `act_core.py` → ACT / CNNMLP core model builders
- `backbone.py` → CNN image backbone + positional encoding
- `transformer.py` → transformer encoder/decoder
- `policy.py` → training-facing policy wrapper and losses
- `__init__.py` → package import convenience

This folder is the main patch point for:
- encoder changes
- auxiliary heads
- force/contact prediction heads
- loss weighting
- fusion changes
- model architecture extensions

---

### `source/training/`
Training loop and debugging.

Files:
- `engine.py` → training / validation loop
- `debug.py` → normalization debug and AMP helpers
- `plotting.py` → training history plotting

Responsibilities:
- batch forward pass
- support 4-item and 5-item batch formats
- validation / checkpoint save
- normalization debug print
- optional AMP handling

---

### `source/custom/`
Custom environment / task-specific helpers kept from the original research workflow.

This includes optional task config or hardware/environment support code used around the ACT project.

---

## 6. Observation / Action Definition

### Current action definition
The model still predicts the same 9D action as before:

\[
a_t = [x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

So **the action space has not changed**.

### Current observation definition
The current observation remains based on:
- pose/orientation: `x y z wx wy wz`
- force: `fx fy fz`
- multi-camera RGB images

What changed is **how the observation is encoded**.

---

## 7. Old Encoder Structure vs New Encoder Structure

## Before
A single shared state encoder processed the current 9D state directly:

\[
q_t = [x,y,z,w_x,w_y,w_z,f_x,f_y,f_z]
\]

\[
e_t = \phi_{shared}(q_t)
\]

Characteristics:
- position/orientation and force were mixed immediately
- force used only the current timestep
- no temporal force context

---

## After
The observation is now encoded in a split manner.

### 1) Position encoder
Current pose/orientation is encoded separately:

\[
p_t = [x,y,z,w_x,w_y,w_z]
\]

\[
e_t^{pos} = \phi_{pos}(p_t)
\]

### 2) Force encoder (GRU)
A short force history window is encoded:

\[
H_t = [f_{t-L+1}, \dots, f_t], \quad f_t = [f_x,f_y,f_z]
\]

\[
e_t^{force} = \phi_{force}(H_t)
\]

Here `phi_force` is a **GRU-based force-history encoder**.

### 3) Fusion encoder
Position and force embeddings are fused into one observation embedding:

\[
e_t = \phi_{fuse}([e_t^{pos}; e_t^{force}])
\]

### 4) Image encoder
RGB observations are encoded by the image backbone:

\[
e_t^{img} = \phi_{img}(I_t)
\]

These are then used by ACT / CNNMLP policy logic.

---

## 8. Why the New Structure Matters

The new structure improves the state representation in two ways.

### A. Position / force disentangling
Before, pose and force were forced into the same encoder.  
Now they are represented separately first, which reduces early entanglement.

### B. Temporal force modeling
Before, the model only saw current force:

\[
f_t
\]

Now it can see force trend:

\[
f_{t-L+1}, \dots, f_t
\]

This is especially important for:
- non-contact → contact transition
- pressing phase detection
- force continuity
- contact-aware action generation

---

## 9. Force History: How It Is Added Without Changing Raw `.hdf5`

Raw demo files are **not rewritten**.

Instead, `source/data/dataset.py` builds force history on-the-fly from the episode’s full force trajectory.

If the current sampled timestep is `t`, dataset constructs:

\[
H_t = [f_{t-L+1}, \dots, f_t]
\]

using `/observations/force` inside the same episode file.

### Episode start padding
If `t < L-1`, the left side is padded by repeating the first available force value.

Example:

\[
[f_0, f_0, \dots, f_0, f_1, \dots, f_t]
\]

### Normalization of force history
`force_history` uses the same min-max statistics as the force part of `qpos`.

So the raw `.hdf5` stays the same, while the dataset becomes history-aware.

---

## 10. Files Changed for the New Force-History Pipeline

### Newly added / important
- `source/models/encoder.py`

### Main modified files
- `source/models/act_core.py`
- `source/models/policy.py`
- `source/data/dataset.py`
- `source/data/loader.py`
- `source/training/engine.py`
- `source/training/debug.py`
- `scripts/act/train_act.py`

### Roles of these changes
- `encoder.py` → position encoder / force GRU encoder / image encoder definitions
- `dataset.py` → builds `force_history`
- `loader.py` → enables force-history dataset mode
- `engine.py` → supports both 4-item and 5-item batches
- `debug.py` → prints `force_history` stats too
- `train_act.py` → exposes CLI flags for force history

---

## 11. Encoder Components in `source/models/encoder.py`

### `PositionStateEncoder`
Encodes:

\[
[x,y,z,w_x,w_y,w_z]
\]

into a learned embedding.

### `ForceHistoryGRUEncoder`
Encodes:

\[
[f_{t-L+1}, \dots, f_t]
\]

with a GRU and uses the last hidden state as force embedding.

### `PositionForceFusionEncoder`
Takes concatenated position and force embeddings and maps them to one fused embedding.

Currently implemented as shallow fusion:

\[
\text{Linear} + \text{Activation}
\]

### `ImageObservationEncoder`
Used for ACT image features.

### `CNNMLPImageEncoder`
Used for the CNNMLP baseline path.

---

## 12. Current Data Format Assumption

Each dataset directory should contain:

```text
episode_0.hdf5
episode_1.hdf5
...
```

Expected keys:
- `/observations/position`
- `/observations/force`
- `/observations/images`
- `/observations/is_pad`
- `/action/position`
- `/action/force`

Current dimensionality:
- observation qpos: `position(6) + force(3) = 9D`
- action: `position(6) + force(3) = 9D`
- force history: `(L, 3)` when enabled

Camera names used by default:
- `cam_top`
- `cam_ee`

Fallback mapping:
- `cam_top -> cam_front`
- `cam_ee -> cam_head`

---

## 13. Normalization

### qpos / action
Per-dimension min-max normalization to `[0, 1]`.

For each dimension independently:
- x
- y
- z
- wx
- wy
- wz
- fx
- fy
- fz

### images
- uint8 → float `[0,1]`
- ImageNet normalization is still applied inside `source/models/policy.py`

### force history
When enabled, `force_history` is normalized using the same min/max used for the force portion of qpos.

---

## 14. Training Flow

Training entrypoint:

```bash
python3 scripts/act/train_act.py ...
```

Flow:
1. parse CLI args
2. resolve dataset dir / task config
3. build `policy_config`
4. call `load_data(...)`
5. create train / val loaders
6. optionally print normalization debug
7. build policy
8. train / validate / save checkpoints

---

## 15. Main Training Command

### Standard ACT training with force history

```bash
cd /home/eunseop/nrs_act && python3 scripts/act/train_act.py \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-4 \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --debug_norm \
  --use_force_history \
  --force_history_len 10
```

### Without force history

```bash
cd /home/eunseop/nrs_act && python3 scripts/act/train_act.py \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-4 \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --debug_norm
```

---

## 16. Important CLI Flags for the New Structure

### Force history flags
- `--use_force_history`  
  enables dataset-side force history generation and passes it to the model

- `--force_history_len 10`  
  sets the GRU history window length

### Split encoder hyperparameters
- `--position_dim`
- `--force_dim`
- `--position_encoder_hidden_dim`
- `--force_encoder_hidden_dim`
- `--force_encoder_num_layers`
- `--force_encoder_dropout`
- `--observation_encoder_activation`
- `--cnnmlp_observation_embed_dim`

---

## 17. Inference / Evaluation Notes

Evaluation command:

```bash
cd /home/eunseop/nrs_act && python3 scripts/act/train_act.py \
  --eval \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-4 \
  --use_force_history \
  --force_history_len 10
```

### Important
If the model was trained with force history, online inference should also maintain a recent force buffer:

\[
[f_{t-L+1}, \dots, f_t]
\]

Otherwise train-time and inference-time input structures do not match.

---

## 18. Checkpoints and Saved Files

Training creates a timestamped directory under `checkpoints/<task_name>/...`.

Typical contents:
- `policy_best.ckpt`
- `policy_last.ckpt`
- `dataset_stats.pkl`
- optional plot outputs

`dataset_stats.pkl` stores normalization statistics for later denormalization / deployment.

---

## 19. Debug Output

When `--debug_norm` is enabled, training prints normalized statistics before training begins.

Current debug output includes:
- image shape
- qpos shape
- action shape
- is_pad shape
- force_history shape (if enabled)
- qpos per-dimension mean/std
- action per-dimension mean/std
- image RGB mean/std
- force_history mean/std
- range checks for normalized values

This is useful to verify:
- normalization correctness
- force-history value scale
- dataset pipeline integrity
- train/val split sanity

---

## 20. What Did *Not* Change

Even after the new encoder patch:
- final action dimension is still **9D**
- output action remains:

\[
[x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

So the patch changes the **input representation**, not the final action definition.

---

## 21. Expected Effect of the New Structure

### Before
- current force only
- no temporal force trend
- pose and force immediately entangled

### After
- separate position encoder
- GRU-based force-history encoder
- fused observation representation
- better opportunity to model:
  - force transition
  - contact onset
  - contact maintenance
  - force-aware action chunk prediction

Expected inference-side benefits:
- more context-aware force prediction
- better non-contact → contact transition handling
- more consistent force-conditioned action chunks
- cleaner separation between geometry and force representation

---

## 22. Current Limitations

This patch improves representation, but does **not** automatically solve all force prediction issues.

Still likely future patch targets:
- force dimension loss weighting
- contact auxiliary loss
- phase label / phase loss
- onset weighting
- auxiliary heads from latent / hidden state

Most likely future files to patch:
- `source/data/dataset.py`
- `source/models/encoder.py`
- `source/models/act_core.py`
- `source/models/policy.py`
- `source/training/engine.py`

---

## 23. Recommended Patch Rules Going Forward

1. keep `scripts/act/train_act.py` thin
2. keep folder responsibilities separated
3. do not mix dataset logic into model files unnecessarily
4. preserve attribution and license files
5. preserve import stability (`__init__.py` where needed)
6. prefer modifying `source/` rather than rewriting the training script

---

## 24. Summary

`nrs_act` is now in a stronger baseline state for force-aware imitation learning.

Current baseline status:
- refactored modular structure completed
- ACT training runs successfully
- position / force encoder separation added
- force-history GRU path added
- raw `.hdf5` dataset format preserved
- dataset-side on-the-fly force-history generation supported
- debug pipeline updated to show force-history statistics

In short, the project has evolved from:

\[
\text{shared current-state encoder}
\]

into:

\[
\text{position encoder} + \text{force-history GRU encoder} + \text{fusion encoder}
\]

while still predicting the same 9D action target.

