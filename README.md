# nrs_act

Refactored ACT-based imitation learning codebase for robotic polishing and manipulation experiments.

This repository is a **refactored baseline** built from a customized ACT codebase, reorganized so that data, model, training, and common utilities are clearly separated. The current baseline is designed around **9D state/action trajectories + multi-camera RGB observations**, and is intended to make future patches—especially **force/contact-aware learning**—easier to implement and maintain.

---

## 1. Overview

### Purpose
`nrs_act` is used for ACT-based imitation learning experiments in robotic manipulation / polishing settings.

Current learning setup:
- **Observation**
  - pose-like 6D channel: position + orientation
  - force 3D channel
  - multi-camera RGB images
- **Action**
  - pose-like 6D channel: position + orientation
  - force 3D channel

In the current implementation, the pose-like part is stored in the `position` datasets as **6 dimensions**.
That is why the current state/action dimension is:
- `qpos`: 6 (pose-like position/orientation) + 3 (force) = **9D**
- `action`: 6 (pose-like position/orientation) + 3 (force) = **9D**

### Current baseline status
- ACT-based imitation learning code has been **fully refactored**.
- The old monolithic structure has been split into:
  - `source/common`
  - `source/data`
  - `source/models`
  - `source/training`
- Training entry is kept as a **single script**:
  - `scripts/act/train_act.py`
- Training has been verified to **start normally**.
- Future patches should **prefer modifying `source/` modules**, while keeping `train_act.py` as stable as possible.

---

## 2. Design principles of this refactored baseline

The current codebase follows these rules:

1. **Keep `train_act.py` minimal**
   - It should mainly handle argument parsing and high-level orchestration.
   - Training logic, model logic, loss logic, normalization, and dataset behavior should live under `source/`.

2. **Do not mix responsibilities**
   - `data/` handles dataset, stats, normalization, and dataloaders.
   - `models/` handles network definitions and training-facing policy wrappers.
   - `training/` handles loops, logging, AMP helpers, and plotting.
   - `common/` handles reusable utilities and path helpers.

3. **Preserve attribution and licenses**
   - This codebase is a refactored derivative of a customized ACT codebase.
   - Attribution to the original author and upstream projects must remain.

4. **Make future research patches easier**
   - Especially patches related to:
     - force trajectory prediction
     - non-contact to contact transition
     - auxiliary contact/phase heads
     - onset-aware loss reweighting

---

## 3. Repository structure

Core repository tree:

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
    │   ├── policy.py
    │   └── transformer.py
    └── training/
        ├── debug.py
        ├── engine.py
        └── plotting.py
```

Notes:
- Old shim-style structure has been removed.
- Legacy paths like `source/policy.py`, `source/utils.py`, `source/act_train_utils.py`, or `source/act/` are no longer the active structure.
- `__pycache__` directories can be ignored.

---

## 4. What each top-level folder does

### `checkpoints/`
Stores training outputs such as:
- timestamped checkpoint folders
- model weights
- dataset statistics (for example `dataset_stats.pkl`)
- training history / plots depending on the configuration

Typical usage:
- one task-specific directory under `checkpoints/`
- then one timestamped run directory per training run

Example:

```text
checkpoints/
└── ur10e_swing/
    └── 20260316_1547/
        └── dataset_stats.pkl
```

---

### `datasets/`
Stores imitation learning datasets.

Expected format:
- a dataset directory contains multiple episode files directly:
  - `episode_0.hdf5`
  - `episode_1.hdf5`
  - `episode_2.hdf5`
  - ...

This repository currently assumes ACT-style episodic HDF5 data with multi-camera observations and action trajectories.

---

### `scripts/`
Contains executable entry scripts.

#### `scripts/act/train_act.py`
This is the **main training / evaluation entrypoint**.

Responsibilities:
- parse command-line arguments
- assemble training configuration
- determine checkpoint and dataset paths
- call loader / policy / engine modules
- branch between training and evaluation mode

Important rule:
- **keep this file as stable as possible**
- future research changes should usually go into `source/` rather than here

---

### `source/`
Contains the actual implementation, separated by role.

Subfolders:
- `common/`
- `custom/`
- `data/`
- `models/`
- `training/`

This is the main place to modify when extending the project.

---

## 5. Detailed explanation of `source/` subfolders

## 5.1 `source/common/`
Common reusable utilities shared across multiple modules.

### `source/common/fs.py`
Path and checkpoint helper utilities.

Current known role:
- find latest timestamped checkpoint/run directory

Current function:
- `find_latest_timestamped_subdir()`

Use this when:
- locating the most recent run folder
- resuming or evaluating latest checkpoints
- standardizing checkpoint discovery logic

---

### `source/common/utils.py`
General-purpose helper functions.

Current known roles:
- seed setup
- dictionary tensor/value processing
- averaging utilities

Current functions include:
- `set_seed()`
- `detach_dict()`
- `compute_dict_mean()`

Use this when:
- reproducibility is needed
- logging detached metrics
- aggregating loss dictionaries across batches or epochs

---

## 5.2 `source/custom/`
Custom environment- or project-specific helper code.

This directory contains utilities that are not part of the ACT core architecture itself, but support project-specific experiments.

Files:
- `check_cam_serial.py`
- `custom_constants.py`
- `custom_real_env.py`
- `custom_robot_utils.py`
- `demo_data_act_form.py`

Typical roles of this folder:
- camera checks
- real-environment integration helpers
- robot-specific constants/utilities
- custom demo data handling / conversion

This directory is useful when connecting the learning code to real robot pipelines or project-specific data capture tools.

---

## 5.3 `source/data/`
All dataset-related logic.

This is the first place to modify for:
- dataset return structure changes
- new labels
- normalization changes
- loading behavior changes
- train/val split logic

### `source/data/dataset.py`
Creates actual per-sample training items.

Current known responsibilities:
- episodic sampling
- valid-length handling
- camera key resolution
- dataset object definition

Current known functions/classes:
- `_get_valid_len()`
- `_resolve_cam_key()`
- `EpisodicStartDataset`

This is the **main patch point** for dataset-side research features such as:
- `contact_label`
- `phase_label`
- `onset_weight`
- history stacking / previous action input
- return format changes

If you want to add new supervision signals, this is usually the first file to edit.

---

### `source/data/normalization.py`
Handles normalization and denormalization of state/action values.

Current known functions:
- `compute_norm_stats_all()`
- `denormalize_action()`

Current normalization policy:
- state/action (`qpos`, `action`): **per-dimension min-max normalization to [0, 1]**
- image: `uint8 -> float [0,1]`
- ImageNet normalization is applied later in the policy wrapper, not here

This file is the main patch point for:
- force-specific scaling
- switching away from min-max normalization
- removing possible train/val statistic leakage
- different normalization for state vs action
- per-group or per-channel normalization rules

---

### `source/data/loader.py`
Creates datasets, computes stats, splits train/val, and builds DataLoaders.

Current known responsibilities:
- find `episode_*.hdf5`
- split training and validation sets
- compute normalization stats
- create DataLoader objects

Current known function:
- `load_data()`

Edit this file when you need to change:
- dataset discovery rules
- episode split logic
- stats computation strategy
- batching / loader options

---

## 5.4 `source/models/`
Model architecture and training-facing policy wrappers.

This folder contains both the ACT core model and the wrapper logic that computes losses and prepares images.

### `source/models/backbone.py`
CNN backbone utilities.

Current known contents:
- `FrozenBatchNorm2d`
- positional encoding helpers
- backbone builder
- `build_backbone()`

Use this file when changing:
- image encoder backbone structure
- positional features
- vision feature extraction details

---

### `source/models/transformer.py`
Transformer implementation used by the model.

Current known contents:
- Transformer
- Encoder / Decoder
- `build_transformer()`

Use this file when changing:
- encoder/decoder depth
- hidden interactions
- attention flow
- sequence handling inside the ACT model

---

### `source/models/act_core.py`
Core model definitions and model/optimizer builders.

Current known contents:
- ACT core model
- CNNMLP core model
- `DETRVAE`
- model builder functions
- optimizer builder functions

Current known functions:
- `build_ACT_model_and_optimizer()`
- `build_CNNMLP_model_and_optimizer()`

This is the main patch point for model-architecture changes such as:
- force prediction head
- contact head
- phase head
- latent representation changes
- output head restructuring
- hidden-state branching for auxiliary tasks

---

### `source/models/policy.py`
Training-facing wrapper around the core model.

Current known contents:
- `ACTPolicy`
- `CNNMLPPolicy`
- `kl_divergence`
- image normalization
- masked loss
- `configure_optimizers()`

This file currently handles important training behavior such as:
- image preprocessing before the model
- reconstruction loss calculation
- KL term handling for ACT
- padding-aware loss masking

Current loss behavior:
- **ACTPolicy**
  - action reconstruction: **masked L1**
  - latent KL: `kl_divergence`
  - total loss = `l1 + kl_weight * kl`
- **CNNMLPPolicy**
  - MSE loss

Important detail:
- padded timesteps are masked out
- current implementation already corrects for the issue where padded timesteps shrink the effective loss scale by using a valid-count-based normalization

This is the main patch point for:
- force-dimension weighting
- auxiliary loss terms
- contact BCE loss
- phase-conditioned loss
- masked loss revisions

---

## 5.5 `source/training/`
Training loop, debugging helpers, AMP utilities, and plotting.

### `source/training/debug.py`
Debugging and mixed precision helper utilities.

Current known roles:
- normalization debug output
- AMP helper functions

Current known functions:
- `debug_norm_once()`
- `make_grad_scaler()`
- `autocast_context()`

Use this file when changing:
- debug print behavior
- AMP/autocast convenience logic
- grad scaler setup

---

### `source/training/plotting.py`
Training history plotting.

Current known function:
- `plot_history()`

Use this file when modifying:
- loss curve visualization
- saved training figure formatting
- post-training metric plots

---

### `source/training/engine.py`
Core training and validation loop.

Current known responsibilities:
- create the policy object
- define forward pass behavior
- train/validation loop

Current known functions:
- `make_policy()`
- `forward_pass()`
- `train_bc()`

This is the main patch point for:
- training loop control
- logging frequency
- validation frequency
- gradient clipping
- checkpoint save rules
- mixed precision integration
- extra metrics and training outputs

---

## 6. Current import structure

Current import relationships:

### `scripts/act/train_act.py`
Imports:
- `from training.engine import train_bc, make_policy`
- `from common.fs import find_latest_timestamped_subdir`
- `from data.loader import load_data`

### `source/training/engine.py`
Imports:
- `from common.utils import compute_dict_mean, set_seed`
- `from models.policy import ACTPolicy, CNNMLPPolicy`
- `from training.debug import ...`
- `from training.plotting import ...`

### `source/models/policy.py`
Imports:
- `from .act_core import ...`

Important notes:
- `source/models/__init__.py` must exist
- it is strongly recommended to also keep `__init__.py` in `common/`, `data/`, and `training/` if package-style imports are used
- if import errors appear, check:
  1. missing `__init__.py`
  2. incorrect path setup in `train_act.py`
  3. typo in module paths

---

## 7. How to run training

### Typical training command

```bash
cd /home/eunseop/nrs_act
python3 scripts/act/train_act.py \
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

### Example shell alias

```bash
alias train_act='cd /home/eunseop/nrs_act && python3 scripts/act/train_act.py --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing --policy_class ACT --task_name ur10e_swing --batch_size 6 --seed 0 --num_epochs 500 --lr 1e-4 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --debug_norm'
```

### Evaluation mode example

```bash
cd /home/eunseop/nrs_act
python3 scripts/act/train_act.py \
  --eval \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/ur10e_swing \
  --policy_class ACT \
  --task_name ur10e_swing \
  --batch_size 6 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-4
```

---

## 8. Expected dataset format

A dataset directory is expected to contain episode files directly:

```text
dataset_dir/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
└── ...
```

### Keys currently expected by the loader

Observation side:
- `/observations/position`
- `/observations/force`
- `/observations/images`
- `/observations/is_pad`

Action side:
- `/action/position`
- `/action/force`

### Current dimensional convention
- `/observations/position`: pose-like 6D
- `/observations/force`: 3D
- `/action/position`: pose-like 6D
- `/action/force`: 3D

Therefore:
- `qpos = 6 + 3 = 9D`
- `action = 6 + 3 = 9D`

### Camera names
Current expected camera names:
- `cam_top`
- `cam_ee`

Fallback mapping:
- `cam_top -> cam_front`
- `cam_ee -> cam_head`

---

## 9. Current normalization and loss behavior

## 9.1 Normalization
Current normalization rules:
- `qpos` / `action`: **per-dimension min-max normalization to [0,1]**
- image: `uint8 -> float [0,1]`
- ImageNet normalization: applied in `source/models/policy.py`

This means image normalization is split across two stages:
1. basic numeric scaling in the data path
2. ImageNet normalization in the policy wrapper

---

## 9.2 Loss
### ACT policy
- reconstruction loss: **masked L1**
- KL term: `kl_divergence`
- total:

```text
total = l1 + kl_weight * kl
```

### CNNMLP policy
- MSE loss

### Padding behavior
Padded timesteps are excluded by masked loss.
The current implementation already compensates for the loss-scaling issue that can happen when many padded timesteps exist by using valid-count-aware normalization.

---

## 10. What changed in the refactor

### Before refactor
Major logic was mixed together:
- backbone / transformer / ACT core / wrapper in one place
- dataset / normalization / loader / common utils mixed together
- training loop / debug / plotting / checkpoint helper mixed together

### After refactor
The code is separated by role:

- `source/models/`
  - `policy.py`
  - `act_core.py`
  - `backbone.py`
  - `transformer.py`
- `source/data/`
  - `dataset.py`
  - `normalization.py`
  - `loader.py`
- `source/training/`
  - `engine.py`
  - `debug.py`
  - `plotting.py`
- `source/common/`
  - `utils.py`
  - `fs.py`

This separation is the key baseline benefit of the current project.

---

## 11. Where to modify for future patches

This is the most important maintenance guide for the current baseline.

## A. Dataset / label patches
Examples:
- add `contact_label`
- add `phase_label`
- add `onset_weight`
- add previous action / history stack
- change dataset return format

Main files to edit:
- `source/data/dataset.py`
- sometimes `source/data/loader.py`
- sometimes `source/data/normalization.py`

---

## B. Normalization patches
Examples:
- force-only scaling
- replace min-max normalization
- train-only stats to avoid leakage
- different normalization rule for state and action

Main files to edit:
- `source/data/normalization.py`
- sometimes `source/data/loader.py`

---

## C. Policy wrapper / loss patches
Examples:
- higher weight on force dimensions
- contact auxiliary BCE loss
- phase-conditioned loss
- onset-region reweighting
- masked loss behavior change

Main files to edit:
- `source/models/policy.py`
- sometimes `source/training/engine.py`

---

## D. Core model architecture patches
Examples:
- add force head
- add contact head
- add phase head
- change latent structure
- branch hidden representation
- modify transformer output usage

Main files to edit:
- `source/models/act_core.py`
- sometimes `source/models/backbone.py`
- sometimes `source/models/transformer.py`

---

## E. Training loop / logging / AMP patches
Examples:
- progress logging changes
- gradient clipping
- validation frequency changes
- checkpoint save interval changes
- mixed precision changes

Main files to edit:
- `source/training/engine.py`
- `source/training/debug.py`
- `source/training/plotting.py`

---

## F. Checkpoint / path logic patches
Examples:
- latest checkpoint discovery changes
- checkpoint naming rule changes
- run folder handling changes

Main files to edit:
- `source/common/fs.py`
- sometimes `scripts/act/train_act.py`

---

## 12. Most likely next research direction

Current practical concern:
- position/orientation trajectory generation can work relatively well,
- but force trajectory prediction—especially the **transition from non-contact (`fz = 0`) to contact (`fz > 0`)**—may remain unstable.

Most likely next patch directions:

### 1) Dataset-side label enhancement
- `contact_label`
- `phase_label`
- `onset_weight`

### 2) Loss-side reinforcement
- increase force-dimension loss weight
- add contact auxiliary BCE loss
- add phase prediction loss
- reweight onset / transition regions

### 3) Model-side auxiliary outputs
- auxiliary head on top of the main action head
- contact prediction head
- phase prediction head
- hidden-state- or latent-based auxiliary prediction

Most likely key files for the next stage:
- `source/data/dataset.py`
- `source/models/policy.py`
- `source/models/act_core.py`
- `source/training/engine.py`

---

## 13. Rules that should not be broken

1. **Keep `train_act.py` minimal whenever possible**
   - Do not move full training logic back into the script.

2. **Do not mix `data / models / training / common` responsibilities**
   - Do not move dataset logic into model files.
   - Do not move loss/model logic into data files.

3. **Preserve attribution and license notices**
   - Keep original attribution visible.
   - Do not remove license texts.

4. **Protect package import structure**
   - Keep `source/models/__init__.py`
   - Prefer keeping `__init__.py` in other package-like folders as well

---

## 14. Common problems and checks

### Problem 1: package import error
Example:

```text
ModuleNotFoundError: No module named 'models.policy'
```

Check:
- does the directory have `__init__.py`?
- is `source/` added to the import path correctly?
- is the import string spelled correctly?

---

### Problem 2: missing function after file split
Examples:
- `build_backbone` missing
- `train_bc` missing

Check:
- whether the split file was fully saved
- whether the function was copied completely
- use grep/search to confirm the symbol exists

---

### Problem 3: mixing old and new paths
Do **not** mix:
- old/legacy structure
- `source/models/policy.py`
- removed shim modules

Current active path is:
- `source/models/policy.py`

---

### Problem 4: too many `__pycache__` folders
These are safe to ignore.
They can be removed for cleanup, but they are not part of the core logic.

---

## 15. README and LICENSE status

### `README.md`
Should preserve and communicate:
- current structure
- usage
- dataset format
- credits / attribution
- license guidance

### `LICENSE`
Currently preserves upstream license notices for:
- ACT: MIT License
- DETR: Apache License 2.0

Do not remove these files.
Do not strip license or attribution notices during redistribution.

---

## 16. Attribution / original code authors / upstreams

This repository is **not a from-scratch implementation**.
It is a refactored baseline built on top of a customized ACT-based codebase.

### Main customized ACT codebase origin
- **Chemin Ahn**
- Homepage: <https://chemx3937.github.io/>
- GitHub: <https://github.com/Chemx3937>

### Upstream project references
#### ACT — Action Chunking with Transformers
- **Tony Z. Zhao**
- Project page: <https://tonyzhaozh.github.io/aloha/>
- License basis: MIT License

#### DETR
- **Facebook Research**
- GitHub: <https://github.com/facebookresearch/detr>
- License basis: Apache License 2.0

### Attribution rule
Any future patch, external sharing, redistribution, or derivative work should preserve:
- original customized-code attribution to **Chemin Ahn**
- ACT upstream attribution
- DETR upstream attribution
- license notices in the root `LICENSE`

---

## 17. Recommended continuation workflow

When continuing development from this baseline, a good rule is:

- keep `scripts/act/train_act.py` mostly unchanged
- implement research changes inside `source/`
- patch the narrowest correct module first

Examples:
- want contact labels? start with `source/data/dataset.py`
- want force weighting? start with `source/models/policy.py`
- want new head? start with `source/models/act_core.py`
- want validation/logging changes? start with `source/training/engine.py`

---

## 18. Baseline summary

The current `nrs_act` baseline is best understood as:
- a **refactored ACT-based imitation learning repository**
- already reorganized for maintainability
- already capable of starting training
- structured for future **force/contact-aware learning patches**

The most important current files for future work are:
- `source/data/dataset.py`
- `source/data/normalization.py`
- `source/models/policy.py`
- `source/models/act_core.py`
- `source/training/engine.py`

If those files are patched carefully while keeping the current separation of responsibilities, the repository should remain maintainable and extensible.

---

## 19. Quick reference

### Main entry
- `scripts/act/train_act.py`

### Main training loop
- `source/training/engine.py`

### Main dataset logic
- `source/data/dataset.py`

### Main normalization logic
- `source/data/normalization.py`

### Main policy/loss logic
- `source/models/policy.py`

### Main core model logic
- `source/models/act_core.py`

### Keep stable whenever possible
- `scripts/act/train_act.py`
- `README.md`
- `LICENSE`

