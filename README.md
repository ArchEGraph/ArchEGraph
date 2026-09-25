<p align="center">
	<img src="asset/ArchEGraph_logo.svg" alt="ArchEGraph logo" width="420" />
</p>

![Figure 1. ArchEGraph dataset overview](asset/ArchEGraph_abstract.png)



<p align="center">
	<a href="https://github.com/ArchEGraph/ArchEGraph" style="display:inline-block;">
		<img src="https://img.shields.io/badge/GitHub-ArchEGraph-C4B5FD?label=Code&logo=github&logoColor=000" style="max-width: 100%;" alt="GitHub repository" />
	</a>
	<a href="https://huggingface.co/datasets/ArchEGraph/ArchEGraph" style="display:inline-block;">
		<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-C4B5FD?label=Dataset" style="max-width: 100%;" alt="Hugging Face dataset" />
	</a>
</p>

# ArchEGraph

This is the official codebase for the paper:  
**ArchEGraph: A Large-Scale Graph Dataset for Geometry–Topology–Physics Aligned Building Energy Modeling** .

Minimal and reproducible training code for ArchEGraph dataset experiments.

![Figure 2. Batch visualization of ArchEGraph dataset](asset/pack_cases_graph_zone_overlay_8_20260502_082523.gif)

## Overview

- mesh2graph: reconstruct topology-related structure from geometry and building features.
- graph2energy: predict energy from graph and weather features.
- Unified CLI entry at main.py.
- Reproducibility controls (seed, deterministic mode).

<p align="center">
	<img src="asset/ArchEGraph_task.png" alt="Figure 3. ArchEGraph task overview" width=300 />
</p>

## Repository layout

```text
ArchEGraph/
	main.py
	requirements.txt
	asset/
	configs/
		mesh2graph.minimal.json
		graph2energy.minimal.json
	scripts/
		run_mesh2graph.sh
		run_graph2energy.sh
		reproduce_minimal.sh
	mesh2graph/
	graph2energy/
	cache/                  # generated artifacts (ignored by git)
```

## Environment

Python 3.10+ is recommended. CUDA is optional but recommended for full-scale training.

```bash
pip install -r requirements.txt
```

If your machine needs a specific PyTorch/CUDA build, install PyTorch first from the official index, then run the command above.

## Dataset setup from HuggingFace

The dataset is not included in this repository.

1. Download dataset from HuggingFace: https://huggingface.co/datasets/ArchEGraph/ArchEGraph
2. The verified default HuggingFace layout (after `huggingface-cli download ... --local-dir ./data`) is:

Expected layout for direct training:

```text
data/
	manifest.csv
	building/*.npz
	geometry/*.npz
	weather/*.npz
	energy/**/*.npz
	split/
		split_m.csv				# Get the ArchEGraph-M dataset with this split 
		split_p.csv				# Get the ArchEGraph-P dataset with this split
		split_demo.csv 			# Minimal split for quick start
		split_building_bias.csv # Building-OOD: disjoint P train/val and M test buildings
		split_weather_bias.csv  # Weather-location-OOD: disjoint 14/3/3 weather IDs
		split_metadata.json     # Exact protocol definitions, cohorts, and overlap checks
```

This layout is consistent with the current HuggingFace dataset tree (`building`, `energy`, `geometry`, `weather`, `split`, `manifest.csv` at the dataset root).

You can download with HuggingFace CLI:

```bash
pip install hf
hf auth login

# Download main dataset to ArchEGraph/data (default layout)
hf download ArchEGraph/ArchEGraph --repo-type dataset --local-dir ./data
```

Important: `ArchEGraph` is very large (Currently about 34.2GB on HuggingFace).

If you want a fast quick-start, use the demo dataset:

- https://huggingface.co/datasets/ArchEGraph/ArchEGraph-demo
- size is currently about 234MB
- it is extracted based on `split_demo`

Quick-start commands for demo:

```bash
# 1) Download the small demo data package
hf download ArchEGraph/ArchEGraph-demo --repo-type dataset --local-dir ./data_demo

# 2) Download split_demo.csv only (small file) from the main dataset
hf download ArchEGraph/ArchEGraph --repo-type dataset --include "split/split_demo.csv" --local-dir ./data_demo
```

## Quick start

After placing data under `data`, run:

```bash
bash scripts/run_mesh2graph.sh --data_dir ./data/ --split split_m_mesh

bash scripts/run_graph2energy.sh --data_dir ./data/ --split split_p
```

For graph2mesh (mesh2graph) runs, use split files with the `_mesh` suffix.

Examples:

```bash
# split name
bash scripts/run_mesh2graph.sh --data_dir ./data/ --split split_demo_mesh

# or csv file name/path
bash scripts/run_mesh2graph.sh --data_dir ./data/ --split split/split_demo_mesh.csv
```

Run both sequentially:

```bash
bash scripts/reproduce_minimal.sh
```

## Topology ablation

The paired topology ablation compares the original face-to-space incidence with a deterministic shuffled condition. Shuffling is performed independently within each building and preserves source faces, edge attributes, edge count, and the number of incident faces per space. The same shuffled building topology is reused across weather cases.

Run one paired experiment on the P split:

```bash
python scripts/run_topology_ablation.py \
	--data_dir ../data/ArchEGraph \
	--split split_p \
	--model F2SAttr \
	--seeds 42 43 44 \
	--topology_seed 42 \
	--epochs 100
```

Each training seed runs both `original` and `shuffled` conditions. The fixed topology seed isolates training variation from the randomized topology realization. Per-run artifacts are written under `cache/<run_name>/`; aggregate means, standard deviations, and paired shuffled-minus-original differences are written to `cache/<experiment_name>/summary.json` and `runs.csv`.

Use `--dry_run` to inspect commands without training. Repeat with `--split split_m` to measure the same effect on the manually designed subset.

## Classical Graph2Energy baselines

Three lightweight non-message-passing baselines use hourly weather and per-space aggregate geometry features:

- `DegreeHourRidge`: heating/cooling degree-hours with weather, time, and geometry interactions.
- `GeometryHistGBR`: histogram gradient-boosted trees with weather, time, and geometry summaries.
- `RCInspiredRidge`: ridge regression with causal 6, 24, and 72 hour exponential response features that approximate thermal inertia.

The RC-inspired model is a data-calibrated surrogate, not a physical RC network with known material resistance and capacitance. Aggregate geometry includes space centroids and statistics of incident faces; none of these baselines performs graph message passing.

Run all three baselines on an explicit split:

```bash
python scripts/run_classical_baselines.py \
	--data-dir ../data/ArchEGraph \
	--split split_p \
	--max-train-samples 250000
```

Training rows are sampled deterministically and balanced across training cases. Validation and test metrics use every zone-hour by default; `--max-eval-samples-per-case` can cap them for a smoke test. Results include normalized and original-scale MAE, MSE, RMSE, and R2. Each model writes `model.joblib`, `config.json`, and `metrics.json` under `cache/classical_baselines/<split>/<model>/`, with a combined `summary.csv` in the split directory.

Override any option from CLI, for example:

```bash
python main.py --task graph2energy --config configs/graph2energy.minimal.json --data_dir ./data/ --split split_p
```

`--split` accepts split name (`split_m`, `split_p`, `split_m_mesh`), file name (`split_m.csv`, `split_m_mesh.csv`), or a direct split CSV path.
For graph2mesh (mesh2graph), use the `_mesh` split variants (for example `split_demo_mesh`).
`--run_name` can be used to set the cache folder name manually; when omitted, the run folder defaults to `M2G_YYYYMMDD_HHMMSS` (mesh2graph) or `G2E_YYYYMMDD_HHMMSS` (graph2energy).

### Evaluation split semantics

- `split_p.csv` and `split_m.csv` are IID case-level splits. Cases are unique, but building and weather IDs may overlap across partitions.
- `split_building_bias.csv` holds weather fixed to Chicago and uses disjoint buildings: P for train/validation and M for test.
- `split_weather_bias.csv` uses the same 150 buildings in every partition and globally disjoint weather location IDs: 14 train, 3 validation, and 3 test.

The weather protocol is location-ID-disjoint, not ASHRAE-climate-zone-disjoint. The dataset release includes the exact cohorts and assertions in `split/split_metadata.json`.

To regenerate and validate the OOD split files from a downloaded dataset root:

```bash
python scripts/generate_evaluation_splits.py --data-root ./data
python scripts/generate_evaluation_splits.py --data-root ./data --check
```


## Outputs

- cache/<run_name>/
	- model.pth
	- metrics.json
	- config.json

