# cross_domain_cfd

Training, finetuning and evaluation code for cross-domain neural surrogates of 3D CFD
(surface + volume fields). Models: SMART (`smart`), in-context SMART (`smart_ic`),
Transolver++ (`transolver`) and AB-UPT (`upt`).

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

## Data

Configs use dataset dirs relative to a data root, given by `--data_root`, `data.data_root`
in the config, or `$CFD_DATA_ROOT`:

```
$CFD_DATA_ROOT/<dataset>/collated/         # surface samples + splits + norm_stats_*.npz
$CFD_DATA_ROOT/<dataset>/volume_collated/  # volume samples + norm_stats_volume*.npz
```

Build a dataset from its raw download, then compute normalization stats:

```bash
python -m data_processing.<dataset> surface --root $CFD_DATA_ROOT/<dataset>   # then: volume, prune
python -m data_processing.norm_stats {base,surface,volume,twin} --root $CFD_DATA_ROOT
```

## Preprocessed Datasets
Preprocessed datasets can be downloaded from [Huggingface](https://hf.co/collections/ayz2/cross-domain-cfd). Huggingface stores a downsampled version where each dataset does not exceed 1TB and is around 5TB in total, for easy distribution and experimentation. For the full resolution, the entire preprocessed dataset (around 26TB in total) can be downloaded from Globus. 

These are ready to use with the dataloader and dataset, otherwise the raw data will need to be processed. 

## Pretrained Model Checkpoints

Pretrained model checkpoints for all experiments are available on [Huggingface](https://huggingface.co/ayz2/cross_domain_cfd_models). The joint models ({SMART/AB-UPT/Transolver}/joint/) may be of interest, which can serve as pretrained checkpoints that can be fine-tuned to new CFD datasets. 

## Run

Run from the repo root. Logs and checkpoints go to `./logs/<run>/`; pass `--wandb_mode offline`
or `disabled` to run without a wandb account.

```bash
export CFD_DATA_ROOT=/path/to/data # Or set data.data_root in the config yaml files. 

# Pretraining (single dataset or joint); configs/{abupt,transolver,global_norm} work the same way
python train.py --config configs/smart/drivaernet.yaml
python train.py --config configs/smart/joint.yaml

# Toy in-context experiments: one run per config in configs/toy_examples (*_vanilla, *_ic, *_oracle)
python train.py --config configs/toy_examples/dd_aoa_ic.yaml

# Validation
python val.py --config configs/smart/drivaernet.yaml --checkpoint logs/<run>/last.ckpt

# Few-shot finetuning on a held-out dataset (ahmedml, submarine, shift_cca, hiliftaeroml)
for S in 2 4 8 16 32 64; do
  python finetune.py --config configs/finetuning/cond_smart.yaml --checkpoint logs/<joint_run>/last.ckpt \
    --source_name joint --target submarine --num_samples $S --max_steps 400 --val_every_steps 40
done

Use `--checkpoint scratch` for a randomly initialized baseline, and the matching `configs/finetuning/cond_*.yaml`
for each pretrained model family (`cond_abupt`, `cond_transolver`, `cond_smart_physnorm`, ...).
Common overrides: `--devices`, `--seed`, `--max_steps`, `--data_root`; see `python train.py --help`.
