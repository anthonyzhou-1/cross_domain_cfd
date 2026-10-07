# Cross-Domain Pretraining for Steady-State CFD Surrogates

Anthony Zhou, Amir Barati Farimani, Shirley Ho, Rudy Morel. 

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

## Data

## Downloading from Source

You can either download data from the source (listed below), or download preprocessed versions of the data that we release. The source for each dataset is:
- [AhmedMl](https://arxiv.org/abs/2407.20801)
- [DrivAerML](https://arxiv.org/abs/2408.11969)
- [WindsorML](https://arxiv.org/pdf/2407.19320)
- [HiLiftAeroML](https://arxiv.org/abs/2605.19565)
- [DrivAerNet++](https://arxiv.org/abs/2406.09624)
- [Emmi_Wing](https://arxiv.org/abs/2511.21474)
- [SuperWing](https://arxiv.org/abs/2512.14397)
- [Double-Delta](https://arxiv.org/abs/2512.20941)
- [SHIFT-CCA](https://huggingface.co/datasets/luminary-shift/CCA-sample)
- [SHIFT-Submarine](https://huggingface.co/datasets/luminary-shift/Submarine-sample)

If you do use this data, please cite the original works! We did not generate any of this data.

If downloading raw data from the source, we also provide scripts for preprocessing this data and for computing normalization stats:

```bash
python -m data_processing.<dataset> surface --root $CFD_DATA_ROOT/<dataset>  
python -m data_processing.norm_stats --root $CFD_DATA_ROOT
```

### Downloading Preprocessed Data
Processing data from the source can time-consuming, therefore we also provide preprocessed datasets. There are two versions of the preprocessed data, either a downsampled version hosted on Huggingface (for easy distribution and experimentation), or the full version hosted on Globus. The key features of the preprocessing are:

- Converted from raw simulation files (.vtk, etc.) into per-sample .npy files. Numpy files are stored as a memmap, which allows slices to be read from the file without loading the entire file. 
- Surface/volume fields are randomly permuted before storage. Therefore, contiguous slices of the .npy file are random samples of the point clouds. 
- Combined, this allows for very fast dataloading, by avoiding the need to open million-point meshes (an in general, we only have the memory to train on ~100k points at once). 
- General cleanup of any numerical artifacts, and aligning sign/axis conventions across datasets.

In general, this allows fast data loading and saves on storage compared to the raw simulation files, but there are a few drawbacks:

- Meshes are discarded. Cell connectivity is a large portion of the raw surface/volume data and isn't used during training (except for GNNs perhaps). Unfortunately, this discards information like cell areas/normals that may be useful for geometric deep learning or computing surface integrals for drag/lift. However, the surface quadratures can be approximated quite well from the surface point cloud due to its high density and we observe that we can still back out coefficients of lift/drag through approximate quadratures.
- CAD files (.stl. etc.) are also discarded.
- If using the Huggingface, downsampled datasets, there is a chance that this discards some features in high-resolution, scale-resolving simulations. The time-averaging that is done to produce steady-state fields may already smooth these signals, and it isn't clear if neural surrogates can even predict features at such a high resolution, but this is a potential limitation of training on coarsened data. The Globus data is kept at the original resolution as best as we could. 

### Huggingface [Link to Data](https://hf.co/collections/ayz2/cross-domain-cfd)

Huggingface stores a downsampled version where each dataset does not exceed 1TB and is around 5TB in total. Some datasets were small enough (less than 1TB), and were not downsampled. The details for each dataset are:

| Dataset      | Samples | Surface factor  | Volume factor | Surface size (TB) | Volume size (TB) | Total release (TB) | Source size (TB) |
| :----------- | -------------------------: | ----------------: | ------------: | -----------: | ----------: | ------------: | ----------: |
| ahmedml      | 500                  | ÷1                | ÷1            | 0.019        | 0.257       | **0.28**      | 0.28        |
| windsorml    | 350                  | ÷1                | ÷3            | 0.029        | 0.692       | **0.72**      | 2.1         |
| drivaerml    | 483                  | ÷3                | ÷3            | 0.039        | 0.713       | **0.75**      | 2.3         |
| drivaernet   | 8,128              | ÷1                | ÷5            | 0.102        | 0.660       | **0.76**      | 3.4         |
| emmi_wing    | 29,609           | ÷1                | ÷5            | 0.284        | 0.544       | **0.83**      | 4.6         |
| superwing    | 28,856           | ÷1                | ÷3            | 0.034        | 0.670       | **0.70**      | 2.6         |
| double_delta | 2,448              | ÷1                | ÷1            | 0.007        | 0.219       | **0.23**      | 0.23        |
| hiliftaeroml | 1,787              | ÷12 | ÷10           | 0.294        | 0.582       | **0.88**      | 10.4        |
| **Total**    | **71,161**        |                   |               | 0.81         | 4.34        | **5.15**      | 25.9        |

Emmi Wing and Superwing datasets are stored as .tar shards (to comply with maximum file quotas on Huggingface). These need to be unzipped before using. The source size for HiLiftAeroML was already downsampled by 2x to save space on our cluster. Otherwise all data was kept at the original resolution as the source. 

### Globus [Link to Data](https://app.globus.org/file-manager?origin_id=52794d38-2c8f-4ded-91df-31d5aeec5e83&origin_path=%2F)
For the full resolution, the entire preprocessed dataset (around 26TB in total) can be downloaded from Globus. 

All of these datasets are ready to use with the dataloader and dataset.
For SHIFT-CCA and SHIFT-Submarine, these are privately maintained so we do not release these. However, these can be downloaded from [Huggingface](https://huggingface.co/luminary-shift/datasets) by sending a request to Luminary, and processed using the provided scripts. 

### Data Configs
After downloading the data, the data root must be set. Configs use dataset directories relative this data root, given by `--data_root` (this a command line argument), `data.data_root` (this can be set in the config .yaml files), or `$CFD_DATA_ROOT` (set as an environment variable):

```
$CFD_DATA_ROOT/<dataset>/collated/         # surface samples + splits + norm_stats_*.npz
$CFD_DATA_ROOT/<dataset>/volume_collated/  # volume samples + norm_stats_volume*.npz
```

## Pretrained Model Checkpoints

Pretrained model checkpoints for all experiments are available on [Huggingface](https://huggingface.co/ayz2/cross_domain_cfd_models). The joint models ({SMART/AB-UPT/Transolver}/joint/) may be of interest, which can serve as pretrained checkpoints that can be fine-tuned to new CFD datasets. 

## Run

The code expects a logging directory to save logs and checkpoints (currently at `./logs/<run>/`). Furthermore, the code expects an authenticated wandb instance for logging. 

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
