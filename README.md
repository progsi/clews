# Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching

_This repository is a fork of *CLEWs* to benchmark the *DVI2* datasets._

## CLEWs reference

J. Serrà, R. O. Araz, D. Bogdanov, & Y. Mitsufuji (2025). Supervised Contrastive Learning from Weakly-Labeled Audio Segments for Musical Version Matching. ArXiv: 2502.16936.

[[`arxiv`](https://arxiv.org/abs/2502.16936)] [[`checkpoints`](https://zenodo.org/records/15045900)]

## Preparation

CLEWS requires python>=3.10. We used python 3.10.13.

You should be able to create the environment by running [install_requirements.sh](install_requirements.sh). However, we recommend to just check inside that file and do it step by step.


## Inference

We provide a basic inference script to extract embeddings using a pre-trained checkpoint:

```bash
OMP_NUM_THREADS=1 python inference.py --checkpoint=logs/model/checkpoint_best.ckpt --path_in=data/audio_files/ --path_out=cache/extracted_embeddings/
```

It will go through all audio files in the folder and subfolders (recursive) and create the same structure in the output folder. Alternatively, you can use the following arguments for processing just a single file:

```bash
OMP_NUM_THREADS=1 python inference.py --checkpoint=logs/model/checkpoint_best.ckpt --fn_in=data/audio_files/filename.mp3 --fn_out=cache/extracted_embeddings/filename.pt
```

## Training and testing

Note: Training and testing assume you have at least one GPU.

### Folder structure

Apart from the structure of this repo, we used the following folders:
* `data`: folder pointing to original audio and metadata files (can be a symbolic link).
* `cache`: folder where to store preprocessed metadata files.
* `logs`: folder where to output checkpoints and tensorboard files.

You should create/organize those folders prior to running any training/testing script. The folders are not necessary for regular operation/inference.

### Preprocessing
#### DVI2 datasets

The repo of `discogs-vi-2` needs to be in the same dir as this repo and the dataset files need to be extracted in its data subdirectory. Then run the script ```make_datas.sh```. 

Then:
```bash
python data_preproc.py --njobs=16 --dataset=DVI2 --path_meta=data/dvi2/ --path_audio=data/audio/ --ext_in=mp4/ --fn_out=cache/metadata-dvi2.pt
python data_preproc.py --njobs=16 --dataset=DVI2 --path_meta=data/dvi2fm_light/ --path_audio=data/audio/ --ext_in=mp4/ --fn_out=cache/metadata-dvi2fm_light.pt
```

This script takes time as it reads/checks every audio file (so that you do not need to run checks while training or in your dataloader). You just do this once and save the corresponding metadata file. Depending on the path names/organization of your data set it is possible that you have to modify some minor portions of the `data_preproc.py` script.

### Training

Before every training run, you need to clean the logs path and copy the configuration file (with the specific name `configuration.yaml`):
```bash
rm -rf logs/dvi2-clews/ ; mkdir logs/dvi2-clews/ ; cp config/dvi2-clews.yaml logs/dvi2-clews/configuration.yaml
rm -rf logs/dvi2fm_light-clews/ ; mkdir logs/dvi2fm_light-clews/ ; cp config/dvi2fm_light-clews.yaml logs/dvi2fm_light-clews/configuration.yaml
```

Next, launch the training script using, for instance:

```bash
python train.py jobname=dvi2-clews conf=config/dvi2-clews.yaml nnodes=1 ngpus=2
python train.py jobname=dvi2fm_light-clews conf=config/dvi2fm_light-clews.yaml nnodes=1 ngpus=2
```

### Testing

To launch the testing script, you can run, for instance:

```bash
python test.py jobname=test-script checkpoint=logs/dvi2-clews/checkpoint_best.ckpt nnodes=1 ngpus=4 redux=bpwr-10
python test.py jobname=test-script checkpoint=logs/dvi2fm_light-clews/checkpoint_best.ckpt nnodes=1 ngpus=4 redux=bpwr-10 maxlen=300
```

#### Cross-Domain Testing
We recommend to run overall tests first. Then, set `domain` and `domain_mode` or `domain` and `qsdomain`, `csdomain` respectively. In the script `test_conceptwise.sh` we show an example.


## License

The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).

## Notes

* If using this code, parts of it, or developments from it, please cite the reference above.

