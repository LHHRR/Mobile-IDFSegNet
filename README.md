# Mobile-IDFSegNet

Mobile-IDFSegNet is a lightweight plantar infrared thermography (IRT) segmentation network for diabetic foot screening workflows. It performs thermal-only plantar region extraction from camera-exported pseudo-color thermograms without paired visible-light RGB images or task-specific preprocessing.

This repository contains the code for the supervised Mobile-IDFSegNet model described in the latest manuscript version. The public model name and decoder terminology are aligned with the paper:

- **Mobile-IDFSegNet**: Mobile Infrared Diabetic Foot Segmentation Network.
- **DBSD**: Dual-Branch Sampling Decoder, the default decoder head used by Mobile-IDFSegNet.
- **PT-3**: the controlled-access multicenter plantar IRT benchmark dataset.

The repository does **not** include patient data, trained checkpoints, or training logs.

## Highlights

- Thermal-only input from pseudo-color plantar IRT images.
- MobileSAM/TinyViT image encoder with a compact DBSD decoder.
- Training, k-fold training, evaluation, prediction, and throughput benchmarking scripts.
- Defaults and examples aligned with the PT-3 dataset repository structure.
- `.gitignore` excludes datasets, logs, generated outputs, and model weights by default.

## Repository Structure

```text
.
+-- README.md
+-- requirements.txt
+-- train_autoconfig.py          # train/validation split training
+-- train_autoconfig_kfold.py    # k-fold training
+-- test.py                      # checkpoint evaluation
+-- vis_result_paper.py          # prediction and contour overlays
+-- evaluate_old.py              # Dice evaluation used by training
+-- evaluate.py                  # extended metric utilities
+-- sam/
|   +-- SamFeatSeg.py            # Mobile-IDFSegNet wrapper and decoder heads
|   +-- build_sam.py             # model builder
|   +-- mobile_sam.py            # TinyViT/MobileSAM encoder implementation
|   +-- busd.py                  # backward-compatible alias to DBSD
|   +-- build_sam_test_eff.py    # FPS/FLOPs benchmark CLI
+-- utils/
    +-- data_loading.py
    +-- dice_score.py
    +-- utils.py
```

## Installation

Use Python 3.9 or newer. Install PyTorch according to your CUDA version first if needed, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The optional MobileSAM encoder checkpoint is not included. If you use it, place it at:

```text
weights/mobile_sam.pt
```

or pass a custom path with `--encoder-checkpoint`.

## Dataset

PT-3 is distributed separately as a controlled-access dataset:

[LHHRR/Plantar-Thermogram-3-Dataset](https://github.com/LHHRR/Plantar-Thermogram-3-Dataset)

Main PT-3 branch structure:

```text
Thermal_Image_PT3/
Thermal_Image_SegmentationMasks_PT3/
```

External validation branch:

```text
Validation_Image_PT3/
Validation_Mask_PT3/
```

After access is approved, one convenient local layout is:

```text
Mobile-IDFSegNet/
+-- data/
    +-- PT-3/
        +-- Thermal_Image_PT3/
        +-- Thermal_Image_SegmentationMasks_PT3/
```

The `data/` directory is ignored by Git so patient data are not committed with the code.

## Training

Train with a train/validation split:

```bash
python train_autoconfig.py \
  --images-dir data/PT-3/Thermal_Image_PT3 \
  --masks-dir data/PT-3/Thermal_Image_SegmentationMasks_PT3 \
  --encoder-checkpoint weights/mobile_sam.pt \
  --checkpoint-dir outputs/checkpoints_dbsd \
  --epochs 30 \
  --batch-size 4 \
  --learning-rate 0.0005 \
  --classes 2
```

Run five-fold training:

```bash
python train_autoconfig_kfold.py \
  --fold all \
  --images-dir data/PT-3/Thermal_Image_PT3 \
  --masks-dir data/PT-3/Thermal_Image_SegmentationMasks_PT3 \
  --encoder-checkpoint weights/mobile_sam.pt \
  --checkpoint-dir outputs/checkpoints_dbsd_kfold \
  --epochs 30 \
  --batch-size 4 \
  --classes 2
```

Training checkpoints and `training_log.txt` are written under `outputs/` by default and are ignored by Git.
Training-time color augmentation is enabled by default for the training split only; pass `--no-augment` to disable it.

## Evaluation

Evaluate a saved checkpoint:

```bash
python test.py \
  --model outputs/checkpoints_dbsd/checkpoint_epoch30_0.97920.pth \
  --images-dir data/PT-3/Thermal_Image_PT3 \
  --masks-dir data/PT-3/Thermal_Image_SegmentationMasks_PT3 \
  --classes 2
```

Evaluate the external validation branch after cloning it locally:

```bash
python test.py \
  --model outputs/checkpoints_dbsd/checkpoint_epoch30_0.97920.pth \
  --images-dir data/PT-3-validation/Validation_Image_PT3 \
  --masks-dir data/PT-3-validation/Validation_Mask_PT3 \
  --classes 2
```

## Prediction

Generate masks and optional contour overlays:

```bash
python vis_result_paper.py \
  --model outputs/checkpoints_dbsd/checkpoint_epoch30_0.97920.pth \
  --input data/PT-3/Thermal_Image_PT3 \
  --output-dir outputs/predictions \
  --save-overlay \
  --classes 2
```

## Benchmarking

Measure throughput:

```bash
python -m sam.build_sam_test_eff \
  --checkpoint outputs/checkpoints_dbsd/checkpoint_epoch30_0.97920.pth \
  --device cuda \
  --runs 1000
```

Add `--flops` to estimate FLOPs if `ptflops` is installed.

## Data and Log Policy

This repository is intended for code release only.

- Do not commit PT-3 images or masks to this code repository.
- Do not commit training logs, checkpoints, or model weights unless a separate release policy is defined.
- Use the controlled-access PT-3 repository for dataset distribution.
- Keep any clinical data access requests directed to the dataset maintainer.

## Contact

For dataset access or project questions, contact:

**Huarui Liu**  
**liuhuarui@mail.ustc.edu.cn**

## License

A code license has not been added yet. Choose and add an appropriate license before making the repository public.
