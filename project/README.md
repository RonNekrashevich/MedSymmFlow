# Synthetic PneumoniaMNIST augmentation utility

This directory contains a small helper workflow for generating synthetic PneumoniaMNIST-style images with an existing MedSymmFlow checkpoint.

## Download pretrained weights in Google Colab

The official MedSymmFlow pretrained weights are published on the Zenodo record linked from the repository README.

Use the following commands in Google Colab to clone your fork, download the archive, extract it, and inspect the checkpoint files:

```python
!git clone https://github.com/your-username/MedSymmFlow.git
%cd MedSymmFlow
!gdown --fuzzy https://zenodo.org/records/16086025/files/models.zip
!unzip -q models.zip -d models_extracted
!python project/list_checkpoints.py --models_dir models_extracted
```

If you prefer to keep the extracted directory in a different location, replace `models_extracted` with your preferred folder name.

## Windows (PowerShell)

```powershell
cd C:\path\to\MedSymmFlow
python project/generate_pneumoniamnist.py `
  --checkpoint C:\path\to\checkpoint.pt `
  --num_normal 8 `
  --num_pneumonia 8 `
  --seed 42 `
  --beta 4.0 `
  --image_size 256 `
  --solver euler `
  --step_size 0.04
```

## Google Colab

```python
!git clone https://github.com/your-org/MedSymmFlow.git
%cd MedSymmFlow
!pip install -r requirements/requirements.txt
!python project/generate_pneumoniamnist.py \
  --checkpoint /content/your_checkpoint.pt \
  --num_normal 8 \
  --num_pneumonia 8 \
  --seed 42 \
  --beta 4.0 \
  --image_size 256 \
  --solver euler \
  --step_size 0.04
```

## Output structure

- outputs/pneumoniamnist/normal
- outputs/pneumoniamnist/pneumonia
- outputs/pneumoniamnist/metadata.csv

## Visualize generated samples

```powershell
python project/visualize_generated.py --output_dir outputs/pneumoniamnist
```
