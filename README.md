# RadMultiBench Model Reproduction

This repository contains the code to reproduce all experiments for the paper:  
**"RadMultiBench: A Multimodal Benchmark Dataset for Diagnostic Radiology"**

The code is modular and config-driven. All results in the paper are reproducible using these instructions and the released configuration files.  
> **Note:** The dataset will be made public upon paper acceptance. For review, please contact the authors for private access if needed.

***

## 1. Setup

### Prerequisites
- Python 3.11+
- PyTorch (>= 1.10, as used in the original experiments)
- NVIDIA GPU (T4 recommended; experiments use this hardware for benchmarking)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NahinAlam001/RadMultiBench_Reproduce.git
   cd RadMultiBench_Reproduce
   ```

2. **Install all required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   (Includes necessary dependencies like `hi-ml-multimodal` and `clip`.)

3. **Obtain the Dataset:**
   - For private review access, download the dataset directly using `gdown`:
     ```bash
     pip install gdown
     gdown 'https://docs.google.com/uc?export=download&id=1nZyV1NPTcEV6zN_MZPiWGzLujBqI_3pH'
     unzip data.zip
     rm data.zip
     mv data data
     ```
   - Once extracted, ensure the directory structure is as follows:
     ```
     data/
         clean_map.csv
         images/
            [all images...]
     ```

4. **Check and update config paths (if needed):**
   In `configs/*.yaml`:
   ```yaml
   DATA:
     IMG_DIR: "./data/images"
     CSV_PATH: "./data/clean_map.csv"
   ```

***

## 2. Training and Evaluation

**Main script:** `train.py` (handles both classification and generation/multimodal tasks).

### How to Run
Always specify a config file with `--config-file`.  
Example commands:

**Classification – ResNet-LSTM:**
```bash
python train.py --config-file configs/model_resnet_lstm_class.yaml
```

**Generation – ResNet-LSTM:**
```bash
python train.py --config-file configs/model_resnet_lstm_gen.yaml
```

(Other configs available in `configs/` for CLIP-GPT and BioViL-T models.)

**Evaluation:**  
By default, metrics and predictions are saved to `.csv` or `.json` in the `output/` folder.

***

## 3. Reproducibility Checklist

- [x] Same dataset splits as in the paper  
- [x] All model parameters/configs included  
- [x] Hardware used and dependencies clearly specified  
- [x] Scripted training and evaluation pipeline  
- [x] Code is self-contained (as soon as data becomes public)

***

## 4. Frequently Asked Questions

- **Q: Why can't I access the dataset?**  
  *A: The dataset is under embargo until publication. Contact the authors for review access, or check back after acceptance.*

- **Q: Can I use my own medical images?**  
  *A: Yes, as long as images follow the format expected by `clean_map.csv` and are placed in the `/data/images/` directory.*

- **Q: Where can I find configuration details for each experiment?**  
  *A: See the `configs/` directory for YAML files replicating each experimental setup from the paper.*

***
