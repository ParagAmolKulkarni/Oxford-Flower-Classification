```markdown

To comprehend the structure of a public dataset, load and examine it (e.g., COCO, Oxford-102 Flowers). Analyze dataset statistics such as the number of classes, description length, and image resolution, and explore and display text descriptions combined with photos.

# Oxford-102 Flowers Classification 🌸

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A deep learning solution for classifying 102 flower species using **MobileNetV3** with transfer learning. Achieves **77-78% accuracy** in under 5 minutes/epoch on GPU.

<img src="sample_predictions.png" width=600 alt="Sample Predictions">

## Features

- **Transfer Learning**: Uses pretrained MobileNetV3-small
- **Data Augmentation**: Random crops, flips, and normalization
- **Evaluation Metrics**:
  - Classification report (precision/recall/F1)
  - Confusion matrix visualization
  - Training progress tracking
- **Auto-Generated Outputs**: Predictions CSV, visualizations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib
- pandas
- scikit-learn
- tqdm

```bash
pip install -r requirements.txt
```

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/flowers-classification.git
cd flowers-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The Oxford-102 Flowers dataset will auto-download to `data/` on first run.  
Class labels are auto-generated to `flower_classes.txt` if missing.

## Usage

```bash
python flower_classifier_final.py
```

**Optional Arguments**:
```bash
--epochs 20           # Number of training epochs
--batch_size 64       # Batch size (reduce if OOM errors occur)
--img_size 224        # Input image resolution
```

## Results

**Output Files**:
- `best_model.pth` : Trained model weights
- `confusion_matrix.png` : Class-wise error analysis
- `training_progress.png` : Accuracy vs epochs
- `sample_predictions.png` : 9 test samples with predictions
- `predictions.csv` : Full prediction results

**Example Output**:
```
Epoch 1/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:26<00:00,  1.64s/it]

Epoch 1 Results:
Accuracy: 2.24%
--------------------------------------------------
Epoch 2/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.49s/it]

Epoch 2 Results:
Accuracy: 4.99%
--------------------------------------------------
Epoch 3/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.50s/it]

Epoch 3 Results:
Accuracy: 10.07%
--------------------------------------------------
Epoch 4/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.51s/it]

Epoch 4 Results:
Accuracy: 17.65%
--------------------------------------------------
Epoch 5/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.52s/it]

Epoch 5 Results:
Accuracy: 26.75%
--------------------------------------------------
Epoch 6/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:25<00:00,  1.59s/it]

Epoch 6 Results:
Accuracy: 34.66%
--------------------------------------------------
Epoch 7/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.53s/it]

Epoch 7 Results:
Accuracy: 41.75%
--------------------------------------------------
Epoch 8/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.54s/it]

Epoch 8 Results:
Accuracy: 47.26%
--------------------------------------------------
Epoch 9/20: 100%|██████████████████████████████████████████████████████████████████████| 16/16 [00:25<00:00,  1.57s/it]

Epoch 9 Results:
Accuracy: 52.63%
--------------------------------------------------
Epoch 10/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.56s/it]

Epoch 10 Results:
Accuracy: 56.66%
--------------------------------------------------
Epoch 11/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:25<00:00,  1.61s/it]

Epoch 11 Results:
Accuracy: 60.37%
--------------------------------------------------
Epoch 12/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.54s/it]

Epoch 12 Results:
Accuracy: 63.25%
--------------------------------------------------
Epoch 13/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.47s/it]

Epoch 13 Results:
Accuracy: 65.39%
--------------------------------------------------
Epoch 14/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.47s/it]

Epoch 14 Results:
Accuracy: 67.86%
--------------------------------------------------
Epoch 15/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.47s/it]

Epoch 15 Results:
Accuracy: 69.85%
--------------------------------------------------
Epoch 16/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.46s/it]

Epoch 16 Results:
Accuracy: 71.51%
--------------------------------------------------
Epoch 17/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:25<00:00,  1.60s/it]

Epoch 17 Results:
Accuracy: 72.37%
--------------------------------------------------
Epoch 18/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.49s/it]

Epoch 18 Results:
Accuracy: 74.35%
--------------------------------------------------
Epoch 19/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:24<00:00,  1.50s/it]

Epoch 19 Results:
Accuracy: 75.61%
--------------------------------------------------
Epoch 20/20: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:23<00:00,  1.48s/it]

Epoch 20 Results:
Accuracy: 76.00%
--------------------------------------------------

Classification Report:
              precision    recall  f1-score   support

   class_001       0.33      0.85      0.48        20
   class_002       1.00      0.93      0.96        40
   class_003       0.67      0.30      0.41        20
   class_004       0.41      0.36      0.38        36
   class_005       0.66      0.73      0.69        45
   class_006       0.73      0.88      0.80        25
   class_007       0.47      0.80      0.59        20
   class_008       0.85      0.98      0.91        65
   class_009       0.54      0.81      0.65        26
   class_010       0.92      0.96      0.94        25
   class_011       0.58      0.39      0.46        67
   class_012       0.92      0.87      0.89        67
   class_013       0.79      0.93      0.86        29
   class_014       0.64      0.96      0.77        28
   class_015       0.93      0.97      0.95        29
   class_016       0.50      0.86      0.63        21
   class_017       0.94      0.91      0.92        65
   class_018       0.88      0.68      0.76        62
   class_019       0.54      0.69      0.61        29
   class_020       0.79      0.72      0.75        36
   class_021       0.55      0.80      0.65        20
   class_022       0.71      0.82      0.76        39
   class_023       0.95      0.83      0.89        71
   class_024       0.72      0.82      0.77        22
   class_025       0.70      0.90      0.79        21
   class_026       0.55      0.86      0.67        21
   class_027       0.90      0.90      0.90        20
   class_028       0.67      0.93      0.78        46
   class_029       0.90      0.76      0.82        58
   class_030       0.51      0.66      0.57        65
   class_031       0.53      0.62      0.57        32
   class_032       0.29      0.40      0.34        25
   class_033       0.67      0.92      0.77        26
   class_034       0.71      0.85      0.77        20
   class_035       0.77      1.00      0.87        23
   class_036       0.72      0.60      0.65        55
   class_037       0.93      0.99      0.96        88
   class_038       0.65      0.67      0.66        36
   class_039       0.28      0.52      0.37        21
   class_040       0.76      0.62      0.68        47
   class_041       0.93      0.80      0.86       107
   class_042       0.62      0.87      0.72        39
   class_043       0.91      0.45      0.60       110
   class_044       0.55      0.92      0.69        73
   class_045       0.50      0.90      0.64        20
   class_046       0.76      0.97      0.85       176
   class_047       0.94      1.00      0.97        47
   class_048       0.74      0.90      0.81        51
   class_049       0.83      1.00      0.91        29
   class_050       0.92      0.93      0.92        72
   class_051       0.82      0.47      0.60       238
   class_052       0.85      0.89      0.87        65
   class_053       0.65      0.48      0.55        73
   class_054       0.93      0.95      0.94        41
   class_055       0.73      0.84      0.78        51
   class_056       0.85      0.93      0.89        89
   class_057       0.85      0.94      0.89        47
   class_058       0.65      0.97      0.78        94
   class_059       0.85      0.96      0.90        47
   class_060       0.83      0.99      0.90        89
   class_061       0.83      1.00      0.91        30
   class_062       0.62      0.69      0.65        35
   class_063       0.97      1.00      0.99        34
   class_064       0.89      0.97      0.93        32
   class_065       0.96      0.88      0.92        82
   class_066       0.95      0.95      0.95        41
   class_067       0.54      0.68      0.60        22
   class_068       0.81      0.65      0.72        34
   class_069       0.78      0.91      0.84        34
   class_070       0.89      0.93      0.91        42
   class_071       0.97      0.98      0.97        58
   class_072       0.49      0.66      0.56        76
   class_073       0.92      0.75      0.83       174
   class_074       0.86      0.58      0.69       151
   class_075       0.82      0.83      0.83       100
   class_076       0.91      0.72      0.81        87
   class_077       0.98      0.88      0.93       231
   class_078       0.73      0.68      0.70       117
   class_079       0.72      0.86      0.78        21
   class_080       0.77      0.85      0.80        85
   class_081       0.89      0.99      0.94       146
   class_082       0.71      0.66      0.69        92
   class_083       0.80      0.48      0.60       111
   class_084       0.56      0.52      0.54        66
   class_085       0.49      0.91      0.63        43
   class_086       0.61      0.89      0.72        38
   class_087       0.54      0.91      0.68        43
   class_088       0.77      0.40      0.52       134
   class_089       0.79      0.71      0.75       164
   class_090       0.67      0.48      0.56        62
   class_091       0.57      0.79      0.66        56
   class_092       0.77      0.87      0.82        46
   class_093       0.65      0.50      0.57        26
   class_094       0.92      0.80      0.85       142
   class_095       0.86      0.67      0.75       108
   class_096       0.59      0.27      0.37        71
   class_097       0.68      0.46      0.55        46
   class_098       0.72      0.50      0.59        62
   class_099       0.79      0.88      0.84        43
   class_100       0.74      1.00      0.85        29
   class_101       0.55      0.55      0.55        38
   class_102       0.81      0.89      0.85        28

    accuracy                           0.76      6149
   macro avg       0.74      0.78      0.74      6149
weighted avg       0.78      0.76      0.76      6149


Output Files Generated:
- confusion_matrix.png
- training_progress.png
- sample_predictions.png
- predictions.csv
- best_model.pth

```

## Model Architecture

**Tech Stack**:
- Backbone: MobileNetV3-small (pretrained on ImageNet)
- Classifier: Custom linear layer (102 classes)
- Optimizer: RMSprop with StepLR scheduling

**Training**:
- Input Size: 224x224 RGB
- Augmentation: Random crops/horizontal flips
- Loss: Cross Entropy

**Output**:
-It will be generated separately and directly in folder named "confusion_matrix.png", "sample perdiction.png", "training_progress.png", also "prediction.csv"

## Acknowledgments

- Dataset: [Oxford-102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- PyTorch: [Official Documentation](https://pytorch.org/docs/stable/index.html)
```
