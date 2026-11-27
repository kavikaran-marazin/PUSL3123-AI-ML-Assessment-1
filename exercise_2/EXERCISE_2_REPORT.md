# Experimental Report: Exercise 2 – Convolutional Neural Network Classifier

**Module:** PUSL3123 – AI and Machine Learning  
**Assessment:** Assessment 1 (Set Exercises – 50% Weighting)  
**Exercise:** Exercise 2 – Convolutional Neural Network Classifier (10 Marks)  
**Dataset:** CIFAR-10 (`torchvision.datasets.CIFAR10`)  
**Document Type:** Formal Experimental Report and Traceability Record  

---

## 1. Executive Summary

The objective of Exercise 2 is to replace the Multi-Layer Perceptron (MLP) with a Convolutional Neural Network (CNN) containing exactly three trainable layers, and critically investigate the influence of convolutional operations on image classification performance.

Using the CIFAR-10 dataset under strictly controlled, identical training settings (Adam optimizer, learning rate $\eta = 0.001$, batch size 64, 15 epochs, identical 45,000/5,000 train-validation partition, and held-out 10,000 test set), both models were trained and comprehensively evaluated.

### Key Experimental Findings:
1. **Superior Classification Accuracy:** The 3-layer CNN achieved a **Test Accuracy of 68.42%**, outperforming the 3-layer baseline MLP (**51.30%**) by an absolute margin of **+17.12%**.
2. **Drastic Parameter Reduction (32x Efficiency):** The CNN utilized only **25,578 trainable parameters**, compared to **820,874 trainable parameters** in the baseline MLP—representing a **96.88% reduction in parameter complexity**.
3. **Overfitting Resistance and Generalisation:** The baseline MLP exhibited pronounced overfitting: while training accuracy climbed to 74.14%, validation accuracy plateaued at 52.00% and validation loss diverged sharply from 1.3785 to 1.7610. Conversely, the CNN demonstrated robust generalisation, maintaining close tracking between training loss (0.6335) and validation loss (0.9543).

---

## 2. Assessment Requirements and Traceability

| ID | Official Brief Requirement | Implementation Evidence | Verification Status |
|---|---|---|---|
| **REQ-1** | Develop a CNN with three trainable layers. Pooling and activation operations may be included as required. | Implemented in `model.py` (`ThreeLayerCNN`): two 2D convolutional layers and one fully connected layer. | Verified (3 weight layers) |
| **REQ-2** | Train and evaluate the CNN using the same dataset partitions as Exercise 1. | Exactly 45,000 train / 5,000 validation images partitioned via seeded `random_split`; 10,000 test images preserved. | Verified (Identical split) |
| **REQ-3** | Use comparable training settings wherever practical. | Adam ($\beta_1=0.9, \beta_2=0.999$, lr=0.001), CrossEntropyLoss, batch size 64, 15 epochs. | Verified (Identical settings) |
| **REQ-4** | Compare MLP and CNN using classification accuracy, training and validation loss, and model parameter counts. | Full epoch-by-epoch tracking, confusion/performance metrics, parameter counting, and side-by-side tables. | Verified (Comparative metrics recorded) |
| **REQ-5** | Discuss differences in performance and explain how convolutional operations influence image feature learning. | In-depth theoretical and empirical analysis of spatial hierarchy, weight sharing, and receptive fields. | Verified (Report Section 6) |

---

## 3. Architectural Design and Parameter Mathematical Derivation

A **trainable layer** is formally defined as a parameterized layer whose tensor weights and biases receive gradient updates during optimization. Activation functions (`ReLU`) and spatial pooling operators (`MaxPool2d`) perform fixed mathematical operations with zero learnable parameters.

### 3.1 Three-Layer CNN Architecture (`ThreeLayerCNN`)

$$\text{Input: } \mathbf{X} \in \mathbb{R}^{3 \times 32 \times 32}$$

1. **Trainable Layer 1 — 2D Convolution (`conv1`):**
   - *Specification:* `in_channels=3`, `out_channels=16`, `kernel_size=3`, `stride=1`, `padding=1`
   - *Weight Tensor:* $[16, 3, 3, 3] \implies 16 \times 3 \times 3 \times 3 = 432 \text{ weights}$
   - *Bias Tensor:* $[16] \implies 16 \text{ biases}$
   - *Layer Parameters:* $432 + 16 = \mathbf{448}$
   - *Followed by:* Non-linear activation $\text{ReLU}(\cdot)$ and spatial downsampling $\text{MaxPool2d}(2, 2)$, halving spatial dimensions:
     $$\mathbf{H}_1 \in \mathbb{R}^{16 \times 16 \times 16}$$

2. **Trainable Layer 2 — 2D Convolution (`conv2`):**
   - *Specification:* `in_channels=16`, `out_channels=32`, `kernel_size=3`, `stride=1`, `padding=1`
   - *Weight Tensor:* $[32, 16, 3, 3] \implies 32 \times 16 \times 3 \times 3 = 4,608 \text{ weights}$
   - *Bias Tensor:* $[32] \implies 32 \text{ biases}$
   - *Layer Parameters:* $4,608 + 32 = \mathbf{4,640}$
   - *Followed by:* Non-linear activation $\text{ReLU}(\cdot)$ and spatial downsampling $\text{MaxPool2d}(2, 2)$:
     $$\mathbf{H}_2 \in \mathbb{R}^{32 \times 8 \times 8}$$

3. **Feature Flattening:**
   - Unrolls spatial feature map into a 1D representation: $32 \times 8 \times 8 = 2,048 \text{ features}$.

4. **Trainable Layer 3 — Fully Connected Output Layer (`fc`):**
   - *Specification:* `in_features=2048`, `out_features=10`
   - *Weight Tensor:* $[10, 2048] \implies 10 \times 2048 = 20,480 \text{ weights}$
   - *Bias Tensor:* $[10] \implies 10 \text{ biases}$
   - *Layer Parameters:* $20,480 + 10 = \mathbf{20,490}$

$$\mathbf{\text{Total Trainable CNN Parameters}} = 448 + 4,640 + 20,490 = \mathbf{25,578}$$

---

### 3.2 Baseline Three-Layer MLP Architecture (`ThreeLayerMLP`)

$$\text{Input: Flattened } \mathbf{x} \in \mathbb{R}^{3072} \quad (32 \times 32 \times 3 = 3072)$$

1. **Trainable Layer 1 (`fc1`):**
   - $\text{Linear}(3072, 256) \implies (3072 \times 256) + 256 = \mathbf{786,688 \text{ parameters}}$
   - Followed by $\text{ReLU}(\cdot)$
2. **Trainable Layer 2 (`fc2`):**
   - $\text{Linear}(256, 128) \implies (256 \times 128) + 128 = \mathbf{32,896 \text{ parameters}}$
   - Followed by $\text{ReLU}(\cdot)$
3. **Trainable Layer 3 (`fc3`):**
   - $\text{Linear}(128, 10) \implies (128 \times 10) + 10 = \mathbf{1,290 \text{ parameters}}$

$$\mathbf{\text{Total Trainable MLP Parameters}} = 786,688 + 32,896 + 1,290 = \mathbf{820,874}$$

---

## 4. Experimental Setup & Dataset Integrity

- **Dataset Source:** CIFAR-10 via `torchvision.datasets.CIFAR10`.
- **Normalization:** Per-channel mean $\mu = (0.4914, 0.4822, 0.4465)$ and standard deviation $\sigma = (0.2470, 0.2435, 0.2616)$.
- **Partitioning Strategy:**
  - Standard training set of 50,000 images deterministically divided into:
    - **Training Subset:** 45,000 images (90%)
    - **Validation Subset:** 5,000 images (10%)
  - **Test Set:** 10,000 images preserved strictly untouched throughout training and hyperparameter selection.
- **Reproducibility Guarantee:** Random seeds fixed (`seed = 42`) across Python `random`, `numpy`, PyTorch CPU (`torch.manual_seed`), and PyTorch CUDA generators.

---

## 5. Empirical Results: All 15 Epochs Recorded

The table below provides the verbatim epoch-by-epoch telemetry recorded during the training of both models:

| Epoch | CNN Train Loss | CNN Train Acc (%) | CNN Val Loss | CNN Val Acc (%) | MLP Train Loss | MLP Train Acc (%) | MLP Val Loss | MLP Val Acc (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.4066 | 50.30% | 1.1985 | 57.34% | 1.6663 | 40.73% | 1.5899 | 44.02% |
| **2** | 1.0992 | 61.54% | 1.0613 | 62.52% | 1.4667 | 48.41% | 1.4736 | 47.90% |
| **3** | 0.9807 | 65.96% | 1.0392 | 64.14% | 1.3664 | 51.85% | 1.4493 | 49.94% |
| **4** | 0.9107 | 68.38% | 0.9729 | 66.90% | 1.2890 | 54.39% | 1.3785 | 52.36% |
| **5** | 0.8589 | 70.14% | 0.9569 | 66.92% | 1.2145 | 57.01% | 1.4324 | 50.44% |
| **6** | 0.8181 | 71.70% | 0.9830 | 66.12% | 1.1566 | 59.04% | 1.4128 | 52.24% |
| **7** | 0.7813 | 73.19% | 0.9681 | 66.28% | 1.0943 | 60.92% | 1.4228 | 52.14% |
| **8** | 0.7561 | 73.94% | 0.9629 | 67.80% | 1.0421 | 63.19% | 1.4658 | 50.96% |
| **9** | 0.7358 | 74.50% | 0.9421 | 68.20% | 0.9909 | 64.73% | 1.4883 | 52.48% |
| **10** | 0.7095 | 75.49% | 0.9051 | 69.14% | 0.9345 | 66.86% | 1.5831 | 50.88% |
| **11** | 0.6933 | 75.93% | 0.9424 | 68.60% | 0.8911 | 68.29% | 1.5645 | 51.64% |
| **12** | 0.6750 | 76.75% | 0.9388 | 69.36% | 0.8444 | 69.79% | 1.5940 | 50.94% |
| **13** | 0.6622 | 77.05% | 0.9292 | 69.52% | 0.8005 | 71.33% | 1.6798 | 50.94% |
| **14** | 0.6466 | 77.46% | 0.9417 | 68.74% | 0.7640 | 72.56% | 1.6875 | 51.42% |
| **15** | 0.6335 | 78.04% | 0.9543 | 68.92% | 0.7235 | 74.14% | 1.7610 | 52.00% |

### 5.1 Final Evaluation Summary (Requirement 4)

| Metric | 3-Layer MLP Baseline | 3-Layer CNN (Exercise 2) | Difference ($\Delta$) |
|---|:---:|:---:|:---:|
| **Trainable Parameters** | 820,874 | **25,578** | **-795,296 (-96.88%)** |
| **Final Training Loss** | 0.7235 | **0.6335** | -0.0900 |
| **Final Validation Loss** | 1.7610 | **0.9543** | **-0.8067** |
| **Final Training Accuracy** | 74.14% | **78.04%** | +3.90% |
| **Final Validation Accuracy** | 52.00% | **68.92%** | **+16.92%** |
| **Final Unbiased Test Accuracy** | 51.30% | **68.42%** | **+17.12%** |
| **Final Unbiased Test Loss** | 1.7617 | **0.9759** | **-0.7858** |

---

## 6. Critical Discussion: Convolutional Operations & Feature Learning (Requirement 5)

### 6.1 Spatial Locality and Inductive Bias
Natural photographic images possess a foundational statistical property known as **spatial locality**: adjacent pixels exhibit intense statistical mutual information (forming visual edges, gradients, and textures), whereas widely separated pixels have near-zero direct correlation.
- **The MLP's Structural Defect:** Flattening a $32 \times 32 \times 3$ image into a 3,072-dimensional vector irrevocably severs the 2D spatial coordinates $(x, y)$. The linear matrix multiplication $W \mathbf{x}$ assumes that pixel $(0, 0)$ is as structurally related to pixel $(31, 31)$ as it is to its immediate neighbor $(0, 1)$. To learn spatial concepts, the MLP must independently discover pixel neighbor relationships through brute-force optimization, leading to inefficient learning and poor data efficiency.
- **The CNN's Inductive Bias:** A convolutional layer slides small $3 \times 3$ kernel tensors over the spatial lattice. Computations are restricted exclusively to local receptive fields. The model inherently embeds the prior knowledge (inductive bias) that local pixel patterns matter, enabling rapid, targeted visual feature extraction.

### 6.2 Weight Sharing and Translation Equivariance
In an image classification task, an object feature (e.g., an animal's eye or a vehicle's tire) possesses identical semantic meaning regardless of whether it appears in the top-left or bottom-right corner of the image frame.
- **Weight Sharing:** Unlike the dense layer where every single input-output connection possesses an independent weight scalar, a CNN applies the **exact same kernel filter** across every spatial window of the image. 
- **Parameter Efficiency:** Because the weights are tied, 16 filters in the first layer require only 448 parameters, yet they process $32 \times 32 \times 3 = 3,072$ inputs. As proven in our experiment, this reduces the parameter count from 820,874 down to 25,578.
- **Translation Equivariance:** If an input pattern shifts by $(\Delta x, \Delta y)$, its response on the feature map shifts by the exact same offset: $f(g(x)) = g(f(x))$. When paired with `MaxPool2d`, the network acquires **local translation invariance**, ensuring that minor shifts or object movements do not perturb the classification decision.

### 6.3 Analysis of Overfitting and Generalisation Gap
The telemetry in Section 5 presents a quintessential demonstration of the **bias-variance tradeoff** and the mechanics of overfitting:
- **MLP Generalisation Failure:** The MLP began overfitting aggressively as early as **Epoch 4**. Between Epoch 4 and Epoch 15:
  - Training loss decreased from 1.2890 to 0.7235 (Training accuracy rose from 54.39% to 74.14%).
  - Validation loss **diverged**, increasing from 1.3785 to 1.7610 (Validation accuracy stagnated at ~51–52%).
  - The massive parameter space (820,874 weights) allowed the MLP to memorize idiosyncratic high-frequency noise and training sample specifics rather than generalizable visual semantics.
- **CNN Generalisation Stability:** The CNN maintained synchronous convergence between training and validation metrics. Even at Epoch 15, the validation loss was 0.9543 with a validation accuracy of 68.92%, tracking closely with the final test accuracy of 68.42%. The structural constraints of weight sharing and local receptive fields prevented parameter explosion, constraining the model's hypothesis space to semantically viable visual representations.

### 6.4 Hierarchical Feature Representation
The two-stage convolutional hierarchy implemented in `ThreeLayerCNN` reflects biological visual processing:
1. **Low-Level Primitives (`conv1`):** The 16 kernels learn localized spatial derivatives—detecting horizontal/vertical edges, color opponency, and high-contrast boundaries directly from the raw RGB channels.
2. **Intermediate Representations (`conv2`):** By taking the pooled $16 \times 16$ feature maps from Layer 1, the 32 kernels in Layer 2 combine basic edge primitives into composite visual tokens—such as curvatures, parallel lines, textures, and corner intersections.
3. **Global Spatial Summarization (`MaxPool2d`):** The two pooling operations compress the spatial footprint from $32 \times 32$ to $8 \times 8$, expanding the effective receptive field so that individual units in Layer 2 perceive broader multi-pixel contextual regions of the original image.
4. **Classification Readout (`fc`):** The final dense layer acts as a soft linear decision maker, weighting the presence of detected composite features to produce logits for the 10 target classes.

---

## 7. Submission Artifacts

All corresponding source files for Exercise 2 are organized in the project workspace:

- **Jupyter Notebook:** [`exercise_2/Exercise_2_CNN.ipynb`](file:///c:/Users/Asus/Desktop/AI&ML/PUSL3123-AI-ML-Assessment-1/exercise_2/Exercise_2_CNN.ipynb) (Executable notebook with pure English explanations, ready for ZIP archive submission).
- **Model Definitions:** [`exercise_2/model.py`](file:///c:/Users/Asus/Desktop/AI&ML/PUSL3123-AI-ML-Assessment-1/exercise_2/model.py) (`ThreeLayerCNN`, `ThreeLayerMLP`, `count_trainable_parameters`).
- **Training Pipeline:** [`exercise_2/train.py`](file:///c:/Users/Asus/Desktop/AI&ML/PUSL3123-AI-ML-Assessment-1/exercise_2/train.py) (Deterministic pipeline with automated archive repair).
- **Visualization Artifact:** `exercise_2_comparison.png` (Training and validation loss and accuracy comparative curves).
- **Traceability Documentation:** [`exercise_2/README.md`](file:///c:/Users/Asus/Desktop/AI&ML/PUSL3123-AI-ML-Assessment-1/exercise_2/README.md) and [`exercise_2/EXERCISE_2_REPORT.md`](file:///c:/Users/Asus/Desktop/AI&ML/PUSL3123-AI-ML-Assessment-1/exercise_2/EXERCISE_2_REPORT.md).
