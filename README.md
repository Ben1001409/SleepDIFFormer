<h2 align="center"> <a href="https://openreview.net/forum?id=ftGnpZrW7P">SleepDIFFormer</a></h2>

<h3 align="center"><a href="https://ispamm.github.io/GRAM/"> Project page</a></h3>

<div align=center><img src=assets/gram_method-compresso-1.png width="75%" height="75%"></div>


<h5 align="center">

## 🧩 Introduction
SleepDIFFormer is a transformer-based framework for automated modeling of multivariate sleep signals across multiple modalities. It is designed to effectively capture cross-modal dependencies and support accurate and robust downstream tasks. SleepDIFFormer has the following three highlights:

### 🧬 Differential Transformer for Multimodal Time-Series 
We design a transformer-based architecture with differential attention to model the feature fusion and timing dependencies of multimodal, multivariate physiological signals.

### 💤 EEG/EOG-Based Sleep Stage Classification
The model processes EEG and EOG signals using CNN-based series embedding and attention-based encoders, effectively capturing intra- and inter-modality patterns for accurate sleep staging.

### 🌍 Cross-Domain Generalization on Public Datasets
To ensure robustness, we evaluate the model on 5 publicly available datasets and apply multi-level domain alignment losses to enhance generalization and outperform existing baselines.

## 📊 Main Results

<div align=center><img src=figure/result.png width="80%" height="80%"></div>

## 🚀 Getting Started

### ⚙️ Environment Setup
We follow a Domain Generalization (DG) setup using 4 datasets for training and 1 for testing in rotation. Each input consists of 20 consecutive sleep epochs with 128-dimensional features. The model is trained for 50 epochs with a batch size of 16 using the Adam optimizer (lr = 5e-4) and evaluated by Accuracy and Macro-F1. All experiments are implemented in PyTorch and run on a single RTX 4090 GPU.

### 📁 Data Preparation
data/  
├── [SleepEDFx](https://doi.org/10.1109/10.867928)  
├── [HMC](https://doi.org/10.13026/gp48-ea60)  
├── [ISRUC](https://doi.org/10.1016/j.cmpb.2015.10.013)  
├── [SHHS](https://doi.org/10.1093/sleep/20.12.1077)  
├── [P2018](https://doi.org/10.22489/cinc.2018.049)  

### 🏃‍♀️ Run Training

### 🔍 Run Evaluation


## 🧠 Acknowledgments
This work is heavily based on [SleepDG](https://arxiv.org/abs/2401.05363). Thank all the authors and their contributions.
