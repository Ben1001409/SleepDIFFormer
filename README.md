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



## 🧠 Acknowledgments
This work is heavily based on [SleepDG](https://arxiv.org/abs/2401.05363). Thank all the authors and their contributions.
