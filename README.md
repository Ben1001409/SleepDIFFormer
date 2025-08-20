<h2 align="center"> <a href="https://openreview.net/forum?id=ftGnpZrW7P">SleepDIFFormer</a></h2>

<h3 align="center"><a href="https://lanxin1105.github.io/SleepDIFFormer-Page/"> 📄 Project page</a></h3>
<h4 align="center"><a href="https://huggingface.co/Benjamin1001/SleepDIFFormer/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=yellow" alt="Hugging Face Model Checkpoints" />
</a></h4>

<div align=center><img src=figure/general.png width="95%" height="95%"></div>


<h5 align="center">

## 🧩 Introduction
SleepDIFFormer is a transformer-based framework for automated modeling of sleep signals across multiple modalities. It is designed to effectively mitigate variability in biosignals and provide interpretability of the model through attention weights visualization. SleepDIFFormer has the following three highlights:

### 🧬 Differential Transformer for Multimodal Time-Series 
We design a transformer-based architecture with differential attention to model non-stationarity of multimodal bio-signals under domain generalization settings

### 💤 EEG/EOG-Based Sleep Stage Classification
The model processes EEG and EOG signals using CNN-based series embedding and attention-based encoders, effectively capturing intra- and inter-modality patterns for accurate sleep staging. It also provides attention weights visualization to identify which part of the signal is suppressed or amplified.

### 🌍 Cross-Domain Generalization on Public Datasets
To ensure robustness, we evaluate the model on 5 publicly available datasets and apply multi-level domain alignment losses to enhance generalization and outperform existing baselines.

## 📊 Main Results

<div align=center><img src=figure/result.png width="95%" height="95%"></div>
Cross-domain sleep-stage classification performance comparison. TD = target domain. Each column shows results when TD is the test domain and remaining are used for training. Best values are bolded.

###  
<div align=center><img src=figure/eeg_visual.png width="95%" height="95%"></div>
ttention weight allocation visualization on EEG and EOG across 4 layers with wake and different sleep stages


## 🚀 Getting Started

### 📁 Data Preparation
data/  
├── [SleepEDFx](https://doi.org/10.1109/10.867928)  
├── [HMC](https://doi.org/10.13026/gp48-ea60)  
├── [ISRUC](https://doi.org/10.1016/j.cmpb.2015.10.013)  
├── [SHHS](https://doi.org/10.1093/sleep/20.12.1077)  
├── [P2018](https://doi.org/10.22489/cinc.2018.049)  

### ⚙️ Environment Setup
We follow a Domain Generalization (DG) setup using 4 datasets for training and 1 for testing in rotation. Each input consists of 20 consecutive sleep epochs with 128-dimensional features. The model is trained for 50 epochs with a batch size of 16 using the Adam optimizer (lr = 5e-4) and evaluated by Accuracy and Macro-F1. All experiments are implemented in PyTorch and run on a single RTX 4090 GPU.


### 🏃‍♀️ Run Training and Evaluation
```python
python main.py --batch_size 16 --return_attention True --lr 0.0005 --num_heads 4 --num_layers 4 --d_model 128 --d_ff 512

```


## 🖊️ BibTeX
```BibTeX
@article{xxx2025sleepdifformer,
  author  = {Name},
  title   = {SleepDIFFormer: Multimodal Differential Transformer for Sleep Staging},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025}
}
```

## 🧠 Acknowledgments
This work is heavily based on [SleepDG](https://arxiv.org/abs/2401.05363). Thank all the authors and their contributions.
