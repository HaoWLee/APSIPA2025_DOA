# DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes



Contact: **Haowen Li** (haowen.li@ntu.edu.sg), *Nanyang Technological University*

This repository contains the training and evaluation code for our APSIPA2025 paper:

"DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes"


## Python Files

- **`cp_mobile_2025.py`**  
  Implements a lightweight MobileNet-style convolutional network for direction-of-arrival (DOA) classification, including the core building blocks and the model factory function.

- **`cp_mobile_train.py`**  
  Training script for the compact Mobile-based DOA model, built with PyTorch Lightning and integrated with Weights & Biases for logging, checkpointing, and evaluation.

- **`cp_mobile_dataset.py`**  
  Dataset and feature-processing utilities for DOA experiments. It loads stereo audio and annotations, extracts STFT/IPD/ILD features, and provides multiple dataset variants for different angle resolutions and task settings.

- **`mobile_gru_net_train_ipd5.py`**  
  Training pipeline for the lightweight GRU-based DOA model (`CPGruNet5`), including data loading, logging, checkpointing, and final evaluation.

- **`cp_mobile_gru.py`**  
  Defines several compact CNN/GRU hybrid architectures for DOA classification, built around depthwise separable convolutions and optional GRU sequence modeling.

- **`crnn_model.py`**  
  Defines a CRNN-based DOA classifier that combines convolutional feature extraction, a bidirectional LSTM, and a fully connected classifier.

- **`CRNN_train.py`**  
  Training script for the CRNN-based DOA model, including dataset loading, weighted loss support, Weights & Biases logging, checkpointing, and testing.


## Citation

If you use this code or find it helpful for your research, please cite our paper:

```bibtex
@inproceedings{li2025doa,
  title={DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes},
  author={Li, Haowen and Luo, Zhengding and Shi, Dongyuan and Wang, Boxiang and Ji, Junwei and Yang, Ziyi and Gan, Woon-Seng},
  booktitle={2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={423--428},
  year={2025},
  organization={IEEE}
}

