***

# 🏥 MedTime-LLM: Bridging the Modality Gap in Clinical Time-Series

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--1.5B-green)](https://huggingface.co/Qwen/Qwen2.5-1.5B)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/Target-ICIC_2026-purple.svg)]()

*A novel cross-modal architecture that elegantly aligns continuous physiological time-series with the discrete semantic space of LLMs for interpretable predictive modeling.*

[**📖 Paper (Coming Soon)**](#) •[**⚡ Quick Start**](#-quick-start) • [**📊 Dataset**](#-dataset-preparation) •[**🤖 Citation**](#-citation)

</div>

---

## 💡 About The Project

While Large Language Models (LLMs) have demonstrated exceptional cognitive reasoning capabilities, their direct application to multivariate time-series forecasting is fundamentally hindered by the **Modality Gap**. 

**MedTime-LLM** overcomes this barrier by introducing a **Dual-Pathway Alignment Strategy**:
1. **Implicit Pathway (MSTR)**: A parallel bank of 1D-CNNs (kernels 3, 7, 11) extracts multi-resolution temporal dynamics (acute spikes & chronic trends) and projects them into the LLM's latent space as *soft tokens*.
2. **Explicit Pathway (CoT)**: Dynamic deterministic prompts extract Top-3 volatile biomarkers (Max, Min, Latest) to ground the LLM's diagnostic reasoning.

By transforming opaque numerical probability scores into **auditable, evidence-based narratives**, MedTime-LLM effectively mitigates clinical *alarm fatigue* and empowers human-in-the-loop ICU triage.

## ✨ Key Features

- **🧠 Multi-Scale Temporal Reprogramming (MSTR):** Bypasses lossy numerical discretization by mapping continuous 48-hour physiological signals directly to the 1536-D LLM embedding space.
- **📝 Zero-Shot Chain-of-Thought (CoT) Triage:** Generates highly readable clinical rationales alongside its predictions, justifying alerts based on the patient's latest stabilization trajectory.
- **⚡ Parameter-Efficient Fine-Tuning (PEFT):** Leverages LoRA (Low-Rank Adaptation) on `q_proj` and `v_proj` attention layers, making it feasible to train on a single consumer-grade GPU (e.g., RTX 3090/4090).
- **🛡️ High Clinical Safety (Recall-Oriented):** Asymmetrically optimizes for high sensitivity to prevent fatal false negatives in ICU bed allocation.
<div align="center">
  <img src="assets/medtime_architecture.png" alt="MedTime-LLM Architecture" width="85%">
  <p><em>Figure 1: The Dual-Pathway Architecture of MedTime-LLM. The implicit pathway extracts multi-scale temporal dynamics (MSTR), while the explicit pathway constructs dynamic Chain-of-Thought (CoT) prompts. Both are fused to guide the Qwen-1.5B LLM.</em></p>
</div>

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/MedTime-LLM.git
cd MedTime-LLM
```

**2. Create a virtual environment and install dependencies**
```bash
conda create -n medtime python=3.10 -y
conda activate medtime

# Install PyTorch (Update the CUDA version according to your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install transformers peft scikit-learn numpy tqdm
```

---

## 📊 Dataset Preparation

We use the publicly available **MIMIC-III** clinical database. Due to data privacy regulations (PhysioNet Data Use Agreement), we cannot host the raw data. 

1. Gain access to MIMIC-III via [PhysioNet](https://physionet.org/content/mimiciii/).
2. Extract the 48-hour multivariate time-series (76 features) and binarize the target (LOS > 7 Days).
3. Save the processed numpy arrays into the `data/processed_numpy/` directory:

```text
MedTime-LLM/
├── data/
│   └── processed_numpy/
│       ├── X_train.npy   # Shape: (N_train, 48, 76)
│       ├── y_train.npy   # Shape: (N_train,)
│       ├── X_test.npy    # Shape: (N_test, 48, 76)
│       └── y_test.npy    # Shape: (N_test,)
```

---

## 🚀 Quick Start

To launch the end-to-end training and evaluation pipeline:

```bash
python main.py
```

### What happens under the hood?
1. **Dataloader**: Dynamically applies golden-ratio oversampling (1:1) to balance the positive (Prolonged LOS) and negative classes. Extracts the *Top-3 most volatile biomarkers*.
2. **Training**: Aligns `Qwen2.5-1.5B` via LoRA and trains the MSTR module jointly using gradient accumulation (Batch Size = 1 $\times$ 4 steps = 4).
3. **Inference**: Autoregressively generates predictions and logs clinical case studies.

---

## 🔍 Explainable Clinical Output Example

Unlike "black-box" models that merely output a probability (e.g., `0.85`), MedTime-LLM yields actionable narrative reports:

```text
[=============== Case Study #1 ===============]
📌 【Ground Truth】: 🔴 Prolonged Stay (> 7 days)
📝 【AI Diagnostic & Triage Report】:
🚨 PROLONGED STAY ALERT: The patient requires an extended stay (> 7 days). 
Reasoning: The biomarkers (Heart Rate (Max: 135.0, Min: 78.0, Latest: 95.0 bpm); 
Respiratory Rate (Max: 28.0, Min: 15.0, Latest: 20.0 insp/min)) show severe fluctuations, 
and the 'Latest' values indicate that the patient has NOT stabilized at the end of the 
48-hour window. 
Recommendation: Multidisciplinary review recommended to address unresolved complications.
-----------------------------------------------------------------
```

---

## 📁 Repository Structure

```text
├── main.py               # Main training & evaluation loop, MSTR & TimeLLM Architectures
├── dataloader.py         # MIMIC-III dataset loader, Prompt engineering & CoT generation
├── README.md             # Project documentation
├── requirements.txt      # Dependency list
└── data/                 # Directory for local .npy dataset files
```

---

## ⚖️ License & Disclaimer

This project is licensed under the [MIT License](LICENSE).

**⚠️ Medical Disclaimer**: *MedTime-LLM is designed for research purposes only. It is not intended to be used as a standalone medical device or diagnostic tool. Any clinical deployment must adhere to local regulations and remain strictly within a Human-in-the-Loop (HITL) operational paradigm.*

---

## 🤖 Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@inproceedings{zhao2026medtimellm,
  title={MedTime-LLM: Bridging the Modality Gap in Clinical Time-Series via Multi-Scale Temporal Reprogramming},
  author={Zhao, Boxian and [Co-author Name] and[Corresponding Author]},
  booktitle={Proceedings of the International Conference on Intelligent Computing (ICIC)},
  year={2026},
  publisher={Springer LNCS}
}
```

<div align="center">
  <i>Developed with ❤️ by the MedTime-LLM Research Team</i>
</div>

***

