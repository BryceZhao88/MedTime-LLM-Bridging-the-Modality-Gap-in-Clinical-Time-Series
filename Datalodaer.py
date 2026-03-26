"""
dataloader.py
====================================
Dataset processing and Chain-of-Thought (CoT) prompt generation for MedTime-LLM.
Handles class imbalance, multimodal feature alignment, and text tokenization.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

class MIMICNumpyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MIMIC-III time-series data and dynamically 
    generating clinical Chain-of-Thought (CoT) prompts for Large Language Models.
    """
    def __init__(self, x_path, y_path, tokenizer_name="Qwen/Qwen2.5-1.5B", max_length=512, balance_data=True):
        # Load preprocessed continuous time-series arrays
        X_all = np.load(x_path, allow_pickle=True)
        
        # Load binary labels: 0 (Standard Stay <= 7 days) and 1 (Prolonged Stay > 7 days)
        # Note: If y_all contains exact days, uncomment the following line to binarize:
        # y_all = (y_all > 7.0).astype(int) 
        y_all = np.load(y_path, allow_pickle=True).astype(np.float32)
        
        # Class Balancing: Oversampling the minority class (Prolonged Stay)
        if balance_data:
            print("⚖️ Initializing 1:1 Oversampling to balance Standard and Prolonged LOS classes...")
            pos_indices = np.where(y_all == 1)[0]
            neg_indices = np.where(y_all == 0)[0]
            
            np.random.seed(42)
            # Match the number of positive samples to the negative samples (1:1 absolute equality)
            target_pos_count = int(len(neg_indices) * 1.0) 
            sampled_pos_indices = np.random.choice(pos_indices, target_pos_count, replace=True)
            
            balanced_indices = np.concatenate([neg_indices, sampled_pos_indices])
            np.random.shuffle(balanced_indices)
            
            self.X_data = X_all[balanced_indices]
            self.y_data = y_all[balanced_indices]
            print(f"📊 Dataset Balanced | Prolonged (>7d): {target_pos_count} | Standard (<=7d): {len(neg_indices)}")
        else:
            self.X_data = X_all
            self.y_data = y_all

        # Initialize LLM Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        # Ordered list of core physiological features
        self.FEATURE_NAMES =[
            "Capillary Refill Rate", "Diastolic Blood Pressure", "Fractional Inspired Oxygen",
            "Glascow Coma Scale Eye", "Glascow Coma Scale Motor", "Glascow Coma Scale Verbal",
            "Glucose", "Heart Rate", "Height", "Mean Blood Pressure", "Oxygen Saturation",
            "Respiratory Rate", "Systolic Blood Pressure", "Temperature", "Weight"
        ]

        # Dictionary for reverse standardization (Mean/Std mapping to recover real clinical values)
        self.NORM_DICT = {
            "Capillary Refill Rate": {"mean": 0.0, "std": 1.0, "unit": ""}, 
            "Diastolic Blood Pressure": {"mean": 59.0, "std": 13.0, "unit": "mmHg"},
            "Fractional Inspired Oxygen": {"mean": 0.5, "std": 0.2, "unit": "%"},
            "Glascow Coma Scale Eye": {"mean": 3.4, "std": 0.8, "unit": "points"},
            "Glascow Coma Scale Motor": {"mean": 5.3, "std": 1.1, "unit": "points"},
            "Glascow Coma Scale Verbal": {"mean": 3.9, "std": 1.4, "unit": "points"},
            "Glucose": {"mean": 135.0, "std": 45.0, "unit": "mg/dL"},
            "Heart Rate": {"mean": 85.0, "std": 16.0, "unit": "bpm"},
            "Height": {"mean": 170.0, "std": 15.0, "unit": "cm"},
            "Mean Blood Pressure": {"mean": 77.0, "std": 15.0, "unit": "mmHg"},
            "Oxygen Saturation": {"mean": 97.0, "std": 3.0, "unit": "%"},
            "Respiratory Rate": {"mean": 19.0, "std": 5.0, "unit": "insp/min"},
            "Systolic Blood Pressure": {"mean": 118.0, "std": 21.0, "unit": "mmHg"},
            "Temperature": {"mean": 36.9, "std": 0.8, "unit": "°C"},
            "Weight": {"mean": 81.0, "std": 24.0, "unit": "kg"}
        }

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = self.X_data[idx]
        T = x.shape[0]
        
        # Temporal Padding / Truncation to strictly enforce 48-hour windows
        if T < 48:
            x = np.pad(x, ((0, 48 - T), (0, 0)), mode='constant', constant_values=0.0)
        elif T > 48:
            x = x[:48, :]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        label = self.y_data[idx]
        
        # Calculate feature statistics across the temporal axis
        fluctuations = np.std(x, axis=0) 
        max_vals = np.max(x, axis=0)
        min_vals = np.min(x, axis=0)
        # 🌟 Core Feature: Extract the latest physiological state at the 48th hour
        last_vals = x[-1, :] 
        
        # Dynamically identify the Top-3 most volatile biomarkers for this specific patient
        top3_indices = np.argsort(fluctuations[:15])[-3:][::-1] 
        
        alert_messages =[]
        for feat_idx in top3_indices:
            biomarker_name = self.FEATURE_NAMES[feat_idx]
            mean = self.NORM_DICT[biomarker_name]["mean"]
            std = self.NORM_DICT[biomarker_name]["std"]
            unit = self.NORM_DICT[biomarker_name]["unit"]
            
            # De-standardize to recover human-readable clinical values
            real_max = max_vals[feat_idx] * std + mean
            real_min = min_vals[feat_idx] * std + mean
            real_last = last_vals[feat_idx] * std + mean 
            
            # 🌟 Evidence Chain: Construct statistical anchors (Max, Min, Latest)
            alert_messages.append(
                f"{biomarker_name} (Max: {real_max:.1f}, Min: {real_min:.1f}, Latest: {real_last:.1f} {unit})"
            )
            
        combined_alerts = "; ".join(alert_messages)

        # 🌟 Implicit & Explicit Prompt Fusion: Force LLM to evaluate the stabilization trend
        input_text = (
            "You are an expert ICU triage physician. Analyze the 48-hour physiological data provided as soft embeddings. "
            f"Clinical Context (Top 3 volatile biomarkers): {combined_alerts}. "
            "INSTRUCTION: Think step-by-step. Compare the 'Latest' value with the 'Max' and 'Min' values. "
            "If the 'Latest' value shows stabilization (returning to normal), the patient is recovering. "
            "If the 'Latest' value remains critical, classify as PROLONGED STAY (>7 days). Otherwise, classify as STANDARD STAY.\n\n"
            "### Clinical Assessment:\n"
        )

        # 🌟 Target Generation Logic: Embed explicit clinical reasoning into the ground truth
        report_parts =[]
        if label == 1: 
            report_parts.append("🚨 PROLONGED STAY ALERT: The patient requires an extended stay (> 7 days).")
            # Teach the model how to deduce "chronic deterioration"
            report_parts.append(f"Reasoning: The biomarkers ({combined_alerts}) show severe fluctuations, and the 'Latest' values indicate that the patient has NOT stabilized at the end of the 48-hour window.")
            report_parts.append("Recommendation: Multidisciplinary review recommended to address unresolved complications.")
        else: 
            report_parts.append("✅ STANDARD STAY: The patient is expected to follow a standard recovery (<= 7 days).")
            # Teach the model how to deduce "recovery and stabilization"
            report_parts.append(f"Reasoning: Although historical extremes were observed ({combined_alerts}), the 'Latest' values demonstrate a stabilizing trend towards baseline at the end of the 48-hour window.")
            report_parts.append("Recommendation: Continue care protocol. Evaluate for step-down unit transfer.")
            
        target_text = " ".join(report_parts) + self.tokenizer.eos_token
        
        # Tokenization
        full_text = input_text + target_text
        tokenized = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        # 🌟 Loss Masking: Ignore the prompt tokens during Cross-Entropy Loss calculation
        prompt_len = len(self.tokenizer(input_text, truncation=True, max_length=self.max_length)['input_ids'])
        labels[:prompt_len] = -100 
        labels[attention_mask == 0] = -100

        return {
            'x_seq': x_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels, 
            'prompt_text': input_text,
            'y_true': label
        }
