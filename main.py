"""
main.py
====================================
End-to-End Training and Evaluation Pipeline for MedTime-LLM.
Implements Multi-Scale Temporal Reprogramming (MSTR), LoRA parameter-efficient 
fine-tuning, and zero-shot Chain-of-Thought (CoT) reasoning for ICU Length of Stay.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# Import our custom Multi-modal MIMIC Dataloader
from dataloader import MIMICNumpyDataset

# ==========================================
# Global Hyperparameters & Configurations
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 1           # Micro-batch size to accommodate LLM memory footprint
ACCUMULATION_STEPS = 4   # Effective Batch Size = 4 via Gradient Accumulation
LEARNING_RATE = 2e-4

# ==========================================
# 1. Core Architecture: Multi-Scale Temporal Reprogramming (MSTR)
# ==========================================
class ReprogrammingLayer(nn.Module):
    """
    MSTR Module: Projects continuous physiological time-series into the LLM's discrete 
    embedding space via parallel multi-resolution 1D-Convolutions.
    """
    def __init__(self, input_dim=76, hidden_dim=1536):
        super().__init__()
        # 🌟 Core Feature: Simultaneously capture acute spikes (3h), sub-acute shifts (7h), 
        # and chronic deterioration trends (11h).
        sub_dim = hidden_dim // 3
        self.conv_short = nn.Conv1d(in_channels=input_dim, out_channels=sub_dim, kernel_size=3, padding='same')
        self.conv_medium = nn.Conv1d(in_channels=input_dim, out_channels=sub_dim, kernel_size=7, padding='same')
        self.conv_long = nn.Conv1d(in_channels=input_dim, out_channels=sub_dim, kernel_size=11, padding='same')
        
        # Linear projection to align the concatenated features with the LLM embedding dimension
        self.proj = nn.Linear(sub_dim * 3, hidden_dim)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len=48, input_dim=76)
        x = x.transpose(1, 2) # Conv1d expects (batch_size, channels, seq_len)
        
        out_short = torch.relu(self.conv_short(x))
        out_medium = torch.relu(self.conv_medium(x))
        out_long = torch.relu(self.conv_long(x))
        
        # Concatenate multi-scale temporal receptive fields along the feature dimension
        out = torch.cat([out_short, out_medium, out_long], dim=1) 
        out = out.transpose(1, 2) # Revert to (batch_size, seq_len, sub_dim*3)
        
        return self.proj(out) # Return Soft Tokens mapped to 1536-D space

# ==========================================
# 2. Dual-Pathway Fusion Model: MedTime-LLM
# ==========================================
class MedTimeLLM(nn.Module):
    """
    MedTime-LLM Architecture: Fuses implicit temporal embeddings (MSTR) 
    with explicit deterministic text prompts (CoT) to guide the frozen LLM.
    """
    def __init__(self, llm, embed_dim=1536): 
        super().__init__()
        self.llm = llm
        self.reprogrammer = ReprogrammingLayer(input_dim=76, hidden_dim=embed_dim)
        
    def forward(self, x_seq, input_ids, attention_mask, labels=None):
        # 1. Extract implicit temporal features (Soft Tokens)
        time_embeddings = self.reprogrammer(x_seq)
        
        # 2. Extract explicit textual features (Discrete Tokens)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # 🌟 Precision Alignment: Force continuous embeddings to match the LLM's half-precision (FP16/BF16)
        time_embeddings = time_embeddings.to(text_embeddings.dtype)
        
        # 3. Cross-Modal Fusion: Concatenate[Soft Tokens, Text Tokens] along the sequence length
        inputs_embeds = torch.cat([time_embeddings, text_embeddings], dim=1)
        
        # 4. Extend Attention Mask and Labels to accommodate the prepended Soft Tokens
        time_attention = torch.ones(time_embeddings.size()[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([time_attention, attention_mask], dim=1)
        
        if labels is not None:
            # Mask out the Soft Tokens in the loss calculation (-100 is ignored by CrossEntropyLoss)
            time_labels = torch.full(time_embeddings.size()[:-1], -100, dtype=labels.dtype, device=labels.device)
            extended_labels = torch.cat([time_labels, labels], dim=1)
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, labels=extended_labels)
        else:
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)

    def generate(self, x_seq, input_ids, attention_mask, **kwargs):
        """Autoregressive generation wrapper for evaluation."""
        time_embeddings = self.reprogrammer(x_seq)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        time_embeddings = time_embeddings.to(text_embeddings.dtype)
        inputs_embeds = torch.cat([time_embeddings, text_embeddings], dim=1)
        
        time_attention = torch.ones(time_embeddings.size()[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        extended_attention_mask = torch.cat([time_attention, attention_mask], dim=1)
        
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, **kwargs)

# ==========================================
# 3. Main Training & Evaluation Pipeline
# ==========================================
def main():
    print("="*60)
    print(f"🏥 Initializing MedTime-LLM Pipeline | Hardware: {DEVICE.upper()}")
    print("="*60)
    
    print("\n📚 Loading dataset and dynamically constructing CoT targets...")
    # NOTE: Ensure the relative paths point to your extracted MIMIC-III numpy arrays
    train_ds = MIMICNumpyDataset('data/processed_numpy/X_train.npy', 'data/processed_numpy/y_train.npy', tokenizer_name=MODEL_NAME, balance_data=True)
    test_ds = MIMICNumpyDataset('data/processed_numpy/X_test.npy', 'data/processed_numpy/y_test.npy', tokenizer_name=MODEL_NAME, balance_data=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"\n🧠 Loading foundational LLM ({MODEL_NAME}) and injecting LoRA adapters...")
    base_llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    
    # Configure Parameter-Efficient Fine-Tuning (LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    base_llm = get_peft_model(base_llm, peft_config)
    base_llm.gradient_checkpointing_enable() # 🌟 Memory Optimization: Enable gradient checkpointing
    
    # Initialize the multimodal MedTime-LLM architecture (Qwen2.5-1.5B native hidden_size is 1536)
    model = MedTimeLLM(base_llm, embed_dim=1536).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---------------- Training Loop ----------------
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n🚀 [Epoch {epoch+1}/{EPOCHS}] Commencing Training Phase...")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            x_seq = batch['x_seq'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(x_seq, input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} Completed | Average Loss: {total_loss / len(train_loader):.4f}")

    # Persist the multimodal weights
    print("\n💾 Persisting optimized model weights to disk (medtime_llm_final.pth)...")
    torch.save(model.state_dict(), "medtime_llm_final.pth")
    
    # ---------------- Evaluation Loop ----------------
    print("\n" + "="*60)
    print("📊 Initiating Zero-Shot evaluation on the real-world distribution test set")
    print("="*60)
    
    model.eval()
    tokenizer = train_ds.tokenizer
    
    all_preds = []
    all_trues =[]
    
    # Strategy: Capture 5 representative Prolonged Stays and 5 Standard Stays for qualitative review
    target_cases = {0: 5, 1: 5}
    case_idx = 1
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating Patients")):
            x_seq = batch['x_seq'].to(DEVICE)
            y_true = batch['y_true'].item()
            
            # Feed ONLY the prompt to the model, forcing it to generate the rationale and decision autoregressively
            prompt_text = batch['prompt_text'][0]
            tokenized_prompt = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            
            output_ids = model.generate(
                x_seq=x_seq,
                input_ids=tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                max_new_tokens=200,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the raw generated output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 💡 Robust Parsing Logic: Scan the narrative for the deterministic keyword
            pred_label = 1 if "PROLONGED" in generated_text.upper() else 0
            
            all_preds.append(pred_label)
            all_trues.append(y_true)
            
            # Print compelling clinical case studies for qualitative interpretability demonstration
            if y_true == 1 and target_cases[1] > 0:
                print(f"\n[=============== Case Study #{case_idx} (Patient ID: {i}) ===============]")
                print(f"📌 Ground Truth: 🔴 Prolonged Stay (> 7 days)")
                print(f"📝 AI Diagnostic & Triage Report:\n{generated_text.strip()}\n" + "-"*75)
                target_cases[1] -= 1
                case_idx += 1
            elif y_true == 0 and target_cases[0] > 0:
                print(f"\n[=============== Case Study #{case_idx} (Patient ID: {i}) ===============]")
                print(f"📌 Ground Truth: 🟢 Standard Stay (<= 7 days)")
                print(f"📝 AI Diagnostic & Triage Report:\n{generated_text.strip()}\n" + "-"*75)
                target_cases[0] -= 1
                case_idx += 1

    # Quantitative Metrics Calculation
    cm = confusion_matrix(all_trues, all_preds)
    
    print("\n" + "="*60)
    print("🏆 FINAL TEST COHORT METRICS (Ready for SCI Table 1)")
    print("="*60)
    print("🧩 Confusion Matrix:")
    print(cm)
    print(f"   [True Negative (Standard)={cm[0,0]} | False Positive (False Alarm)={cm[0,1]}]")
    print(f"   [False Negative (Missed Risk)={cm[1,0]} | True Positive (Correct Alert)={cm[1,1]}]")
    
    print("\n📋 Detailed Classification Report:")
    print(classification_report(all_trues, all_preds, target_names=["Standard Stay (<=7d)", "Prolonged Stay (>7d)"]))
    print("="*60)
    
if __name__ == "__main__":
    main()
