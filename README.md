# 🌍 Multilingual Sentiment Analysis using Gemma + LoRA

## 📌 Overview
This project focuses on **multilingual sentiment classification** across 13 Indian languages using **large language models (LLMs)**. The model is fine-tuned using **Low-Rank Adaptation (LoRA)** to efficiently adapt a pretrained model under limited compute constraints.

_competetion link: https://www.kaggle.com/competitions/nppe-dlp-2026-term-1_

The goal is to classify text into:
- **Positive**
- **Negative**

---

## 🚀 Approach

### 🔹 Model
- **Gemma (Transformer-based LLM)**

### 🔹 Fine-Tuning Technique
- **LoRA (Low-Rank Adaptation)** via PEFT
- Parameter-efficient training instead of full fine-tuning

### 🔹 Libraries Used
- `transformers`
- `peft`
- `bitsandbytes`
- `accelerate`
- `pandas`, `numpy`

---

## ⚙️ Pipeline

1. **Data Loading**
   - Train and test datasets with multilingual text

2. **Preprocessing**
   - Cleaning and formatting text inputs
   - Preparing labels for classification

3. **Tokenization**
   - Using pretrained tokenizer compatible with Gemma

4. **Model Setup**
   - Loading base LLM
   - Applying LoRA configuration (rank, alpha, etc.)

5. **Training**
   - Fine-tuning using HuggingFace Trainer / similar pipeline
   - Memory-efficient setup using `bitsandbytes`

6. **Evaluation**
   - Predictions generated on validation/test data
   - Performance measured using **F1 Score**

---

## 📊 Dataset

- **Train Set**: ~900 labeled samples  
- **Test Set**: ~100 samples  
- Each sample includes:
  - `sentence`
  - `label` (train only)
  - `language`

---

## 🌐 Supported Languages
- Assamese, Bodo, Bengali, Gujarati, Hindi  
- Kannada, Malayalam, Marathi, Odia, Punjabi  
- Tamil, Telugu, Urdu  

---

## 📈 Evaluation Metric

### F1 Score
Balances:
- Precision
- Recall

Used as the primary metric for model performance.

---

## 🧠 Key Learnings

- Practical use of **LoRA for efficient LLM fine-tuning**
- Handling **multilingual data with limited samples**
- Trade-offs between **model performance and compute efficiency**
- Importance of preprocessing and tokenization in multilingual NLP

---

## ⚡ Limitations

- Limited dataset size affects generalization  
- Model performance varies across languages  
- Compute constraints restrict extensive hyperparameter tuning  

---

## 🔮 Future Work

- Experiment with **QLoRA / quantization**
- Improve multilingual balance with data augmentation
- Hyperparameter tuning for better generalization
- Explore **optimization strategies for LoRA**

---

## 🧑‍💻 Author
**Sagandeep Kaur**  
