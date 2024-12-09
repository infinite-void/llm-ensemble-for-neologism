# LLM-Based Fine-Tuning for Neologism Analysis

This project explores fine-tuning five different language models for **neologism sentiment analysis**. The trained models are integrated into an ensemble.

## Overview
Neologisms, or newly coined words and expressions, present unique challenges in natural language processing (NLP). This project fine-tunes pre-trained language models to identify and analyze neologisms effectively. The models used in this study include:

1. **DistilBERT**
2. **GPT-2**
3. **RoBERTa**
4. **BERTweet**
5. **VADER**
6. **FinBERT**
   
---
## Fine-Tuning Approaches and Results for DistilBERT
The related code for the finetuning apporoaches can be found in DistilBERT_Fine_Tuning_For_Neologism.ipynb

### 1. **LoRA Fine-Tuning on Neologism Data**
- **Description:** Applied Low-Rank Adaptation (LoRA) to fine-tune specific layers of DistilBERT while freezing the rest. 
- **Result:** Achieved good training accuracy but experienced overfitting on the test set due to the limited dataset size.

### 2. **Task-Based Fine-Tuning for Sentiment Analysis**
- **Description:** Fine-tuned DistilBERT for sentiment analysis, leveraging labeled data to align the model with the specific task.
- **Result:** Improved accuracy from **0.36 to 0.59** on the reddit dataset, addressing some challenges in handling neologisms.

### 3. **Adaptation through DoRA (Domain-Specific Robust Adaptation)**
- **Description:** Introduced domain-specific robust adaptation to enhance the model’s ability to handle neologisms and reduce language drift.
- **Result:** Further improved accuracy to **0.62** on the reddit dataset, but still struggled to generalize effectively.

### 4. **Task-Based LoRA Fine-Tuning for Neologisms**
- **Description:** Combined task-specific fine-tuning with LoRA to adapt the model to the Neologism Data efficiently.
- **Result:** Achieved the best performance with an accuracy of **0.74** on the reddit dataset, significantly reducing overfitting. This model had an accuracy of **0.86** on the twitter dataset.

---

## Finetuning the GPT-2 model
While finetuning the GPT-2 model, the model is first fine-tuning for the task-specific objective of sentiment analysis. Once the model performs well on sentitment analysis, we then
finetune the model to perform well on Neologisms. 

1. finetuning_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task
2. finetune_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis to perform well on sentences with neologisms
3. lora_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task using the Low Rank Adaptation (LoRA) finetuning
4. lora_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis (using LoRA method) to perform well on sentences with neologisms using LoRA finetuning
5. ada_lora_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task using the Adaptive Budget Low Rank Adaptation (AdaLoRA) finetuning
6. ada_lora_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis (using AdaLoRA method) to perform well on sentences with neologisms using AdaLoRA finetuning

## Ensemble Approach
Each model is trained individually on the neologism dataset and contributes to the final prediction in the deployed application. The ensemble aggregates outputs from all models to improve accuracy and robustness.
We perform three ensemble techniques.

### Majority Voting: Simplest Ensemble Model
The simplest ensemble model employs **majority voting**, where the final label is selected based on the label predicted by the majority of models in the ensemble. **Accuracy Achieved:** 80%

#### Key Insight
- **Limitation:** The accuracy can decrease when the individual model predictions diverge significantly. 
- **Reason:** Higher divergence reduces the likelihood of the ensemble aligning with the true label.

#### Example Scenario
- **Prediction by Models:**
  - 3 models predict **positive**.
  - 3 models predict **negative**.
- **Ensemble Decision:** Chooses either **positive** or **negative** randomly.
- **Ideal Answer:** **Neutral** (missed due to lack of consensus).

