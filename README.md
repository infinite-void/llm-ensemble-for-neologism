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
finetune the model to perform well on Neologisms. The raw GPT-2 model was able to achieve only 48% accuracy with sentiment analysis on neologisms. With task specific full finetuning for sentiment analysis an accuracy of upto 75% was possible. Then we finetuned the model further to perform better on neologisms. Through this we were able to achieve an overall accuracy of 83% on Neologisms with full-finetuning.

1. finetuning_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task
2. finetune_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis to perform well on sentences with neologisms
3. lora_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task using the Low Rank Adaptation (LoRA) finetuning
4. lora_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis (using LoRA method) to perform well on sentences with neologisms using LoRA finetuning
5. ada_lora_gpt2_sentiment.ipynb - Finetuning the base gpt2 model to perform well on the sentiment analysis task using the Adaptive Budget Low Rank Adaptation (AdaLoRA) finetuning
6. ada_lora_gpt2_neologism.ipynb - Finetuning gpt2 for sentiment analysis (using AdaLoRA method) to perform well on sentences with neologisms using AdaLoRA finetuning


---

## Finetuning the BERTweet Model

### 1. **Full Finetune against Neologisms Dataset:** 
The model performs relatively poorly, correctly identifying sentiments less than half the time (43% accuracy). While it is slightly better at being precise (54%), it fails to identify many of the correct sentiments (43% recall). This suggests that the basic fine-tuning is not sufficient to handle the complexity of neologisms in sentiment analysis.​

### 2. **LoRA Fine-tuning against tweet_eval and Neologisms Dataset:** 
Adding LoRA improves the model’s ability to identify sentiments correctly (51% accuracy) with reduced training time and hardware usage. Essentially, the model is better at capturing more true cases but also makes more errors, indicating that it trades off precision for recall.​

### 3. **DoRA Fine-tuning against neologisms only:** 
This model demonstrates significant improvement, achieving high accuracy (79%) while maintaining a good balance between precision (72%) and recall (79%) implying that the model is both accurate and consistent in detecting sentiments from neologisms.​

### 4. **DORA Fine-tuning against Twitter_eval & Neologisms:** 
This yielded the most accurate results (89% accuracy). The model is first finetuned against twitter_eval dataset for sentimental analysis improvement. Then, further finetuned aginst neologisms specific dataset using DoRA improved the results, but with a tradeoff in training time.​

---


## Ensemble Approach
Each model is trained individually on the neologism dataset and contributes to the final prediction in the deployed application. The ensemble aggregates outputs from all models to improve accuracy and robustness.
We perform three ensemble techniques.

### Majority Voting: Simplest Ensemble Model
The simplest ensemble model employs **majority voting**, where the final label is selected based on the label predicted by the majority of models in the ensemble. **Accuracy Achieved:** 80%. Code is available in ensemble_learner_voting.ipynb

#### Key Insight
- **Limitation:** The accuracy can decrease when the individual model predictions diverge significantly. 
- **Reason:** Higher divergence reduces the likelihood of the ensemble aligning with the true label.

#### Example Scenario
- **Prediction by Models:**
  - 3 models predict **positive**.
  - 3 models predict **negative**.
- **Ensemble Decision:** Chooses either **positive** or **negative** randomly.
- **Ideal Answer:** **Neutral** (missed due to lack of consensus).

---

### Weighted Average Ensemble Approach
Multiple models make predictions independently, a model’s prediction is assigned a **weight**, typically based on the model’s **performance** and **reliability**. The output of all models are converted to a scale of 0-1 and the end result is returned.
#### Key Insights
- **Strength:** Improves Robustness\
  **Explanation:** Weighted averaging reduces the impact of individual model errors by combining predictions, leading to a more balanced and reliable outcome.

- **Limitation:** Does Not Consider Individual Model Confidence\
  **Explanation:** Weighted averaging applies fixed weights to model predictions regardless of how confident each model is about a specific input, potentially leading to less reliable decisions.


#### Example Scenario
**Prediction by Models:**
    - Model A predicts positive with 70% probability.
    - Model B predicts positive with 60% probability.
    - Model C predicts positive with 80% probability.\
**Ensemble Decision:** 
The weighted average combines these predictions to confidently select positive, reducing the risk of relying on a single model’s error.\
**Ideal Answer:** 
A robust prediction of positive, leveraging the strengths of multiple models to improve reliability.

---
### Stacking Ensemble
We build a meta model with multlayer perceptron to build a stacking ensemble. This achieved the highest possible accuracy of 87%. The code can be found in the ensemble_learner.ipynb. This model was able to capture the non-linear relations between the strengths and weaknesses of the underlying LLM models. This was even able to crack sentences with slang terms. 
---

### Link to open-source packages and models used
- DistillBERT - https://huggingface.co/docs/transformers/en/model_doc/distilbert
- GPT-2 - https://huggingface.co/openai-community/gpt2
- BERTweet - https://huggingface.co/vinai/bertweet-base
- FinBERT - https://huggingface.co/ProsusAI/finbert
- RoBERTA - https://huggingface.co/docs/transformers/en/model_doc/roberta
- VADER - https://github.com/cjhutto/vaderSentiment
- LoRA - https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- AdaLoRA - https://huggingface.co/docs/peft/main/en/package_reference/adalora
- DoRA - https://github.com/NVlabs/DoRA
