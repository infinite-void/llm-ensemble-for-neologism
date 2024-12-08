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
