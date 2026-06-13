# LLM from Scratch

A hands-on implementation of a GPT-style Large Language Model (LLM) built from scratch using PyTorch. This project recreates the fundamental building blocks behind modern decoder-only Transformer architectures, focusing on both implementation and conceptual understanding through detailed code comments and experiments.

## Project Overview

This project explores the complete language modeling pipeline, starting from tokenization and data preparation to Transformer architecture implementation, training, evaluation, and autoregressive text generation.

The codebase includes explanatory comments and experiments designed to understand the mathematical and architectural foundations of Large Language Models rather than simply using pretrained models.

## Features

### Data Processing

* GPT-2 Byte Pair Encoding (BPE) tokenization using `tiktoken`
* Custom Dataset and DataLoader implementations
* Sliding-window sequence generation for next-token prediction
* Train-validation dataset splitting

### Transformer Architecture

* Token Embeddings
* Positional Embeddings
* Layer Normalization
* GELU Activation Functions
* Feed Forward Networks (FFN)
* Residual Connections
* Scaled Dot-Product Attention
* Causal (Masked) Self-Attention
* Multi-Head Attention
* Transformer Blocks
* GPT-style Decoder-Only Architecture

### Training & Evaluation

* Cross-Entropy Loss
* Perplexity Evaluation
* AdamW Optimization
* Training and Validation Loops
* Loss Tracking
* Training Curve Visualization

### Text Generation

* Autoregressive Next-Token Prediction
* Greedy Decoding
* Temperature Sampling
* Top-k Sampling
* Multinomial Sampling
* Controlled Text Generation

## Project Structure

```text
LLM-from-Scratch/
│
├── tokenizer_class.py      # Basic tokenizer implementation
├── bpe.py                  # Byte Pair Encoding concepts
├── createingITpairs.py     # Input-target pair creation
├── impDataLoader.py        # Dataset and DataLoader pipeline
├── attentionmech.py        # Self-attention implementation
├── maskedAttentionMech.py  # Causal masking
├── multiheadAttention.py   # Multi-head attention
├── GptArchiImp.py          # GPT architecture implementation
├── ModelEvalve.py          # Evaluation, perplexity, validation
├── train.py                # Training loop and optimization
├── sampling.py             # Temperature and top-k sampling
│
├── the-verdict.txt         # Training corpus
├── requirements.txt
└── README.md
```

## Concepts Explored

This project was built as a learning-oriented implementation to gain a deeper understanding of:

* Transformer Architectures
* Attention Mechanisms
* Language Modeling
* Next-Token Prediction
* Decoder-Only Models
* Training Dynamics of LLMs
* Sampling Strategies for Text Generation
* Evaluation Metrics such as Cross-Entropy Loss and Perplexity

## Technologies Used

* Python
* PyTorch
* NumPy
* tiktoken
* Matplotlib

## Dataset

The project uses *The Verdict* by Edith Wharton as a sample text corpus for language modeling experiments. The text is tokenized and converted into training sequences using a sliding-window approach.

## Learning Focus

Unlike production-ready LLM frameworks, this project prioritizes transparency and understanding. Many files contain explanatory comments, intermediate experiments, and implementation notes that document the reasoning behind architectural choices and training procedures.

## Future Improvements

* Learning Rate Scheduling
* Weight Tying
* Mixed Precision Training
* Larger Training Corpora
* Model Checkpoint Management
* Fine-Tuning Workflows

## Acknowledgements

This project was inspired by the concepts presented in *Build a Large Language Model (From Scratch)* by Sebastian Raschka and was developed as a practical exploration of modern Transformer-based language models.
