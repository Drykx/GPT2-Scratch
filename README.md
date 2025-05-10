# GPT2-Scratch

## 📚 Overview

This project is a personal implementation of GPT-2 from scratch using PyTorch. It reflects my effort to gain a deeper understanding of the inner workings of large language models (LLMs).

To support this learning journey, I relied on two excellent resources:

- 📖 [*Deep Learning: Foundation and Concepts* by Christopher M. Bishop](https://www.bishopbook.com/) – Chapters 6–13
- 🎥 [Neural Networks: Zero to Hero by Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)

In my experience, these are among the most comprehensive and effective learning materials available. I highly recommend them to both beginners and intermediate learners. Even if you’ve previously worked with LLMs, Karpathy’s clear explanations and hands-on approach offer valuable insights that bridge theory and implementation.

## 🧠 Goals

- Deepen understanding of transformer architecture to explore improvements  
- Rebuild GPT-2 components to sharpen coding skills  
- Practice turning research papers into code  
- Learn best practices from expert implementations  
- Strengthen debugging and scaling from first principles  

## 🔧 Tools Used

- Python & PyTorch  
- Hugging Face Datasets (for experimentation)  
- AWS (for compute resources)  

## 🗂️ Structure Repo

```
Drykx/
├── Makemore/        # Character-level bigram model with scalable extensions to reduce training loss
├── Tokenizer/       # Custom Byte Pair Encoding (BPE) tokenizer implementation
├── GPT/             # Lightweight GPT model for generating French poetry using the custom tokenizer
├── GP2T/            # Scalable GPT-2-style model trained on 10B tokens; evaluated on HellaSwag using GPT-2 tiktoken encoding
├── requirements.txt # Python dependencies and environment setup
```

## 🚀 Getting Started

```
pip install -r requirements.txt
```
