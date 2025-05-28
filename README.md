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

The PyTorch/CUDA versions listed are for Windows, so these files will only work for Windows users. If you're using Linux or macOS, visit https://pytorch.org/get-started/locally/ to find the appropriate version for your system.

To install all necessary dependencies, run:
```
pip install -r requirements.txt
```
## Running Training on the Cloud

If you'd like to replicate the training process, you can choose from cloud platforms like **AWS**, **Azure**, or **Google Cloud**. I recommend **Google Cloud Platform (GCP)** because it offers a **$300 credit** for new users.

### Here’s how to set up your training environment on Google Cloud:

#### 1. Create a new project
Log in to [Google Cloud Console](https://console.cloud.google.com/) and create a new project for your training workflow.

#### 2. Set up a VM Instance
Go to the **Compute Engine** section and create a new **VM instance**. Make sure to choose a machine type with sufficient CPU and RAM for your needs.

#### 3. Request access to GPUs
You’ll need to request a GPU quota:

- Navigate to **IAM & Admin > Quotas**
- Filter by **NVIDIA GPUs**
- Select the desired region and request a quota increase  
⚠️ *This process can take up to 48 hours.*
#### 4. Create a VM with a GPU (after approval)

Once your GPU quota is approved, create a VM with an attached GPU (e.g., **NVIDIA Tesla T4** or **V100**). Choose a compatible image such as the **Deep Learning VM** (which comes with pre-installed PyTorch and CUDA) to save setup time.

#### 5. SSH into your VM and set up your environment

- Clone your repository  
- Set up a virtual environment *(optional but recommended)*
- Download the dataset with:

```bash
python3 fineweb.py
```
#### 6. Train

- Training time and performance depend on the number of GPUs and your budget.
- Using 4 GPUs, you can expect reasonable results within about an hour.
