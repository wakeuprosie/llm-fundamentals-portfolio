# llm-fundamentals-portfolio

This is a portfolio demonstrating fundamentals knowledge of LLM Infra and Training.

Scaling Large Language Models requires an understanding of distributed systems, hardware economics, and scaling laws. While application-layer AI often focuses on model performance metrics like latency and prompt adherence, Core AI infrastructure requires systems-level approaches to unblocking compute bottlenecks.

This repository serves as a study of LLM infrastructure—built from first principles. Rather than relying on high-level abstractions, these artifacts deconstruct the pipeline from token compression algorithms to multi-node GPU synchronization.

## Portfolio Artifacts

### 01_llm_system_map.md

This file provides a high-level overview of LLM architecture.

### 02_basic_tokenizer

This is a notebook that demonstrates how a basic tokenizer works for encoding and decoding text in LLMs. It includes a custom implementation of a Byte-Pair Encoding Tokenizer in Python.

### 03_transformer_architecture

This notebook outlines the trasnformer architecture and attention mechainism that powers modern LLMs.

### 04_llm_training_pipeline

This notebook outlines the key components and considerations for training an LLM, with a focus on optimization and efficiency. Included is a from-scratch implementation of a small scale LLM using TensorFlow, and iterations to increase its scale to a GPT-2 model, including data parallelism and multi-GPU usage.