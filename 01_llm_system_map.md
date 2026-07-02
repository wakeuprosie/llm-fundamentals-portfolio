# LLM System Map
This document outlines the high-level architecture of an LLM.

**In summary, the high-level LLM system is structured like this:**

Data set → Neural Net Training → Base model → Assistant Layer Training → Assistant

The starting data set consists of internet data in text form. The final product, the assistant, is what most of us are used to interacting with, i.e. ChatGPT, Gemini.

Next, I will go into key details of each system layer.

## Pre-training

As we discussed, the source knowledge base of LLMs are text data found on the internet. This data is collected and then optimized for training and this overall phase is known as **pre-training**. 

A few of the key steps in pre-training include:

1. Collecting data. One method is through web crawling. You start with a seed URL and crawl the internet and its associated web pages to collect text data.  
2. Filtering. A couple common reasons for filtering are:  
   1. Adversarial content which you don’t want included in your LLMs knowledge base, such as hate language or spam.  
   2. Language. You can optimize your LLM to have a native language. For example, the Gemini App’s native language is English as it is predominantly trained with English data.

There are several more sophisticated filtering steps which were not mentioned in this lecture. A few examples:

* Text length filters \- filtering out documents which are too short  
* Symbol-to-word ratio (e.g. $, \#, %) \- this could point to metadata or be too noisy  
* Bullet point formatting \- this could indicate a landing page or a catalog  
* De-duplication \- removing identical data for training efficiencies  
* Personally identifiable information (PII) \- this can include emails, ip addresses, and social security numbers

This demonstrates just a few of the reasons you wouldn’t want to use raw internet data as your dataset. 

The end goal of this pre-training step is to create a high quality text dataset to be used for training your base model.

In the earlier days of LLMs, it was believed that the more data you trained on, the better the quality of the LLM. However, the industry has shifted towards a quality over quantity philosophy in modern AI development.

### Tokenization

Once you have your data set, the next step is to process this data into efficient “chunks” for optimal training. As such, you take the text data in a binary format (0s and 1s), group the binaries into common sequences called bytes, and further group the bytes into common combinations of bytes which represent tokens. This step of data processing is known as **tokenization**.

**The tokenization process can be visualized like this:**

Raw internet text data → binary data (0s and 1s) → byte sequences (typically 8 binary sequences) → combinations of bytes (pairing common sequence patterns) → **tokens**

The industry standard is to use **100K byte combinations or tokens** as your “glossary”.

In summary, tokenization is the process of refining raw text data into tokens for optimized neural net training.

### Transformer Internals

At its core, LLMs take an input sequence of tokens and output a series of tokens. The input sequence length is typically 1-8000 tokens. It generates the output tokens one by one to produce the overall output sequence. In order to know how to predict what token to generate next, the model goes through a process called **transformation.**

Transformation involves a series of mathematical calculations between inputs and outputs. Through these mathematical calculations, the LLM assigns probabilities to all next possible tokens based on the inputs.

The neural net starts with an initial set of probabilities and random parameters. The parameters are then adjusted so that the probabilities associated with the next likely token match the pattern seen in the source dataset. This process is called **training**.

In summary, training a neural network means updating its parameters to produce token outputs that are consistent with the training dataset. These parameters are one of the ways LLMs compete with one another \- the better the parameters, the better the output.

Modern models typically have parameters in the order of Trillions. The latest GPT model, GPT-5, is estimated to have **10-50T parameters.**

One important note about neural net training is it requires hardware and software resources. GPUs are often highlighted as the key hardware in LLM training, but the training bottle neck is actually spread across optimizing compute, memory bandwidth, and interconnects. Hardware resources have improved significantly in efficiency over the last few years, but optimizing resources remains a critical component of neural net training.

### Inference

Lastly, using the transformer, the LLM now has the capability to generate the next best token and it continues this one by one, to produce its complete output. This step is called **inference.** LLMs are **stochastic**, so they will not produce the same next token every time for the same input.

## Post-Training

### Supervised fine tuning

So far, we covered the main pre-training steps. Collecting the data, refining it into a quality data set, and training a model to get a **base model**. The base model outputs are usually not sufficient for users as they produce outputs that appear like a continuation of internet documentation. The example used to demonstrate this in the lecture is this:   
Input: “What is 2+2”  
Base model output: “Is it 4 or 22? How do you know? What about 1+1+1+1? Is it 4 or 11? Etc.

Users today are most familiar interacting with LLMs in a conversational way. Training the model to output responses in this conversational way is done in **post-training**, in a step called **supervised-fine-tuning** **(SFT)**. Similar to pre-training, this relies on a high quality data set. Companies now use a mix of human and synthetic methods to create conversational datasets, while before they were primarily human generated.

A typical post-training dataset contains \~1M multi-turn conversations.

### Hallucinations

Once you’ve fine tuned the base model into a functional LLM assistant, a key consideration is to evaluate what it cannot do or does inaccurately in generating responses, aka when it **hallucinates**.

A few common methods used to mitigate hallucinations are:

1. **Interrogative investigation.** You interrogate the model to understand its boundaries of what it can and cannot do. One application is generating multiple responses for a single prompt for which there is a factual answer, and then evaluating how many times it answers it correctly. If the model answers this inconsistently, this affirms the model does not contain the knowledge and trains the model to answer that it doesn’t know.  
2. **Tools.** LLMs are trained on a specific set of data as its core knowledge base, but can supplement its context in real-time by providing tools for it to seek that context. For example, an LLM can use a web search tool to grab context on the internet about the topic of a prompt.  
3. **System messages.** System messages can be used to tell the model about its role and how it should behave. This message gets inserted at the beginning of a conversation to provide the model context about itself. Most recently, there was a leak of Anthropic’s Claude system prompts which showed users about its importance in influencing LLM behavior.

## Reinforcement Learning (RL)

After pre-training and post-training, there is a 3rd phase of training which uses **Reinforcement Learning.** This is considered one of the ‘newer’ ways of training (as of the time of this lecture, 2025\) and is a method by which LLMs use reasoning to produce their outputs. In reinforcement learning, the LLMs explore several ways of problem solving their task and evaluating them before it produces its answer.

## Reinforced with Human Feedback Learning (RHFL)

Finally, one other modern training step used is **Reinforced Human Feedback Learning**. This is particularly important for domains where there is no ‘factual’ answer for the model to grade itself on; for example, creating a funny joke. In RHFL, the model generates multiple outputs and associates a quality score with each. Humans then provide their stack ranking of the outputs. The model then evaluates the human’s labeling and adjusts its parameters accordingly.

## LLM Competitor Space

Based on this knowledge of the high-level LLM architecture, I wanted to better understand what angle companies are competing on with their LLMs.

Today they primarily compete on quality and efficiency. While before it was believed that the more data you train on, the better the quality of your model, the leading thinking today is to focus on quality \> quantity, through data filtering and synthetic data. Model parameters, which we discussed in the transformation step, are also considered part of your competitive edge. On model efficiency, companies are competing not just on which models are more intelligent, but which can run faster and cheaper. Lastly, post-training steps through RL and RLHF are also key contributor to competitive differentiation.

In terms of business application of LLMs, there are 4 areas where AI LLM companies are building their competitive edge. 

1. Cost. As an example, Deepseek was able to offer similar model capabilities for 90% lower pricing.  
2. Specialization. Instead of general purpose models, companies are working on specialized models for specific use cases \- like serving specific verticals and training on proprietary vertical / company data.  
3. Agentic. Adding the ‘action’ layer to the LLM.  
4. Self-hosting. Allowing companies to self host models to run on private servers so that clients have more control over the model and data flow.

This wraps up this documentation on the high-level architecture of LLM systems.
