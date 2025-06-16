# Introduction
**Machine Learning (ML)** is a subfield of **Artificial Intelligence (AI)** that focuses on building systems that can learn from and make decisions based on data—without being explicitly programmed. It enables machines to perform tasks autonomously and improve their performance over time as they are exposed to more data.

## Relationship Between AI, Machine Learning, and Deep Learning

### AI (Artificial Intelligence)
- AI refers to any technique that enables machines to **mimic human intelligence**.
- This includes rule-based systems like **if-then logic**, **decision trees**, and **learning algorithms** like **machine learning** and **deep learning**.

### Machine Learning
- ML is a **subset of AI** that uses **statistical methods** to enable machines to **learn patterns** from data and improve their performance with experience.
- It includes both **traditional algorithms** (like linear regression, decision trees) and **deep learning**.

### Deep Learning
- Deep Learning is a **subset of ML** based on **artificial neural networks with multiple layers** (deep neural networks).
- It excels at tasks such as **image recognition**, **speech recognition**, **natural language understanding**, and more.
- Unlike traditional ML, deep learning models can automatically extract features from raw data with **minimal human intervention**.

![alt text](tx32puqCzYlG832gHMhinF3VkJU.avif)  
**Figure: Relationship between AI, ML, and Deep Learning**

---

## What is Machine Learning?
Machine Learning is a way to build AI systems that **learn from data**, rather than following hardcoded rules. ML systems **generalize** from past experience (data) to make **future predictions** or decisions.

---

## Types of Machine Learning

Machine Learning can be broadly classified into the following categories:

1. **Supervised Learning**  
2. **Unsupervised Learning**  
3. **Reinforcement Learning**  
4. **Self-Supervised Learning** *(Emerging technique used in large language models like GPT)*  
5. **Generative AI** *(Special focus on content generation using models like GANs or Transformers)*

---

## 1. Supervised Learning

In supervised learning, the model learns from a **labeled dataset**, where the input data is paired with the correct output. The goal is to learn a mapping from inputs to outputs.

**Applications:**
- Spam email detection
- Weather forecasting
- Price prediction
- Medical diagnosis

### Use Cases
- **Regression**
- **Classification**

---

### Regression

A regression model predicts **continuous numeric values**.

| **Scenario**         | **Possible Input Data**                                                                                                                                      | **Prediction (Output)**                       |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| House price prediction | Square footage, zip code, number of rooms, interest rate, market trends                                                                 | Future house price (in currency)               |
| Traffic estimation     | GPS location, time of day, weather, historic data                                                                                   | Estimated travel time (minutes/seconds)       |

---

### Classification

Classification models predict which **category or class** an input belongs to.

#### Examples:
- Is this email **spam** or **not spam**?
- Does this image contain a **cat**, **dog**, or **neither**?

#### Types of Classification:

##### 1. Binary Classification
- Only **two** classes.
- Example: Fraud detection (fraud or not fraud)

##### 2. Multiclass Classification
- **More than two** possible categories.
- Example: Handwritten digit recognition (0–9)

---

## 2. Unsupervised Learning

In unsupervised learning, the model works with **unlabeled data** and tries to find **hidden patterns or structures**.

### Examples:
- **Clustering**: Grouping similar items (e.g., customer segmentation)
- **Dimensionality Reduction**: Simplifying data (e.g., PCA)

---

## 3. Reinforcement Learning

In reinforcement learning (RL), an agent learns to make decisions by interacting with an environment and receiving **rewards** or **penalties**.

### Key Concepts:
- **Agent**: The learner
- **Environment**: Where the agent operates
- **Reward**: Feedback signal for actions taken

### Example Applications:
- Game AI (like AlphaGo)
- Robotics
- Self-driving cars

---

## 4. Self-Supervised Learning (SSL)

- A bridge between supervised and unsupervised learning.
- The model **generates labels from the input data itself**.
- Used heavily in **natural language processing** (e.g., BERT, GPT) and **vision models**.

---

## 5. Generative AI

Generative AI focuses on systems that **create new content** such as images, text, audio, or video.

### Examples:
- **Text generation**: ChatGPT, Bard
- **Image generation**: DALL·E, Midjourney
- **Music generation**: AIVA, Amper Music

### Common Technologies:
- **GANs (Generative Adversarial Networks)**
- **VAEs (Variational Autoencoders)**
- **Transformers**

---
