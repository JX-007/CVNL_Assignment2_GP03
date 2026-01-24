# CVNL Assignment 2
## AI Prototype for Singapore Changi Airport Operations

### Overview
This project presents an AI prototype developed using Python and PyTorch to support key operational and service workflows at Singapore Changi Airport. The solution integrates two core deep learning capabilities:

- Convolutional Neural Networks (CNNs) for visual classification tasks
- Recurrent Neural Networks (RNNs) for text sequence classification tasks

The objective is to demonstrate how artificial intelligence can enhance passenger experience, operational efficiency, and service quality in a realistic airport environment.

---

### **Objectives**
**1. Model Development**

CNN Model
- Design and train a CNN in PyTorch to classify images for your chosen track
- Apply suitable preprocessing (e.g., resizing, normalization, augmentation)
- Improve performance through tuning (hyperparameters, architecture depth, regularization, augmentation)

RNN Model
- Design and train a RNN using PyTorch to tokenize text, build vocabulary (or use embeddings), and create fixed-length sequences.
- Pad or truncate sequences to ensure uniform length.
- Train an RNN classifier (e.g., vanilla RNN / GRU / LSTM; choose one and justify).
- Improve the model performance using suitable techniques such as tuning appropriate hyperparameters of the model, and document improvements.

### **2. Evaluation** ###
- Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
- Use confusion matrix and misclassification analysis.
---
### Key Use Cases
1. Passenger Assistance
- Classifying passenger messages by intents such as for directions, baggage issues, or special assistance requests.
- Automating routing of queries to appropriate airport services.

2. Operational Support
- Identifying aircraft types, luggage, and airport equipment from images.
- Supporting ground staff and asset tracking operations.

3. Service Recovery (OPTIONAL)*
- Analysing passenger feedback and support messages.
- Detecting sentiment or emotions (e.g., satisfied, frustrated, urgent) to prioritise responses.
---
### Project Structure
- data/                -- (contain datasets)
- models/              -- (contains model architecture)
- AI_Airport_Prototype.ipynb
- README.md
---
## Dataset Source
- FGVC-Aircraft Dataset: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- ATIS (Air Travel Information System) Dataset: https://huggingface.co/datasets/DeepPavlov/snips

## Technologies Used
- Google Colab (Notebook)
- Python
- PyTorch
- Numpy
- Pandas
- Matplotlib (for visualisation)

## Installation
1. ...