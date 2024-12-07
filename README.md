# Toxic Comment Classification

## Dataset
This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset, which consists of comments from a public discussion platform. The goal is to classify each comment into multiple categories indicating the level of toxicity, including:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The dataset contains both toxic and non-toxic comments labeled for each of these categories.

## Model and Architecture
The model is a deep learning architecture based on **TensorFlow** and **Keras**. It performs text classification using a **TextVectorization** layer, which applies the **TF-IDF** (Term Frequency-Inverse Document Frequency) method to convert raw text into numerical vectors. The architecture consists of:

1. **Text Vectorization Layer**: Converts input text into vectors using TF-IDF with bigrams.
2. **Dense Layers**: 
   - First Dense layer: 256 units, ReLU activation, L2 regularization.
   - Second Dense layer: 32 units, ReLU activation, L2 regularization.
3. **Output Layer**: Sigmoid activation for multi-label classification (6 classes).

The model uses the **Adam** optimizer, **binary cross-entropy** loss, and is evaluated with **categorical accuracy** and **AUC**.

## Classification Reports

Below are the classification results for each of the six labels in the dataset:

| **Label**        | **Precision** | **Recall** | **F1-score** | **Support** |
|------------------|---------------|------------|--------------|-------------|
| **toxic**        | 0.92          | 1.00       | 0.96         | 31915       |
| **severe_toxic** | 0.99          | 0.00       | 0.00         | 31915       |
| **obscene**      | 0.96          | 1.00       | 0.98         | 31915       |
| **threat**       | 1.00          | 0.00       | 0.00         | 31915       |
| **insult**       | 0.96          | 1.00       | 0.98         | 31915       |
| **identity_hate**| 0.99          | 0.00       | 0.01         | 31915       |

**Notes**:
- Precision: How many of the predicted positives are actually positive.
- Recall: How many of the actual positives are correctly identified.
- F1-score: A balanced measure between precision and recall.
