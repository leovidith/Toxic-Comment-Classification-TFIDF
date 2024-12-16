# Toxic Comment Classification

## Overview
This project tackles the **Jigsaw Toxic Comment Classification Challenge**, where the goal is to classify each comment into various categories indicating the level of toxicity. These categories include:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The dataset contains both toxic and non-toxic comments, and the task is to classify them into one or more of the aforementioned labels.

## Results

### Model and Architecture
The model is built using **TensorFlow** and **Keras**. It uses a deep learning architecture to perform text classification, employing a **TextVectorization** layer that converts raw text into numerical vectors using the **TF-IDF** (Term Frequency-Inverse Document Frequency) method. The architecture consists of the following layers:

1. **Text Vectorization Layer**: 
   - Converts input text into numerical vectors using **TF-IDF** with bigrams.
2. **Dense Layers**: 
   - First Dense layer: 256 units, **ReLU** activation, and **L2 regularization**.
   - Second Dense layer: 32 units, **ReLU** activation, and **L2 regularization**.
3. **Output Layer**: 
   - **Sigmoid** activation for multi-label classification (6 classes).

The model is optimized using the **Adam** optimizer, and the loss function is **binary cross-entropy**. It is evaluated using **categorical accuracy** and **AUC** metrics.

### Classification Results:

| **Label**        | **Precision** | **Recall** | **F1-score** | **Support** |
|------------------|---------------|------------|--------------|-------------|
| **toxic**        | 0.92          | 1.00       | 0.96         | 31915       |
| **severe_toxic** | 0.99          | 0.00       | 0.00         | 31915       |
| **obscene**      | 0.96          | 1.00       | 0.98         | 31915       |
| **threat**       | 1.00          | 0.00       | 0.00         | 31915       |
| **insult**       | 0.96          | 1.00       | 0.98         | 31915       |
| **identity_hate**| 0.99          | 0.00       | 0.01         | 31915       |

### Notes on Classification Metrics:
- **Precision**: Measures the proportion of positive predictions that are actually correct.
- **Recall**: Measures the proportion of actual positives that are correctly identified.
- **F1-score**: A balanced measure of precision and recall, especially useful for imbalanced datasets.
  
## Features

- **Text Vectorization**: The **TF-IDF** method with bigrams allows the model to capture important word pairs and their significance in the comment, improving the feature representation.
- **Regularization**: **L2 regularization** helps prevent overfitting by penalizing large weights.
- **Multi-label Classification**: Using **sigmoid** activation allows for predicting multiple labels per comment.

## Sprints

### Sprint 1: Data Preprocessing
- **Deliverable**: Preprocess text by tokenizing, converting to lowercase, and removing stop words.

### Sprint 2: Model Architecture Design
- **Deliverable**: Build and configure the model with the TextVectorization layer, dense layers, and output layer.

### Sprint 3: Training and Evaluation
- **Deliverable**: Train the model with **binary cross-entropy** loss and **Adam** optimizer. Evaluate using **precision**, **recall**, and **F1-score** metrics.

## Conclusion

This model successfully classifies comments into multiple categories of toxicity, achieving high precision for most categories. However, the low recall for **severe_toxic**, **threat**, and **identity_hate** suggests the need for further model improvements, such as rebalancing the dataset or using more advanced techniques like class-weighting or ensemble methods.
