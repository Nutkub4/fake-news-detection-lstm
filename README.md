# Fake News Detection Using LSTM

## Project Overview

This project develops a deep learning-based **Fake News Detection system** using a **Long Short-Term Memory (LSTM)** neural network for binary text classification.

The objective is to classify news articles into two categories:

* **Fake News**
* **Real News**

The project applies Natural Language Processing (NLP) techniques including text preprocessing, tokenization, sequence padding, and deep learning-based text classification.

---

## Dataset

Dataset source:

Kaggle Fake News Dataset

The dataset contains two files:

* `fake.csv` - Fake news articles
* `true.csv` - Real news articles

The dataset is transformed into a binary classification problem:

| Label | Class     |
| ----- | --------- |
| 0     | Fake News |
| 1     | Real News |

---

## Project Objectives

The main objectives of this project are:

* Analyze fake and real news article distributions
* Clean and preprocess text data
* Convert text into numerical sequences for deep learning
* Build an LSTM-based text classification model
* Evaluate model performance using classification metrics
* Create an inference pipeline for predicting new unseen news articles

---

# Project Workflow

```
01_EDA.ipynb
        |
        ↓
02_Data_Preprocessing.ipynb
        |
        ↓
03_Feature_Engineering.ipynb
        |
        ↓
04_Model_Training.ipynb
        |
        ↓
05_Model_Evaluation.ipynb
        |
        ↓
06_Inference.ipynb
```

---

# Project Structure

```
fake-news-detection-lstm/

│
├── data/
│   ├── raw/
│   │   ├── fake.csv
│   │   └── true.csv
│   │
│   └── processed/
│       └── cleaned_news.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Inference.ipynb
│
├── models/
│   ├── tokenizer.pkl
│   └── lstm_v2.keras
│
├── outputs/
│   └── metrics/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Methodology

## 1. Exploratory Data Analysis (EDA)

Notebook:

```
01_EDA.ipynb
```

Performed:

* Dataset inspection
* Missing value analysis
* Class distribution analysis
* Text length exploration
* Basic visualization of news characteristics

---

## 2. Data Preprocessing

Notebook:

```
02_Data_Preprocessing.ipynb
```

Steps:

* Combine fake and real news datasets
* Create classification labels
* Clean text data
* Remove unnecessary characters
* Normalize text format

Output:

```
data/processed/

└── cleaned_news.csv
```

---

## 3. Feature Engineering

Notebook:

```
03_Feature_Engineering.ipynb
```

Steps:

* Split dataset into training and testing sets
* Tokenize text data
* Convert words into numerical sequences
* Apply padding

Configuration:

```python
MAX_WORDS = 20000
MAX_LENGTH = 300
```

Outputs:

```
models/

└── tokenizer.pkl
```

and processed numerical features:

```
X_train.npy
X_test.npy
y_train.npy
y_test.npy
```

---

# Model Architecture

Notebook:

```
04_Model_Training.ipynb
```

The LSTM model architecture:

```
Input Text

      ↓

Tokenizer

      ↓

Sequence Padding
(MAX_LENGTH = 300)

      ↓

Embedding Layer

      ↓

LSTM Layer
(128 units)

      ↓

Dropout
(0.5)

      ↓

Dense Layer

      ↓

Sigmoid Output

      ↓

Fake / Real Classification
```

---

# Model Training

The model is trained using:

* Optimizer: Adam
* Loss Function: Binary Cross Entropy
* Evaluation Metric: Accuracy

Training techniques:

* Validation split
* Early stopping
* Model checkpointing

The trained model is saved as:

```
models/

└── lstm_v2.keras
```

---

# Model Evaluation

Notebook:

```
05_Model_Evaluation.ipynb
```

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

Evaluation results are stored in:

```
outputs/

└── metrics/
```

---

# Inference

Notebook:

```
06_Inference.ipynb
```

The inference pipeline allows users to input a new news article:

```
New News Text

      ↓

Tokenizer

      ↓

Sequence Conversion

      ↓

Padding

      ↓

LSTM Model

      ↓

Prediction Probability

      ↓

Fake News / Real News
```

Example output:

```
Prediction: Real News

Confidence: 96.5%
```

---

# Installation and Usage

## Clone Repository

```bash
git clone https://github.com/Nutkub4/fake-news-detection-lstm

cd fake-news-detection-lstm
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Notebooks

Execute notebooks in order:

```
01_EDA.ipynb
02_Data_Preprocessing.ipynb
03_Feature_Engineering.ipynb
04_Model_Training.ipynb
05_Model_Evaluation.ipynb
06_Inference.ipynb
```

---

# Technologies Used

## Programming Language

* Python

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-learn

## Deep Learning

* TensorFlow
* Keras

## Visualization

* Matplotlib
* Seaborn

## Development Tools

* Jupyter Notebook
* Git
* GitHub

---

# Future Improvements

Possible improvements:

* Compare LSTM performance with Transformer-based models:

  * BERT
  * DistilBERT
  * RoBERTa

* Add pretrained word embeddings:

  * Word2Vec
  * GloVe

* Add model explainability using:

  * SHAP
  * LIME

---

# Author

Watchapon Wongapinya

Data Science / Machine Learning Project