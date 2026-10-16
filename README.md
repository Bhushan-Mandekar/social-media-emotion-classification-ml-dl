# 🧠 Social Media Usage and Emotional Well-Being Classification Using Machine Learning & Deep Learning

### Capstone Project 3 | End-to-End Classification and Streamlit Deployment

**Developed by:** Bhushan Mandekar  
**Project Type:** Machine Learning & Deep Learning — Multiclass Classification  
**Programming Language:** Python  
**Development Environment:** Visual Studio Code  
**Dataset Source:** [Kaggle — Social Media Usage and Emotional Well-Being](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being)

---

## 📌 Project Overview

Social media has become an integral part of everyday life, influencing communication, engagement, and emotional experiences.

This project develops a multiclass classification system to predict a user's dominant emotional state based on demographic information, social media platform usage, and daily engagement patterns.

The project implements an end-to-end data science workflow, including data cleaning, exploratory data analysis, feature preprocessing, Machine Learning and Deep Learning model development, evaluation, and interactive deployment using Streamlit.

**The application is intended for educational and research purposes and is not a clinical mental-health assessment tool.**

## 🎯 Project Objectives

- Analyze social media usage and engagement patterns.
- Identify and address missing values, duplicate records, and inconsistent data.
- Perform exploratory data analysis to understand feature distributions and relationships.
- Develop and compare multiple Machine Learning classification algorithms.
- Implement a Deep Learning neural network for multiclass emotion classification.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Select the final model using validation performance.
- Build an interactive Streamlit application for emotion prediction.

## 📊 Dataset Information

**Dataset:** Social Media Usage and Emotional Well-Being  
**Source:** Kaggle  
**Task:** Multiclass Classification

### Input Features

| Feature | Description |
|---|---|
| Age | Age of the user |
| Gender | User's gender |
| Platform | Social media platform |
| Daily_Usage_Time (minutes) | Daily social media usage duration |
| Posts_Per_Day | Number of posts per day |
| Likes_Received_Per_Day | Daily likes received |
| Comments_Received_Per_Day | Daily comments received |
| Messages_Sent_Per_Day | Daily messages sent |

### Target Variable

**Dominant_Emotion**

The classification system predicts one of six emotional categories:

- Happiness
- Sadness
- Anxiety
- Anger
- Boredom
- Neutral

### Data Quality and Preparation

The original supplied datasets contained overlapping observations and inconsistent user identifiers.

After cleaning and deduplication, the analysis identified **261 unique feature-and-target observations**.

A stratified 70% training, 15% validation, and 15% testing split was prepared for model development.

The final splits were checked for identical feature-and-target records to reduce observation-level data leakage.

**Limitation:** User-level independence could not be established reliably, and the relatively small dataset limits the generalizability of the evaluation results.

## 🔍 Exploratory Data Analysis

Exploratory analysis includes:

- Dominant emotion distribution
- Gender distribution
- Age distribution
- Social media platform distribution
- Daily usage duration across emotional categories
- Correlation analysis of numerical engagement features

These visualizations help identify patterns and relationships in the training data.

## ⚙️ Data Preprocessing

The preprocessing workflow includes:

1. Dataset loading and malformed-record repair.
2. Data type correction and categorical standardization.
3. Missing-value and invalid-value handling.
4. Duplicate and overlapping-record investigation.
5. Stratified dataset splitting.
6. Numerical feature imputation and standardization.
7. Categorical feature imputation and one-hot encoding.
8. Target emotion label encoding.

The preprocessing pipeline is fitted exclusively on the training data and subsequently applied to validation and testing data.

## 🤖 Machine Learning Models

The following algorithms are implemented and compared:

| Model | Purpose |
|---|---|
| Dummy Classifier | Majority-class baseline |
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble classification |
| Extra Trees | Randomized tree ensemble |
| Support Vector Machine | Nonlinear classification |
| K-Nearest Neighbors | Distance-based classification |

Model comparison is performed on the validation dataset, with **macro F1-score** used for model selection.

## 🧠 Deep Learning Model

A feedforward Artificial Neural Network (ANN) is implemented using TensorFlow and Keras.

### Neural Network Architecture

| Layer | Configuration |
|---|---|
| Input | Preprocessed numerical and categorical features |
| Dense Layer 1 | 64 neurons, ReLU |
| Dropout Layer 1 | 30% |
| Dense Layer 2 | 32 neurons, ReLU |
| Dropout Layer 2 | 20% |
| Output | 6 neurons, Softmax |

**Training configuration:**

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Maximum Epochs: 100
- Batch Size: 16
- Early Stopping: Enabled
- Model Checkpointing: Enabled

Training and validation accuracy and loss are visualized to examine model behavior.

## 📈 Model Evaluation

Model performance is evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Macro F1-score
- Classification Report
- Confusion Matrix

The final model is selected based on validation macro F1-score and evaluated on the held-out test dataset.

**Note:** Actual performance metrics should be taken from `outputs/final_model_evaluation.csv`. No accuracy or F1-score values are assumed in this README.

## 🌐 Streamlit Web Application

The project includes an interactive web application built using Streamlit.

### Application Features

- User-friendly interface for entering social media usage details.
- Prediction of the dominant emotion category.
- Model-estimated class probabilities, where supported.
- Integration with saved preprocessing and model artifacts.
- Educational disclaimer explaining prediction limitations.

### User Inputs

The application accepts age, gender, social media platform, daily usage time, posts, likes, comments, and messages.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Model Serialization | Joblib, Keras |
| Web Application | Streamlit |
| Development Environment | Visual Studio Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/social-media-emotion-classification-ml-dl.git
cd social-media-emotion-classification-ml-dl
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify Model Artifacts

Ensure that the `outputs/` directory contains:

- `preprocessor.pkl`
- `emotion_label_encoder.pkl`
- `model_metadata.json`
- `final_ml_model.pkl` or `final_dl_model.keras`

The final model file must match the model type recorded in `model_metadata.json`.

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, typically:

`http://localhost:8501`

## 📂 Repository Structure

```text
social-media-emotion-classification-ml-dl/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── outputs/
└── images/
```

## ⚠️ Project Limitations

- The modeling dataset contains only 261 unique observations.
- Some original records have inconsistent or repeated user identifiers.
- Observation-level deduplication does not establish user-level independence.
- The small test set can produce unstable performance estimates.
- Predicted emotions are based on dataset patterns, not direct psychological measurements.
- The application is not intended for diagnosis, treatment, or clinical decision-making.

## 🔮 Future Enhancements

- Expand the dataset with more diverse and independently collected observations.
- Improve user-level data separation and evaluation.
- Perform systematic hyperparameter optimization.
- Evaluate probability calibration.
- Apply explainability methods such as SHAP.
- Compare additional neural network architectures.
- Deploy and monitor the application using a suitable hosting platform.

## 👨‍💻 Developer

**Bhushan Mandekar**

Capstone Project 3 — Machine Learning and Deep Learning

**Project:** Social Media Usage and Emotional Well-Being Classification

**Development Tools:** Python, VS Code, Jupyter Notebook, Scikit-learn, TensorFlow, and Streamlit

## 📚 References

- [Kaggle Dataset — Social Media Usage and Emotional Well-Being](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**⭐ This project demonstrates an end-to-end Machine Learning and Deep Learning workflow, from raw data preprocessing to interactive model deployment.**
