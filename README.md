# Credit Card Default Prediction

### Name: Madhavakumar D
### Subject: ML Assignment-2

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a credit card customer is likely to default on their next month's payment.

Five classification algorithms are implemented and evaluated using the same dataset. Their performance is compared using multiple evaluation metrics.

An interactive Streamlit application is also developed to allow users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

**Dataset:** Default of Credit Card Clients

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

| Attribute | Description |
|---|---|
| Total instances | 30,000 |
| Input features | 23 |
| Target variable | `default payment next month` |
| Problem type | Binary Classification |
| Class 0 | No Default |
| Class 1 | Default |

The dataset contains information about credit card clients, including demographic details, credit limits, repayment status, bill amounts, and previous payment amounts.

The target variable indicates whether a customer defaults on their payment in the following month.

## 3. GitHub Repository Link

**Repository:** [Src Repo Link](https://github.com/Madhavakumar21/BinaryClassificationDemoMLProject)

## 4. Models Used and Performance Comparison

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.7155 | 0.6040 | 0.3684 | 0.4009 | 0.3840 | 0.1997 |
| KNN | 0.7948 | 0.7039 | 0.5561 | 0.3587 | 0.4361 | 0.3292 |
| Naive Bayes | 0.2903 | 0.7240 | 0.2340 | 0.9714 | 0.3771 | 0.1034 |
| Support Vector Machine | 0.8163 | 0.7201 | 0.6662 | 0.3399 | 0.4501 | 0.3824 |

NOTE: These values can be verified in the Output Cell of IPYNB file attached.

### Observations on Model Performance

**Logistic Regression**

Achieved an accuracy of 80.88% and the highest precision of 69.23%. However, its recall is only 24.42%, indicating that it misses a considerable number of actual defaulters.

**Decision Tree**

Achieved a recall of 40.09%, but its accuracy (71.55%) and AUC (0.6040) were the lowest among the five models. Its overall predictive performance was relatively weak.

**K-Nearest Neighbors (KNN)**

Achieved an accuracy of 79.48% and an F1 score of 0.4361. It provides a reasonable balance between precision (55.61%) and recall (35.87%), with the second-highest F1 score.

**Gaussian Naive Bayes**

Achieved the highest recall (97.14%) and AUC (0.7240). It successfully identifies almost all actual defaulters. However, its precision (23.40%) and accuracy (29.03%) are very low, indicating a large number of false positive predictions.

**Support Vector Machine (SVM)**

Achieved the highest accuracy (81.63%), F1 score (0.4501), and MCC (0.3824). It provides the strongest overall balance among the five models based on these evaluation metrics.

### Overall Winner

**Selected Model: Support Vector Machine (SVM)**

SVM is selected as the overall winner because it achieved:

- Highest Accuracy: 81.63%
- Highest F1 Score: 0.4501
- Highest MCC: 0.3824
- Second-highest AUC: 0.7201

Although Naive Bayes achieved the highest recall and AUC, its very low precision and accuracy indicate that it produces a large number of false positive predictions.

Therefore, SVM offers the best overall balance of predictive performance among the five evaluated models.

## 5. Streamlit Application

The project includes an interactive Streamlit application with the following features:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- View the evaluation metrics for the selected model.
- View the confusion matrix.
- View the classification report.

The application loads previously trained and saved models. It does not retrain models using the uploaded test dataset.

**Live Application:** [StreamLit Deployment Link](https://binaryclassificationdemomlproject.streamlit.app/)
