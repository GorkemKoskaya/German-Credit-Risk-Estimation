# German Credit Risk Classification using Machine Learning in R

This project focuses on building and evaluating machine learning models to classify credit risk (Good vs Bad) using the [German Credit dataset](https://www.kaggle.com/datasets/uciml/german-credit). The implementation is done in R, and the workflow includes model comparison, performance visualization, and advanced sampling techniques to address class imbalance.

## 📁 Dataset

- Source: [Kaggle - German Credit](https://www.kaggle.com/datasets/uciml/german-credit)
- Records: 1000 samples
- Target variable: `Class` (Good / Bad)
- Features: 20 financial and categorical attributes related to credit information.

## 🧠 Models Implemented

Three baseline models were trained and evaluated:

- Decision Tree (CART)
- Random Forest
- Support Vector Machine (SVM - Linear)

Each model's performance is evaluated using:
- Accuracy
- AUC (Area Under Curve)
- ROC Curve

## ⚖️ Handling Class Imbalance

Class imbalance is a critical issue in the dataset. To address this, multiple resampling techniques were applied using the `caret` package:

- UpSampling
- DownSampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- ROSE (Random Over Sampling Examples)

Each method was applied to a Random Forest classifier and evaluated based on:

- Accuracy
- AUC
- F1 Score (for "Bad" class)

## 🔧 Hyperparameter Tuning

A final optimized Random Forest model was built using:

- 10-fold cross-validation
- UpSampling
- ROC as the evaluation metric
- Hyperparameter tuning for `mtry` using a predefined grid

## 📊 Visualizations

The project includes several visualizations:
- ROC curves for all models
- Bar chart comparing model accuracies
- Class distribution pie chart
- Performance comparison of sampling methods
- Final model tuning plot

## 📦 Libraries Used

```r
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)
library(MLmetrics)
library(reshape2)
library(ggplot2)
