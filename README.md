## 🇬🇧 English
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

## 🇹🇷 Türkçe

# German Credit Verisi ile Makine Öğrenmesi ile Kredi Riski Sınıflandırması (R)

Bu proje, [German Credit veri seti](https://www.kaggle.com/datasets/uciml/german-credit) kullanılarak kredi riskinin (İyi / Kötü) sınıflandırılmasını amaçlamaktadır. Tüm işlemler R programlama dili ile gerçekleştirilmiş olup, model karşılaştırmaları, performans görselleştirmeleri ve dengesiz veri yapısıyla başa çıkmak için örnekleme teknikleri uygulanmıştır.

## 📁 Veri Seti

- Kaynak: [Kaggle - German Credit](https://www.kaggle.com/datasets/uciml/german-credit)
- Gözlem Sayısı: 1000
- Hedef değişken: `Class` (Good / Bad)
- Özellikler: 20 adet finansal ve kategorik değişken

## 🧠 Uygulanan Modeller

Üç temel model eğitilmiş ve test edilmiştir:

- Karar Ağacı (CART)
- Rastgele Orman (Random Forest)
- Destek Vektör Makineleri (SVM - Doğrusal)

Her model aşağıdaki metriklerle değerlendirilmiştir:
- Doğruluk (Accuracy)
- AUC (Eğri Altı Alan)
- ROC Eğrisi

## ⚖️ Sınıf Dengesizliğiyle Baş Etme

Verideki ciddi sınıf dengesizliğini gidermek için `caret` paketi yardımıyla çeşitli örnekleme yöntemleri uygulanmıştır:

- Yukarı Örnekleme (UpSampling)
- Aşağı Örnekleme (DownSampling)
- SMOTE (Sentetik Azınlık Örnekleme Tekniği)
- ROSE (Rastgele Aşırı Örnekleme)

Her yöntem Rastgele Orman modeli üzerinde denenmiş ve aşağıdaki kriterlerle karşılaştırılmıştır:

- Doğruluk
- AUC
- F1 Skoru ("Bad" sınıfı için)

## 🔧 Hiperparametre Ayarlaması

Son olarak optimize edilmiş bir Rastgele Orman modeli oluşturulmuştur. Özellikleri:

- 10 katlı çapraz doğrulama
- Yukarı Örnekleme
- ROC metriğine göre değerlendirme
- `mtry` parametresi için grid arama yöntemi

## 📊 Görselleştirmeler

Projede aşağıdaki görselleştirmeler yer almaktadır:
- ROC eğrileri
- Modellerin doğruluk karşılaştırma grafiği
- Sınıf dağılımı pasta grafiği
- Örnekleme yöntemlerine göre performans karşılaştırma grafiği
- Final model için hiperparametre tuning grafiği

## 📦 Kullanılan R Paketleri

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

