# 📊 Financial Risk Prediction using Logistic Regression

A machine learning project applying Logistic Regression to predict financial risk using the German Credit Risk dataset.

## 📋 Project Overview
- Course: Machine Learning – Hanoi University of Industry
- Team: Group 06 | Class 20252IT6047009
- Dataset: German Credit Risk (1,000 records, 10 features)
- Goal: Classify customers into high-risk (bad) or low-risk (good) financial groups

## 🛠️ Tech Stack
- Language: Python 3
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, imbalanced-learn
- Tools: Jupyter Notebook

## ⚙️ Methodology
1. Data Cleaning: Handled missing values in Saving accounts (18%) and Checking account (39.5%)
2. Feature Engineering: One-Hot Encoding for categorical variables (26 features total)
3. Normalization: Z-score standardization implemented from scratch using NumPy
4. Class Balancing: SMOTE oversampling (30% → 50% high-risk ratio, 800 → 1,120 samples)
5. Model Training: Logistic Regression built from scratch with Gradient Descent + L2 Regularization
6. Hyperparameter Tuning: Grid search over learning rate, lambda, and classification threshold
7. Model Comparison: Logistic Regression vs Decision Tree

## 📈 Results

| Model | Accuracy | Recall | F1-score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Logistic (before scaling) | 0.70 | 0.117 | 0.189 | 0.680 |
| Logistic (after scaling) | - | - | - | - |
| Logistic (best) | 0.675 | 0.833 | 0.606 | 0.760 |
| Decision Tree | - | - | - | - |

> Best config: learning_rate=0.01, lambda=0.01, threshold=0.4

## 👥 Team Members
- Hoàng Tiến Đạt | Phùng Hải Dương | Nguyễn Quang Huy
- Ngọ Thị Lanh | Nguyễn Văn Linh | Nguyễn Duy Thái
- Nguyễn Thị Thảo | Đào Xuân Xâm
