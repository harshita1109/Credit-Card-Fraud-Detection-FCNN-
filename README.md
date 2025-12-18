# 💳 Credit Card Fraud Detection using FCNN 🧠

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Project Overview

Credit card fraud is a major challenge due to highly **imbalanced datasets**. This project builds a **Fully Connected Neural Network (FCNN)** to accurately detect fraudulent transactions using deep learning techniques.

The model is trained on the popular **Kaggle Credit Card Fraud Dataset**, where features are anonymized using **PCA (V1–V28)**.

---

## 🚀 Key Features

✅ Data cleaning & duplicate removal
✅ Feature scaling using **StandardScaler**
✅ Handling class imbalance with **class weights**
✅ FCNN with **Dropout** for regularization
✅ **Early Stopping** to prevent overfitting
✅ Binary classification using **Sigmoid activation**

---

## 🧠 Model Architecture

```
Input Layer (30 features)
↓
Dense (128 neurons, ReLU) + Dropout (0.3)
↓
Dense (64 neurons, ReLU) + Dropout (0.2)
↓
Output Layer (1 neuron, Sigmoid)
```

---

## 🛠️ Tech Stack & Tools

🔹 Python
🔹 NumPy & Pandas
🔹 Matplotlib & Seaborn
🔹 Scikit-learn
🔹 TensorFlow & Keras

---

## 📂 Dataset

* **Source:** Kaggle – Credit Card Fraud Detection Dataset
* **Records:** 284,807 transactions
* **Fraud Cases:** ~0.17% (Highly Imbalanced)

---

## ⚙️ Workflow

1️⃣ Load & explore dataset
2️⃣ Handle duplicates & scaling
3️⃣ Train-test split
4️⃣ Apply class weights
5️⃣ Build & train FCNN
6️⃣ Evaluate performance

---

## 📊 Results

✔️ Achieved strong accuracy on test data
✔️ Improved fraud detection despite imbalance
✔️ Stable training using early stopping

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/fcnn-credit-card-fraud-detection.git
cd fcnn-credit-card-fraud-detection
python fcnn_for_credit_card_fraud_detection.py
```

---

## 📌 Future Improvements

🔮 Add Precision, Recall & F1-score
🔮 Try SMOTE for imbalance handling
🔮 Compare with ML models (LR, XGBoost)
🔮 Hyperparameter tuning

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## ⭐ Acknowledgements

* Kaggle Dataset Providers
* TensorFlow & Open Source Community

---

### 🌟 If you found this project useful, don’t forget to **star the repository**!
