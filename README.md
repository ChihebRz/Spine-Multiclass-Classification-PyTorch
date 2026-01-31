# 🦴 Spine Multiclass Classification – PyTorch

A machine learning project that classifies spinal conditions into **three categories** using biomechanical features and a neural network built with **PyTorch**.

## 🎯 Classes

- **Normal**
- **Hernia**
- **Spondylolisthesis**

---

## 📂 Project Structure

spine-multiclass-classification/
├── data/
│   └── column_3C_weka.xlsx
│
├── notebooks/
│   ├── exploration.ipynb
│   └── test_inference.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── artifacts/
│   ├── model.pth
│   └── scaler.pkl
│
├── .gitignore
├── README.md
└── requirements.txt


---

## 📊 Dataset

- 6 numerical biomechanical features  
- Target: 3 classes  
- Source: Vertebral Column Dataset (3C)

### Label Encoding

| Class | Encoded |
|-----|---------|
| Normal | 0 |
| Hernia | 1 |
| Spondylolisthesis | 2 |

---

## ⚙️ Preprocessing

- Label encoding  
- Feature scaling with **StandardScaler**  
- Train / Test split (80% / 20%)  
- Conversion to PyTorch tensors  
- Scaler saved for inference (`scaler.pkl`)

---

## 🧠 Model Architecture

- Fully Connected Neural Network  
- Input: 6 features  
- Output: 3 classes  
- Loss: **CrossEntropyLoss**  
- Optimizer: **Adam**

---

## 🚀 Training

```bash
python src/train.py
Trains for 2000 epochs

Saves model to:

artifacts/model.pth
```

📈 Evaluation
```bash
python src/evaluate.py
```
Metrics
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion Matrix
- Classification Report

Current Results
```
Accuracy : 0.79
Precision: 0.75
Recall   : 0.75
F1-score : 0.75
```
Class Performance

| Class             | Precision | Recall | F1   |
|-------------------|-----------|--------|------|
| Normal            | 0.70      | 0.80   | 0.74 |
| Hernia            | 0.64      | 0.58   | 0.61 |
| Spondylolisthesis | 0.93      | 0.87   | 0.90 |

🔎 The model performs best on Spondylolisthesis, while most confusion occurs between Hernia and Normal.

🔮 Inference
```bash
python src/predict.py
```
Example:
```python
sample = np.array([63.0, 22.0, 14.0, 45.0, 12.0, 8.0])
```
Output:
```
Prediction : Spondylolisthesis
Probabilities: [[0.05, 0.12, 0.83]]
```

🧪 Exploratory Analysis
Performed in notebooks:

- Feature distributions
- Correlation heatmap
- PCA visualization
- Class imbalance analysis

🛠 Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```
Main libraries:

- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

📌 Possible Improvements
- Class-weighted loss
- Data augmentation
- Hyperparameter tuning
- Try alternative models (SVM, Random Forest)
- t-SNE visualization

👤 Author
Chiheb Rezgui
Big Data & Data Analytics Graduate
Machine Learning Enthusiast