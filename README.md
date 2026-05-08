# Stroke Risk Prediction: GPU-Accelerated Clinical Analysis

This project implements a high-performance machine learning pipeline to predict stroke risk using clinical biometric data. The primary objective was to move beyond standard accuracy and engineer a decision-support tool optimized for **Recall**, ensuring the highest possible sensitivity for early-stage medical intervention.

### 🚀 Core Engineering Features
* **Multi-Model Comparison:** Evaluated and compared **Random Forest, SVM, k-Nearest Neighbors, and Gaussian Naive Bayes**.
* **GPU Acceleration:** Leveraged GPU-optimized libraries to accelerate hyperparameter tuning and model training.
* **Clinical Calibration:** Implemented a custom classification threshold of **0.2** to minimize life-threatening False Negatives, prioritizing patient safety over balanced accuracy.
* **Significance Analysis:** Utilized Random Forest feature importance to identify the top three clinical predictors: **Age, Average Glucose Level, and BMI**.

### 🛠️ Technical Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Optimization:** GridSearchCV for hyperparameter tuning
* **Environment:** Google Colab / Jupyter Notebook

### 📊 Key Findings
The analysis determined that while all models showed high predictive potential, the **Random Forest** model with an adjusted threshold provided the best balance for clinical application. By shifting the decision boundary, the system achieved a significantly higher **Recall**, effectively flagging high-risk cases that standard 0.5 threshold models would have overlooked.

---

### 📂 Repository Structure
* `GPU Accelerated ML.ipynb`: Main Jupyter Notebook containing the end-to-end pipeline.
* `healthcare-dataset-stroke-data.csv`: Source clinical dataset.
* `visualizations/`: Exported Confusion Matrices and Feature Importance plots.
