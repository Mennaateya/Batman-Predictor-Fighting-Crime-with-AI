# 🦇 Batman Crime Predictor – Fighting Crime with AI  

_"Because even Batman needs data to fight crime"_  

This project applies **Machine Learning** to analyze and predict crime patterns in the **City of Los Angeles**.  
Inspired by Batman’s mission to fight crime, the project uses **real-world crime data** to build predictive models that classify crimes into **violent vs. non-violent** categories.  

---

## 📂 Dataset  

The dataset used in this project is the official **[Crime Data from 2020 to Present – Los Angeles](https://catalog.data.gov/dataset/crime-data-from-2020-to-present))**, provided by the **Los Angeles Police Department (LAPD)**.  

👉 **Please download the dataset from the link above**  

---

## ⚙️ Project Workflow  

1. **Data Loading & Cleaning**
   - Handle missing values & duplicated rows.  
   - Drop irrelevant features.  
   - Encode categorical variables.  
   - Feature engineering: Extract time-based features (hour, day, month, year).  

2. **Exploratory Data Analysis (EDA)**
   - Visualizations using **Matplotlib, Seaborn, and Plotly**.  
   - Crime heatmaps, trends, and victim demographics.  
   - Folium map with interactive crime markers.  

3. **Preprocessing**
   - Iterative Imputation for missing values.  
   - Outlier handling.  
   - Feature scaling: StandardScaler, RobustScaler, PowerTransformer, log transform.  

4. **Model Training**
   - Baseline models: Logistic Regression, KNN, Decision Tree, Naive Bayes.  
   - Ensemble models: Random Forest, Extra Trees, Bagging, AdaBoost.  
   - Gradient boosting: XGBoost, LightGBM, CatBoost.  

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1.  
   - Confusion Matrices & Classification Reports.  
   - Top features identified using Random Forest.  

---

## 🤖 Models Used  

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Extra Trees Classifier  
- Gaussian Naive Bayes  
- Bagging Classifier  
- AdaBoost  
- XGBoost  
- LightGBM  
- CatBoost  

All trained models are saved as `.pkl` files inside the `Files/` directory.  

---

## 📊 Visualizations  

- **Interactive Folium Map** of 5000 sampled crimes.  
- **Crime heatmap** by day & hour.  
- **Violin plots** for crime types across hours.  
- **Monthly trends** showing rise/fall in crime rates.  
- **Victim demographics** (age, gender, descent).  
- **Weapon usage distribution**.  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/USERNAME/Batman-Predictor-Fighting-Crime-with-AI.git
   cd Batman-Predictor-Fighting-Crime-with-AI

