# Batman Predictor – Fighting Crime with AI

Ever wondered how machine learning can help us understand crime patterns and make crime data easier to explore?

**Batman Predictor** is a machine learning project built around historical crime data from the Los Angeles Police Department (LAPD). The goal is to explore crime patterns, analyze reported incidents, and predict whether a crime belongs to **Part 1 or Part 2** according to the LAPD classification system.

The project combines data analysis, machine learning, interactive visualization, and a web application to turn raw crime records into something more accessible and useful.

## Project Overview

The project follows an end-to-end machine learning workflow, starting with data exploration and preprocessing, then moving through model development and evaluation, and finally integrating selected trained models into an interactive Streamlit application.

### 1. Data Exploration and Preprocessing

Before training the models, the dataset went through several preparation steps to make it suitable for machine learning.

Some of the main tasks included:

- Exploring the dataset and understanding its features.
- Handling missing values.
- Analyzing numerical and categorical variables.
- Encoding categorical features.
- Applying transformations such as Yeo-Johnson to improve feature distributions.
- Scaling numerical features using StandardScaler and RobustScaler.
- Exploring crime distributions through statistical analysis and visualizations.

### 2. Machine Learning Models

I experimented with **11 classification algorithms** to explore different approaches to the crime classification task:

- Logistic Regression
- Random Forest
- XGBoost
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- AdaBoost
- Bagging
- Extra Trees
- LightGBM
- CatBoost

The models were evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The training results and model evaluation information were also saved for later comparison.

### 3. Interactive Streamlit Application

After the model development stage, I built an interactive application using Streamlit to make the project easier to explore and use.

The application includes:

- **Crime Data Exploration:** Explore the available crime records and their characteristics.
- **Statistics and Visualizations:** Understand crime distributions and patterns through interactive charts.
- **Interactive Crime Map:** Explore geographical crime information through a Folium-based map.
- **Crime Prediction:** Use trained classification models to predict whether an incident belongs to Part 1 or Part 2.
- **Model Comparison:** Explore the performance of the selected models.

### 4. Docker Deployment

To make the application easier to run in different environments, I containerized it using Docker.

The project includes:
- A `Dockerfile` to build the application image.
- A `docker-compose.yml` file to simplify running the application.
- A `.dockerignore` file to exclude unnecessary files from the Docker build context.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Geographical Visualization | Folium |
| Web Application | Streamlit |
| Deployment | Docker, Docker Compose |

## Project Structure

```text
Batman-Predictor-Fighting-Crime-with-AI/
│
├── Files/
│   ├── Trained models
│   └── Preprocessing artifacts
│
├── Final Proj1.ipynb       # Data analysis and model training
├── app.py                  # Streamlit application
├── crime_sample.csv        # Dataset used by the application
├── crime_map.html          # Interactive crime map
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose configuration
└── .dockerignore           # Docker build exclusions
```

**Note:** The notebook contains experiments with 11 classification algorithms. The current Streamlit application uses three selected trained models: Logistic Regression, Random Forest, and XGBoost, along with the required preprocessing artifacts.

## Getting Started

Want to run the project locally? Follow these steps.

### Option 1: Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/Mennaateya/Batman-Predictor-Fighting-Crime-with-AI.git
cd Batman-Predictor-Fighting-Crime-with-AI
```

**2. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**

```bash
streamlit run app.py
```

Once the application starts, open:

`http://localhost:8501`

### Option 2: Run with Docker

If you prefer Docker, make sure Docker is installed and running.

**1. Build and start the application**

```bash
docker compose up --build
```

**2. Open the application**

`http://localhost:8501`

**3. Stop the application**

```bash
docker compose down
```

## Live Demo

You can also explore the deployed application here:

[Open Batman Predictor on Streamlit](https://batman-predictor-fighting-crime-with-ai.streamlit.app/)

## A Little About the Project

This project was a great opportunity to work through different stages of a machine learning application, from exploring and preparing real-world data to experimenting with classification algorithms and building an interactive application.

It also helped me understand how model development connects with application development and deployment.

---

**Developed by Menna Tallah Ateya**

- GitHub: https://github.com/Mennaateya
- LinkedIn: https://www.linkedin.com/in/menna-ateya/