# Streamlit-for-Stroke-prediction

## 📖 Project Overview

This project provides an accessible, fast, and intuitive tool to estimate stroke risk probability using historical clinical data. Built with a multi-page Streamlit architecture, it allows users to navigate through home information, exploratory data analysis, and interactive prediction tools.

## ✨ Key Features

* **Multi-page Architecture:** Structured navigation via Streamlit's `pages/` directory and `Home.py`.
* **Real-time Prediction:** Input demographic and clinical data to instantly view prediction outcomes using trained models.
* **Exploratory Data Analysis (EDA):** Notebook support (`explore.ipynb`) and dataset visualizations.
* **Model Evaluation:** Tracks model performance metrics using `models_score.json` and a bundled model file (`stroke_model_bundle.pkl`).

## 🛠️ System Requirements

Make sure you have the following installed on your system:

* Python (version 3.8 or higher)
* Pip (Python package manager)

## 🚀 Installation & Local Setup

Follow these steps to clone and run the project on your local machine:

### 1. Clone the repository
```bash
git clone [https://github.com/s358657/Streamlit-for-Stroke-prediction.git](https://github.com/s358657/Streamlit-for-Stroke-prediction.git)
cd Streamlit-for-Stroke-prediction

### 2.Install dependencies
```bash
pip install -r requirements.txt

### 3.Run the application
```bash
streamlit run Home.py
