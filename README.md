🧠 Stroke Prediction App

An interactive multi-page web application built with Streamlit and Machine Learning to predict the risk of stroke based on clinical and demographic parameters.

📋 Project Overview

This project provides an accessible, fast, and intuitive tool to estimate stroke risk probability using historical clinical data. Built with a multi-page Streamlit architecture, it allows users to navigate through home information, exploratory data analysis, and interactive prediction tools.

✨ Key Features

Multi-page Architecture: Structured navigation via Streamlit's pages/ directory and Home.py.

Real-time Prediction: Input demographic and clinical data to instantly view prediction outcomes using trained models.

Exploratory Data Analysis (EDA): Notebook support (explore.ipynb) and dataset visualizations.

Model Evaluation: Tracks model performance metrics using models_score.json and a bundled model file (stroke_model_bundle.pkl).

🛠️ System Requirements

Make sure you have the following installed on your system:

Python (version 3.8 or higher)

Pip (Python package manager)

🚀 Installation & Local Setup

Follow these steps to clone and run the project on your local machine:

Clone the repository:

git clone https://github.com/s358657/Streamlit-for-Stroke-prediction.git
cd Streamlit-for-Stroke-prediction


Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run Home.py


📁 Project Structure

Streamlit-for-Stroke-prediction/
│
├── pages/                               # Additional Streamlit application pages
├── .gitignore                           # Git ignore file
├── Home.py                              # Main entry point for the Streamlit app
├── explore.ipynb                        # Jupyter notebook for data exploration
├── healthcare-dataset-stroke-data.csv   # Reference dataset
├── main.py                              # Core script / utilities
├── models_score.json                    # Model evaluation scores
└── stroke_model_bundle.pkl              # Serialized Machine Learning model bundle


📊 Dataset

The application utilizes the healthcare-dataset-stroke-data.csv dataset, containing information on:

Gender and Age

Hypertension and Heart Disease

Marital status and Work type

Residence type and Average glucose level

Body Mass Index (BMI) and Smoking status
