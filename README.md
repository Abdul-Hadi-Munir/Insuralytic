# Healthcare Insurance Cost Predictor

A Streamlit web application that predicts healthcare insurance costs using Linear Regression.

## Project Structure

```
Insuralytic/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── model.pkl              # Trained Linear Regression model
├── requirements.txt       # Python dependencies
└── data/
    └── v1/
        └── raw/
            └── insurance.csv
```

## Features

- Predicts insurance cost based on age, sex, BMI, children, smoker status, and region
- Clean, professional Streamlit interface with wide layout
- Lottie animation integrated alongside the Age input
- Inputs are encoded and passed to a trained Linear Regression model

## Dataset

The model is trained on the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) containing 1,338 records with 7 columns:

| Feature  | Description                                      |
|----------|--------------------------------------------------|
| age      | Age of policyholder                              |
| sex      | Gender (male / female)                           |
| bmi      | Body Mass Index                                  |
| children | Number of dependents                             |
| smoker   | Smoking status (yes / no)                        |
| region   | US region (northeast, northwest, southeast, southwest) |
| charges  | Insurance cost (target variable)                 |

## Model Performance

| Metric   | Value       |
|----------|-------------|
| R2 Score | 0.7833      |
| MAE      | $4,186.51   |

## Setup

```bash
git clone https://github.com/Abdul-Hadi-Munir/Insuralytic.git
cd Insuralytic
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Run the App

```bash
streamlit run app.py
```

## Tech Stack

- Python
- Streamlit
- scikit-learn
- streamlit-lottie
