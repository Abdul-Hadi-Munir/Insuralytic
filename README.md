# Healthcare Insurance Cost Predictor

A machine learning web application built with Python and Streamlit that predicts healthcare insurance costs based on personal and lifestyle attributes. The model uses Linear Regression trained on the Medical Cost Personal Dataset to estimate annual insurance charges.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Model Details](#model-details)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Application Interface](#application-interface)
- [Input Encoding](#input-encoding)
- [Tech Stack](#tech-stack)
- [Deployment](#deployment)

---

## Overview

Insurance companies need to estimate the cost of healthcare coverage for individuals based on various factors. This project automates that estimation by training a Linear Regression model on historical insurance data and serving predictions through an interactive web interface.

The user provides six inputs (age, sex, BMI, number of children, smoking status, and region), and the application returns a predicted annual insurance cost in USD.

---

## How It Works

1. **Data Loading** -- The training script reads the insurance dataset from `data/v1/raw/insurance.csv`
2. **Encoding** -- Categorical variables (sex, smoker, region) are converted to numeric values
3. **Training** -- A Linear Regression model is trained on 80% of the data, tested on 20%
4. **Saving** -- The trained model is serialized to `model.pkl` using Python's pickle module
5. **Prediction** -- The Streamlit app loads `model.pkl`, takes user inputs, encodes them, and displays the predicted cost

---

## Project Structure

```
Insuralytic/
│
├── app.py                     # Main Streamlit web application
│                                - Loads trained model from model.pkl
│                                - Accepts user inputs via form widgets
│                                - Encodes inputs and runs prediction
│                                - Displays result with Lottie animation
│
├── train_model.py             # Model training script
│                                - Loads and encodes the dataset
│                                - Splits data into train/test sets
│                                - Trains Linear Regression model
│                                - Evaluates and saves as model.pkl
│
├── model.pkl                  # Serialized trained model (auto-generated)
│
├── requirements.txt           # Python package dependencies
│
├── .gitignore                 # Files excluded from version control
│
├── README.md                  # Project documentation (this file)
│
└── data/
    └── v1/
        └── raw/
            └── insurance.csv  # Source dataset (1,338 records)
```

---

## Dataset Description

The project uses the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), a publicly available dataset containing 1,338 insurance records from the United States.

### Features

| Feature    | Type        | Range / Values                              | Description                                |
|------------|-------------|---------------------------------------------|--------------------------------------------|
| `age`      | Numeric     | 18 -- 64                                    | Age of the primary beneficiary             |
| `sex`      | Categorical | male, female                                | Gender of the policyholder                 |
| `bmi`      | Numeric     | 15.96 -- 53.13                              | Body Mass Index (kg/m^2)                   |
| `children` | Numeric     | 0 -- 5                                      | Number of dependents covered               |
| `smoker`   | Categorical | yes, no                                     | Whether the policyholder smokes            |
| `region`   | Categorical | northeast, northwest, southeast, southwest  | Residential area in the US                 |
| `charges`  | Numeric     | 1,121.87 -- 63,770.43                       | Annual insurance cost in USD (target)      |

### Key Observations

- Smokers pay significantly higher insurance costs than non-smokers
- BMI has a positive correlation with charges -- higher BMI leads to higher costs
- Age is positively correlated with charges -- older individuals tend to pay more
- Region has a minor but measurable effect on pricing
- Gender and number of children have relatively small impact on cost

---

## Model Details

### Algorithm

**Linear Regression** from scikit-learn. This algorithm fits a linear equation to the data by minimizing the sum of squared residuals between predicted and actual values.

### Training Configuration

| Parameter       | Value |
|-----------------|-------|
| Algorithm       | LinearRegression (scikit-learn) |
| Test Size       | 20%   |
| Random State    | 42    |
| Training Samples| 1,070 |
| Test Samples    | 268   |

### Performance Metrics

| Metric                      | Value       | Interpretation                                  |
|-----------------------------|-------------|-------------------------------------------------|
| R2 Score                    | 0.7833      | The model explains 78.33% of variance in costs  |
| MAE (Mean Absolute Error)   | $4,186.51   | Average prediction is off by about $4,187       |

### Limitations

- Linear Regression assumes a linear relationship between features and target, which may not fully capture complex interactions (e.g., the combined effect of smoking and high BMI)
- The model uses only 6 features; real-world insurance pricing considers many more factors such as medical history, prescription usage, and pre-existing conditions
- Predictions should be treated as estimates, not final quotes

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/Abdul-Hadi-Munir/Insuralytic.git
cd Insuralytic

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training the Model

Run the training script to generate `model.pkl`:

```bash
python train_model.py
```

Expected output:

```
R2 Score : 0.7833
MAE      : $4,186.51
Model saved as model.pkl
```

This creates `model.pkl` in the project root. The app loads this file at startup to make predictions.

---

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Application Interface

The Streamlit app uses a wide layout with three columns for clean input alignment:

| Column 1         | Column 2              | Column 3        |
|------------------|-----------------------|-----------------|
| Age (with Lottie animation) | BMI            | Smoker Status   |
| Sex              | Number of Children    | Region          |

A Lottie animation is displayed next to the Age input using `streamlit-lottie`. The animation is loaded from a public CDN URL and renders a lightweight health-themed animation.

After clicking "Predict Insurance Cost", the predicted value is displayed prominently below the inputs.

---

## Input Encoding

Categorical features are encoded to numeric values before being passed to the model:

### Sex
| Value  | Encoded |
|--------|---------|
| male   | 1       |
| female | 0       |

### Smoker
| Value | Encoded |
|-------|---------|
| yes   | 1       |
| no    | 0       |

### Region
| Value     | Encoded |
|-----------|---------|
| northeast | 0       |
| northwest | 1       |
| southeast | 2       |
| southwest | 3       |

---

## Tech Stack

| Component       | Technology         |
|-----------------|--------------------|
| Language        | Python             |
| Web Framework   | Streamlit          |
| ML Library      | scikit-learn       |
| Animation       | streamlit-lottie   |
| Data Processing | pandas, numpy      |
| HTTP Requests   | requests           |
| Serialization   | pickle             |

---

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository: `Abdul-Hadi-Munir/Insuralytic`
5. Set the main file path to `app.py`
6. Click Deploy

### Local Deployment

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Author

**Abdul Hadi Munir**

- GitHub: [Abdul-Hadi-Munir](https://github.com/Abdul-Hadi-Munir)
