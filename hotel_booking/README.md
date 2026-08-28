# Hotel Booking Cancellation Prediction

A machine learning project that predicts whether a hotel booking will be canceled based on information available at the time of the reservation.

## Problem

Hotel cancellations create uncertainty for hotels when planning room availability and expected revenue.

The objective of this project is to build a binary classification model that predicts:

* `0` → Booking is not canceled
* `1` → Booking is canceled

## Dataset

The project uses the **Hotel Booking Demand** dataset.

The dataset contains information about hotel reservations, including:

* Hotel type
* Lead time
* Arrival date
* Length of stay
* Number of adults, children, and babies
* Meal type
* Country
* Market segment
* Distribution channel
* Deposit type
* Customer type
* Previous cancellations
* Previous bookings
* Average daily rate
* Special requests

The target variable is:

```text
is_canceled
```

## Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model & Preprocessor
   ↓
Prediction
```

## Project Structure

```text
hotel_booking/
│
├── data/
│   └── hotel_bookings.csv
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   ├── main.ipynb
│   └── exploring_data_2.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predict.py
│   └── preprocessing.py
│
├── main.py
├── main.ipynb
├── predict.ipynb
├── .gitignore
└── README.md
```

## Machine Learning Model

The project uses **XGBoost** for classification.

XGBoost was selected because it is well suited to structured/tabular datasets and can model nonlinear relationships between features effectively.

The preprocessing and model are saved using `joblib`:

```text
models/
├── model.pkl
└── preprocessor.pkl
```

This allows the trained model to be reused for predictions without retraining it.

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The project achieved a ROC-AUC of approximately **0.95** on the test set.

ROC-AUC is particularly useful here because it measures how well the model separates canceled bookings from non-canceled bookings across different classification thresholds.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib
* Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone https://github.com/nimafarahani1919/machine-learning.git
```

Navigate to the project:

```bash
cd machine-learning/hotel_booking
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
python main.py
```

The trained model and preprocessor will be saved in the `models/` directory.

## What I Practiced

This project was built to practice an end-to-end machine learning workflow, including:

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Categorical and numerical preprocessing
* Scikit-learn pipelines
* Model training
* Model evaluation
* Model persistence
* Making predictions with a saved model
* Structuring a machine learning project into separate modules

## Future Improvements

* Hyperparameter optimization
* Cross-validation
* Feature importance analysis
* Model explainability with SHAP
* Prediction API
* Web interface for predictions
* Automated testing
* Model monitoring
