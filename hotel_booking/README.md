# Hotel Booking Cancellation Prediction

A machine learning project for predicting whether a hotel booking will be canceled.

This project uses the Hotel Booking Demand dataset and implements a complete tabular machine learning workflow, including data cleaning, feature engineering, preprocessing, model training, evaluation, and saving the trained model.

## Project Goal

The target variable is `is_canceled`.

```text
0 → Not canceled
1 → Canceled
```

The objective is to predict the cancellation status of a booking using information available in the dataset.

## Workflow

```text
Hotel Booking Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
XGBoost
        ↓
Evaluation
        ↓
Save Model
```

## Data Cleaning

The cleaning stage is implemented in `src/data_cleaning.py`.

The following steps are performed:

* Remove columns considered data leakage:

  * `reservation_status`
  * `reservation_status_date`
  * `assigned_room_type`
* Remove:

  * `arrival_date_week_number`
  * `previous_bookings_not_canceled`
* Convert invalid negative values in selected numerical columns to missing values.
* Remove bookings where both `adults` and `children` are zero.

This keeps information that can reasonably be used for prediction while removing features that could reveal the booking outcome.

## Feature Engineering

Feature engineering is implemented in `src/feature_engineering.py`.

The project creates several new features:

### Company indicator

The `company` column is converted into a binary feature:

```text
has_company
```

It indicates whether a company was associated with the booking.

### Country grouping

Countries with more than 1,000 occurrences are kept individually. Less frequent countries are grouped into:

```text
other
```

This reduces the number of categorical values.

### Arrival season

The arrival month is converted into a seasonal feature:

```text
Winter
Spring
Summer
Autumn
```

### Arrival day period

The arrival day of the month is grouped into four categories:

```text
early
mid_early
mid_late
late
```

### Total guests

A new feature is created:

```text
total_guests = adults + children + babies
```

### Agent grouping

Only the 20 most frequent agents are kept individually. Other agents are grouped into:

```text
other
```

## Preprocessing

The preprocessing pipeline is implemented using `ColumnTransformer`.

### Numerical features

Numerical features are processed using:

```text
SimpleImputer(strategy="median")
        ↓
MinMaxScaler()
```

### Categorical features

Categorical features are processed using:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

This allows numerical and categorical features to be transformed appropriately before they are passed to the model.

## Model

The classification model used in this project is **XGBoost**.

The model is implemented in:

```text
src/model_training.py
```

The current implementation uses:

```python
XGBClassifier()
```

and trains it on the preprocessed training data.

## Train / Test Split

The dataset is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Therefore:

* 80% of the data is used for training.
* 20% is used for testing.
* `stratify=y` preserves the class distribution between the two sets.
* `random_state=42` makes the split reproducible.

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

The evaluation code is located in:

```text
src/evaluation.py
```

The confusion matrix is also visualized in `main.py`.

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
│   ├── exploring_data_2.ipynb
│   └── main.ipynb
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

## Running the Project

From the `hotel_booking` directory:

```bash
python main.py
```

The script:

1. Loads the dataset.
2. Cleans the data.
3. Performs feature engineering.
4. Splits the data.
5. Fits the preprocessing pipeline.
6. Trains the XGBoost classifier.
7. Evaluates the model.
8. Saves the model and preprocessor.

The trained objects are saved as:

```text
models/model.pkl
models/preprocessor.pkl
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib
* Jupyter Notebook

## Project Purpose

This project is part of my machine learning practice and focuses on building a structured machine learning workflow rather than keeping all preprocessing and training code inside a single notebook.

The project separates data cleaning, feature engineering, preprocessing, training, and evaluation into individual modules under `src/`.

## Possible Improvements

Some possible next steps for the project are:

* Hyperparameter tuning for XGBoost
* Cross-validation
* Feature importance analysis
* Threshold optimization
* Model explainability
* A more complete prediction interface
* Automated tests
