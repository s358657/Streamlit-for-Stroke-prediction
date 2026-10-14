import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM = 42

# 1. Caricamento del dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
y = data["stroke"]
X = data.drop(columns=["stroke", "id"])

# 2. Split stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM, stratify=y
)

# 3. Pipeline di preprocessing
categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
numerical_features = ["age", "avg_glucose_level", "bmi"]
binary_features = ["hypertension", "heart_disease"]

categorical_pipeline = Pipeline(
    [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

numerical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

transformer = ColumnTransformer(
    [
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features),
        ("binary", "passthrough", binary_features),
    ]
)

# 4. Tuning di Logistic Regression sul Training Set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

base_pipeline = Pipeline(
    [
        ("preprocessor", transformer),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM, class_weight="balanced", max_iter=1000
            ),
        ),
    ]
)

lr_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "saga"],
}

grid_search = GridSearchCV(
    base_pipeline, lr_params, cv=skf, scoring="average_precision", n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_base_pipeline = grid_search.best_estimator_

# 5. Calibrazione delle probabilità (rimuove la distorsione di class_weight='balanced')
calibrated_model = CalibratedClassifierCV(
    estimator=best_base_pipeline, method="sigmoid", cv=skf
)

# 6. Ricerca della soglia F2 ottimale Out-Of-Fold sulle probabilità calibrate
oof_probs = cross_val_predict(
    calibrated_model,
    X_train,
    y_train,
    cv=skf,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
valid_precision = precision[:-1]
valid_recall = recall[:-1]

f2_scores = np.divide(
    5 * valid_precision * valid_recall,
    4 * valid_precision + valid_recall,
    out=np.zeros_like(valid_precision),
    where=(4 * valid_precision + valid_recall) != 0,
)
best_threshold = float(thresholds[np.argmax(f2_scores)])

# 7. Fit finale e salvataggio del bundle
calibrated_model.fit(X_train, y_train)

model_bundle = {
    "model": calibrated_model,
    "threshold": best_threshold,
}

joblib.dump(model_bundle, "stroke_model_bundle.pkl")
print(
    f"Modello Logistic Regression salvato in 'stroke_model_bundle.pkl' (Best params: {grid_search.best_params_} | Soglia F2 calibrata: {best_threshold:.3f})"
)