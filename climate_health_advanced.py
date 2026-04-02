"""
Climate & Health Risk Prediction - Advanced Model
=================================================
Ensemble approach with advanced feature engineering for maximizing F1 + ROC-AUC

Final Score = 0.60 x F1 + 0.40 x ROC-AUC
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectFromModel

# Advanced models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, skipping...")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, skipping...")

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("CatBoost not available, skipping...")

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "is_climate_sensitive"
ID_COL = "ID"

# =============================================================================
# Load Data
# =============================================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
climate_features = pd.read_csv("climate_features.csv")

# Drop deathdate from climate features to avoid duplicate columns
climate_features = climate_features.drop(columns=["deathdate"], errors='ignore')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Climate features shape: {climate_features.shape}")

# Merge datasets
train = train.merge(climate_features, on=ID_COL, how="left")
test = test.merge(climate_features, on=ID_COL, how="left")

print(f"Merged Train: {train.shape}, Merged Test: {test.shape}")

# =============================================================================
# Advanced Feature Engineering
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def engineer_features(df, is_train=True):
    """Advanced feature engineering for climate-health prediction"""
    df = df.copy()

    # Convert date
    df["deathdate"] = pd.to_datetime(df["deathdate"], errors="coerce")

    # ---- Temporal/Cyclical Features ----
    df["day_of_year"] = df["deathdate"].dt.dayofyear
    df["month"] = df["deathdate"].dt.month
    df["season"] = ((df["month"] % 12 + 3) // 3) % 4  # 0=DJF, 1=MAM, 2=JJA, 3=SON

    # Cyclical encoding for time (captures seasonal patterns)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- Age-based Risk Groups ----
    df["age_group"] = pd.cut(df["age"],
                              bins=[-1, 5, 18, 35, 50, 65, 100],
                              labels=[0, 1, 2, 3, 4, 5])
    df["is_vulnerable_age"] = ((df["age"] < 5) | (df["age"] > 65)).astype(int)
    df["is_elderly"] = (df["age"] > 60).astype(int)
    df["is_child"] = (df["age"] < 5).astype(int)
    df["age_squared"] = df["age"] ** 2

    # ---- Temperature Features ----
    df["temperature_range"] = df["max_temperature"] - df["min_temperature"]
    df["temp_extreme_high"] = (df["avg_temperature"] > 25).astype(int)
    df["temp_extreme_low"] = (df["avg_temperature"] < 18).astype(int)
    df["temp_anomaly"] = df["avg_temperature"] - df["tavg_30d"]
    df["temp_anomaly_7d"] = df["avg_temperature"] - df["tavg_7d"]
    df["max_temp_extreme"] = (df["max_temperature"] > 30).astype(int)
    df["min_temp_extreme"] = (df["min_temperature"] < 15).astype(int)

    # Temperature variability
    df["temp_variability"] = df["temp_range_mean_30d"] / (df["avg_temperature"] + 1)

    # ---- Precipitation Features ----
    df["is_heavy_rain"] = (df["precipitation"] > 5).astype(int)
    df["is_rainy_day"] = (df["precipitation"] > 0).astype(int)
    df["rain_intensity"] = df["precipitation"] / (df["rain_days_30d"] + 1)
    df["rain_anomaly"] = df["rain_sum_7d"] - (df["rain_sum_30d"] / 30 * 7)
    df["recent_rain_ratio"] = df["rain_sum_7d"] / (df["rain_sum_30d"] + 1)
    df["prolonged_rain"] = (df["rain_days_30d"] > 20).astype(int)

    # ---- NDVI (Vegetation) Features ----
    df["ndvi_change"] = df["ndvi_30d"] - df["ndvi_90d"]
    df["ndvi_high"] = (df["ndvi_30d"] > 0.7).astype(int)
    df["ndvi_low"] = (df["ndvi_30d"] < 0.4).astype(int)
    df["vegetation_stress"] = ((df["ndvi_30d"] < 0.5) & (df["avg_temperature"] > 23)).astype(int)

    # ---- Elevation/Terrain Features ----
    df["high_elevation"] = (df["elevation"] > 1500).astype(int)
    df["steep_slope"] = (df["slope"] > 5).astype(int)
    df["elevation_temp_interaction"] = df["elevation"] * df["avg_temperature"] / 1000

    # ---- Hot/Cold Day Features ----
    df["heat_wave"] = (df["hot_days_30d"] > 5).astype(int)
    df["extreme_heat"] = (df["hot_days_30d"] > 10).astype(int)

    # ---- Interaction Features ----
    # Age interactions with climate
    df["elderly_heat"] = df["is_elderly"] * df["temp_extreme_high"]
    df["child_rain"] = df["is_child"] * df["is_rainy_day"]
    df["vulnerable_extreme"] = df["is_vulnerable_age"] * df["temp_extreme_high"]

    # Rain and temperature interaction
    df["hot_humid"] = df["temp_extreme_high"] * df["is_rainy_day"]
    df["cold_wet"] = df["temp_extreme_low"] * df["is_heavy_rain"]

    # Elevation interactions
    df["high_elev_cold"] = df["high_elevation"] * df["temp_extreme_low"]
    df["slope_rain"] = df["steep_slope"] * df["is_heavy_rain"]

    # NDVI interactions
    df["hot_dry"] = df["temp_extreme_high"] * df["ndvi_low"]
    df["vegetation_heat"] = df["ndvi_high"] * df["temp_extreme_high"]

    # ---- Composite Risk Scores ----
    df["climate_stress_score"] = (
        df["hot_days_30d"] / 10 +
        df["temp_anomaly"].abs() +
        df["rain_anomaly"].abs() / 10 +
        df["is_vulnerable_age"] * 2
    )

    df["environmental_risk"] = (
        df["temp_extreme_high"].astype(int) +
        df["is_heavy_rain"].astype(int) +
        df["ndvi_low"].astype(int) +
        df["high_elevation"].astype(int)
    )

    # ---- Location-based features (from coordinates) ----
    df["lat_bin"] = pd.cut(df["latitude"], bins=[-1, 0.5, 1.0, 2.0], labels=[0, 1, 2])
    df["lon_bin"] = pd.cut(df["longitude"], bins=[30, 33, 34, 35], labels=[0, 1, 2])
    df["location_cluster"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    # Distance from equator (affects climate patterns)
    df["dist_from_equator"] = df["latitude"].abs()

    # ---- Gender encoding ----
    df["is_male"] = (df["gender"] == "Male").astype(int)
    df["is_female"] = (df["gender"] == "Female").astype(int)

    # ---- Zone encoding ----
    df["is_rural"] = (df["zone"] == "Rural").astype(int)
    df["is_urban"] = (df["zone"].isin(["Urban", "Peri_urban"])).astype(int)

    # Drop raw date
    df = df.drop(columns=["deathdate"], errors='ignore')

    return df

# Apply feature engineering
train = engineer_features(train, is_train=True)
test = engineer_features(test, is_train=False)

print(f"Features after engineering: {train.shape[1] - 2}")

# =============================================================================
# Prepare Features for Modeling
# =============================================================================

# Identify feature columns
exclude_cols = [ID_COL, TARGET, "zone", "gender", "location", "age_group",
                "lat_bin", "lon_bin", "location_cluster"]
feature_cols = [c for c in train.columns if c not in exclude_cols]

# Separate features and target
X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()

# Identify categorical and numeric
cat_cols = ['zone', 'gender']  # Simple categoricals
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"Numeric features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")
print(f"Total features: {len(feature_cols)}")

# =============================================================================
# Preprocessing Pipeline
# =============================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ]
)

# =============================================================================
# Model Definitions
# =============================================================================
print("\n" + "=" * 60)
print("DEFINING MODELS")
print("=" * 60)

def get_models():
    """Return dictionary of models to evaluate"""
    models = {}

    # Logistic Regression (baseline)
    models['LogisticRegression'] = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            solver="lbfgs"
        ))
    ])

    # Random Forest
    models['RandomForest'] = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting
    models['GradientBoosting'] = Pipeline([
        ("preprocess", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ))
    ])

    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ("preprocess", preprocessor),
            ("model", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])

    if HAS_LGBM:
        models['LightGBM'] = Pipeline([
            ("preprocess", preprocessor),
            ("model", lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1
            ))
        ])

    if HAS_CAT:
        models['CatBoost'] = Pipeline([
            ("preprocess", preprocessor),
            ("model", cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                class_weights="balanced",
                random_state=RANDOM_STATE,
                verbose=0
            ))
        ])

    return models

# =============================================================================
# Cross-Validation Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("CROSS-VALIDATION EVALUATION")
print("=" * 60)

models = get_models()
cv_results = {}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    # Get probability predictions via cross-validation
    cv_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)

    # Calculate metrics
    f1 = f1_score(y, cv_pred)
    auc = roc_auc_score(y, cv_proba)
    final_score = 0.60 * f1 + 0.40 * auc

    cv_results[name] = {
        'model': model,
        'f1': f1,
        'auc': auc,
        'final_score': final_score,
        'cv_proba': cv_proba
    }

    print(f"  F1 Score:    {f1:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Final Score: {final_score:.4f}")

# =============================================================================
# Find Best Model and Optimize Threshold
# =============================================================================
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

best_model_name = max(cv_results, key=lambda x: cv_results[x]['final_score'])
print(f"Best model: {best_model_name}")

# Optimize threshold for best model
best_proba = cv_results[best_model_name]['cv_proba']

thresholds = np.arange(0.30, 0.70, 0.01)
best_threshold = 0.5
best_combined = 0

for thresh in thresholds:
    pred = (best_proba >= thresh).astype(int)
    f1_t = f1_score(y, pred)
    auc_t = roc_auc_score(y, best_proba)  # AUC doesn't depend on threshold
    combined = 0.60 * f1_t + 0.40 * auc_t

    if combined > best_combined:
        best_combined = combined
        best_threshold = thresh
        best_f1_t = f1_t

print(f"Optimal threshold: {best_threshold:.2f}")
print(f"F1 at optimal: {best_f1_t:.4f}")

# =============================================================================
# Ensemble Predictions
# =============================================================================
print("\n" + "=" * 60)
print("ENSEMBLE PREDICTIONS")
print("=" * 60)

# Train all models on full data and create ensemble
ensemble_proba = np.zeros(len(X_test))
model_weights = {}

# Weight by CV performance
total_score = sum(cv_results[m]['final_score'] for m in cv_results)
for m in cv_results:
    model_weights[m] = cv_results[m]['final_score'] / total_score

print("Model weights for ensemble:")
for m, w in sorted(model_weights.items(), key=lambda x: -x[1]):
    print(f"  {m}: {w:.3f}")

for name, model in models.items():
    model.fit(X, y)
    test_proba = model.predict_proba(X_test)[:, 1]
    ensemble_proba += model_weights[name] * test_proba

# Also add simple average
simple_avg = np.mean([cv_results[m]['cv_proba'] for m in cv_results], axis=0)

# Blend weighted ensemble with simple average
final_proba = 0.7 * ensemble_proba + 0.3 * np.mean([
    models[m].fit(X, y).predict_proba(X_test)[:, 1]
    for m in models
], axis=0)

# Apply optimal threshold
final_pred = (final_proba >= best_threshold).astype(int)

print(f"\nEnsemble probability range: [{final_proba.min():.3f}, {final_proba.max():.3f}]")
print(f"Predicted positives: {final_pred.sum()} / {len(final_pred)}")

# =============================================================================
# Feature Importance (using best model)
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Fit best model and extract feature importance
best_model = models[best_model_name]
best_model.fit(X, y)

# Get feature names after preprocessing
try:
    # Get the trained model from pipeline
    trained_model = best_model.named_steps['model']

    if hasattr(trained_model, 'feature_importances_'):
        importances = trained_model.feature_importances_

        # Get feature names
        ohe = best_model.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names = cat_feature_names + num_cols

        # Create importance dataframe
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 20 features:")
        for i, row in imp_df.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
except Exception as e:
    print(f"Could not extract feature importance: {e}")

# =============================================================================
# Final Predictions and Submission
# =============================================================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "TargetF1": final_pred,
    "TargetRAUC": final_proba
})

submission.to_csv("advanced_submission.csv", index=False)
print(f"Submission saved: advanced_submission.csv")
print(f"Shape: {submission.shape}")
print(f"\nTargetF1 distribution:\n{submission['TargetF1'].value_counts()}")
print(f"\nTargetRAUC range: [{submission['TargetRAUC'].min():.4f}, {submission['TargetRAUC'].max():.4f}]")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nCross-Validation Results:")
print("-" * 50)
for name, results in sorted(cv_results.items(), key=lambda x: -x[1]['final_score']):
    print(f"{name:20s} | F1: {results['f1']:.4f} | AUC: {results['auc']:.4f} | Score: {results['final_score']:.4f}")

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"Best Model: {best_model_name}")
print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"Estimated CV Final Score: {cv_results[best_model_name]['final_score']:.4f}")
print(f"Submission file: advanced_submission.csv")
