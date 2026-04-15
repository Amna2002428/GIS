print("Starting the program...")
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 0) إعدادات عامة
# ============================================================
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

DATA_PATH = "Data.csv"   # عدّل المسار إذا لزم
TARGET_COL = "AQI"
OUTPUT_DIR = "results_output"
SHOW_PLOTS = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1) دوال مساعدة
# ============================================================
def finalize_plot(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def mape_safe(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mape_safe(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_%": mape
    }


def save_predictions_excel(y_true, y_pred, model_name):
    df_pred = pd.DataFrame({
        "Actual_AQI": y_true,
        "Predicted_AQI": y_pred,
        "Residual": y_true - y_pred,
        "Absolute_Error": np.abs(y_true - y_pred)
    })
    file_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.xlsx")
    df_pred.to_excel(file_path, index=False)
    return df_pred, file_path


def plot_training_history(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_loss_curve.png"))

    if "mae" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["mae"], label="Train MAE")
        if "val_mae" in history.history:
            plt.plot(history.history["val_mae"], label="Validation MAE")
        plt.title(f"{model_name} - MAE Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_mae_curve.png"))


def plot_actual_vs_pred(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    line_min = min(np.min(y_true), np.min(y_pred))
    line_max = max(np.max(y_true), np.max(y_pred))
    plt.plot([line_min, line_max], [line_min, line_max], "--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"{model_name} - Actual vs Predicted")
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_actual_vs_predicted.png"))


def plot_actual_vs_pred_line(y_true, y_pred, model_name, n_points=None):
    if n_points is not None:
        y_true_plot = y_true[:n_points]
        y_pred_plot = y_pred[:n_points]
        suffix = f"_first_{n_points}"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        suffix = ""

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_plot, label="Actual AQI")
    plt.plot(y_pred_plot, label="Predicted AQI")
    plt.title(f"{model_name} - Actual vs Predicted Curve")
    plt.xlabel("Sample Index")
    plt.ylabel("AQI")
    plt.legend()
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_actual_vs_predicted_curve{suffix}.png"))


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted AQI")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} - Residual Plot")
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_residuals.png"))


def plot_residuals_histogram(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.title(f"{model_name} - Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_residuals_histogram.png"))


def save_feature_importance(feature_names, scores, model_name):
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": scores
    }).sort_values(by="Importance", ascending=False)

    df_imp.to_excel(
        os.path.join(OUTPUT_DIR, f"{model_name}_feature_importance.xlsx"),
        index=False
    )
    return df_imp


def plot_feature_importance(feature_names, scores, model_name):
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": scores
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df_imp["Feature"], df_imp["Importance"])
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_feature_importance.png"))


def plot_comparison_overlay(y_true, preds_dict, n_points=None, title="Model Comparison Curve"):
    if n_points is not None:
        y_true_plot = y_true[:n_points]
        suffix = f"_first_{n_points}"
    else:
        y_true_plot = y_true
        suffix = ""

    plt.figure(figsize=(14, 6))
    plt.plot(y_true_plot, label="Actual AQI", linewidth=2)

    for model_name, pred in preds_dict.items():
        pred_plot = pred[:n_points] if n_points is not None else pred
        plt.plot(pred_plot, label=model_name)

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("AQI")
    plt.legend()
    finalize_plot(os.path.join(OUTPUT_DIR, f"models_overlay_comparison{suffix}.png"))


def plot_metrics_comparison_bars(comparison_df):
    metrics = ["MAE", "RMSE", "R2", "MAPE_%"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(comparison_df["Model"], comparison_df[metric])
        plt.title(f"Models Comparison - {metric}")
        plt.ylabel(metric)
        finalize_plot(os.path.join(OUTPUT_DIR, f"comparison_{metric}.png"))


def plot_metrics_comparison_lines(comparison_df):
    metrics = ["MAE", "RMSE", "R2", "MAPE_%"]

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(comparison_df["Model"], comparison_df[metric], marker="o", label=metric)

    plt.title("Models Comparison - Metrics Curves")
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.legend()
    finalize_plot(os.path.join(OUTPUT_DIR, "comparison_metrics_curves.png"))


def plot_rf_learning_curve(model, X, y, model_name="RandomForest"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="r2",
        n_jobs=-1,
        shuffle=True,
        random_state=SEED
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training R2")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation R2")
    plt.title(f"{model_name} - Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("R2 Score")
    plt.legend()
    finalize_plot(os.path.join(OUTPUT_DIR, f"{model_name}_learning_curve.png"))


# ============================================================
# 2) تحميل وتجهيز البيانات
# ============================================================
print("\n=== LOADING DATA ===")
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"العمود الهدف '{TARGET_COL}' غير موجود في الملف")

print("\n=== DATA OVERVIEW ===")
print(df.head())
print(df.info())
print(df.describe())

df = df.dropna().copy()

y = df[TARGET_COL].astype(float).values
X_df = df.drop(columns=[TARGET_COL]).copy()

X_df = X_df.select_dtypes(include=[np.number]).copy()

low_var_cols = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
if low_var_cols:
    print("Dropping constant columns:", low_var_cols)
    X_df.drop(columns=low_var_cols, inplace=True)

for col in X_df.columns:
    cmin, cmax = X_df[col].min(), X_df[col].max()
    if cmin >= 0 and cmax > 100:
        X_df[col] = np.log1p(X_df[col])

feature_names_all = X_df.columns.tolist()
X = X_df.values.astype(float)

cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.xlsx")
pd.concat([X_df, pd.Series(y, name=TARGET_COL)], axis=1).to_excel(cleaned_path, index=False)

print("\nFeatures used:", feature_names_all)
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3) التقسيم Train / Validation / Test
# ============================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=20
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=20
)

x_scaler = StandardScaler()
X_train_s = x_scaler.fit_transform(X_train)
X_val_s = x_scaler.transform(X_val)
X_test_s = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

joblib.dump(x_scaler, os.path.join(OUTPUT_DIR, "x_scaler.pkl"))
joblib.dump(y_scaler, os.path.join(OUTPUT_DIR, "y_scaler.pkl"))
joblib.dump(feature_names_all, os.path.join(OUTPUT_DIR, "feature_names_all.pkl"))


# ============================================================
# 4) نموذج ANN
# ============================================================
def build_ann(input_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.20),

        Dense(64),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.20),

        Dense(32),
        BatchNormalization(),
        Activation("relu"),

        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


print("\n=== TRAINING ANN ===")
ann_model = build_ann(X_train_s.shape[1])

ann_callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
]

ann_history = ann_model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=400,
    batch_size=32,
    callbacks=ann_callbacks,
    verbose=1,
    shuffle=True
)

ann_pred_s = ann_model.predict(X_test_s).ravel()
ann_pred = y_scaler.inverse_transform(ann_pred_s.reshape(-1, 1)).ravel()

ann_metrics = evaluate_model(y_test, ann_pred, "ANN")
save_predictions_excel(y_test, ann_pred, "ANN")

plot_training_history(ann_history, "ANN")
plot_actual_vs_pred(y_test, ann_pred, "ANN")
plot_actual_vs_pred_line(y_test, ann_pred, "ANN")
plot_actual_vs_pred_line(y_test, ann_pred, "ANN", n_points=100)
plot_residuals(y_test, ann_pred, "ANN")
plot_residuals_histogram(y_test, ann_pred, "ANN")

ann_model.save(os.path.join(OUTPUT_DIR, "ann_model.keras"))


# ============================================================
# 5) نموذج ANFIS
# ============================================================
print("\n=== PREPARING ANFIS ===")

TOP_K = min(6, X_train_s.shape[1])
mi_scores = mutual_info_regression(X_train_s, y_train_s, random_state=SEED)
idx_sorted = np.argsort(mi_scores)[::-1]
keep_idx = idx_sorted[:TOP_K]

X_train_anfis = X_train_s[:, keep_idx]
X_val_anfis = X_val_s[:, keep_idx]
X_test_anfis = X_test_s[:, keep_idx]

feature_names_anfis = [feature_names_all[i] for i in keep_idx]
mi_selected = mi_scores[keep_idx]

save_feature_importance(feature_names_anfis, mi_selected, "ANFIS_selected_features")
plot_feature_importance(feature_names_anfis, mi_selected, "ANFIS_selected_features")

joblib.dump(feature_names_anfis, os.path.join(OUTPUT_DIR, "feature_names_anfis.pkl"))


class ANFISRegressor(tf.keras.Model):
    def __init__(self, n_rules, n_features, x_init):
        super().__init__()
        self.R = n_rules
        self.D = n_features

        x_init = np.asarray(x_init, dtype=np.float32)
        n = len(x_init)

        pick = np.random.choice(n, size=self.R, replace=(n < self.R))
        centers_init = x_init[pick]

        sig0 = np.std(x_init, axis=0).astype(np.float32)
        sig0 = np.where(sig0 < 0.3, 0.3, sig0)
        sig_init = np.tile(sig0, (self.R, 1))

        def softplus_inverse(y):
            y = np.clip(y, 1e-6, 50.0)
            return np.log(np.exp(y) - 1.0)

        self.centers = self.add_weight(
            name="centers",
            shape=(self.R, self.D),
            initializer=tf.constant_initializer(centers_init),
            trainable=True
        )

        self.log_sigmas = self.add_weight(
            name="log_sigmas",
            shape=(self.R, self.D),
            initializer=tf.constant_initializer(softplus_inverse(sig_init)),
            trainable=True
        )

        self.consequents = self.add_weight(
            name="consequents",
            shape=(self.R, self.D + 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED),
            trainable=True
        )

        self.grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.grad_norm_tracker, self.mae_metric]

    def call(self, x, training=False):
        x = tf.cast(x, tf.float32)
        B = tf.shape(x)[0]

        x_exp = tf.expand_dims(x, axis=1)
        c = tf.expand_dims(self.centers, axis=0)

        sig = tf.nn.softplus(self.log_sigmas) + 1e-3
        sig = tf.expand_dims(sig, axis=0)

        z = (x_exp - c) / sig
        log_mu = -0.5 * tf.square(z)

        log_w = tf.reduce_sum(log_mu, axis=2)
        w_norm = tf.nn.softmax(log_w, axis=1)

        ones = tf.ones([B, 1], dtype=tf.float32)
        x1 = tf.concat([x, ones], axis=1)

        f = tf.matmul(x1, tf.transpose(self.consequents))
        y_out = tf.reduce_sum(w_norm * f, axis=1)
        return y_out

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        gn = tf.linalg.global_norm([g for g in grads if g is not None])

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.grad_norm_tracker.update_state(gn)
        self.mae_metric.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "grad_norm": self.grad_norm_tracker.result(),
            "mae": self.mae_metric.result()
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result()
        }


print("\n=== TRAINING ANFIS ===")
N_RULES = 12

anfis_model = ANFISRegressor(
    n_rules=N_RULES,
    n_features=X_train_anfis.shape[1],
    x_init=X_train_anfis
)

anfis_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(delta=1.0)
)

anfis_callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
]

anfis_history = anfis_model.fit(
    X_train_anfis, y_train_s,
    validation_data=(X_val_anfis, y_val_s),
    epochs=300,
    batch_size=32,
    callbacks=anfis_callbacks,
    verbose=1,
    shuffle=True
)

anfis_pred_s = anfis_model.predict(X_test_anfis).ravel()
anfis_pred = y_scaler.inverse_transform(anfis_pred_s.reshape(-1, 1)).ravel()

anfis_metrics = evaluate_model(y_test, anfis_pred, "ANFIS")
save_predictions_excel(y_test, anfis_pred, "ANFIS")

plot_training_history(anfis_history, "ANFIS")
plot_actual_vs_pred(y_test, anfis_pred, "ANFIS")
plot_actual_vs_pred_line(y_test, anfis_pred, "ANFIS")
plot_actual_vs_pred_line(y_test, anfis_pred, "ANFIS", n_points=100)
plot_residuals(y_test, anfis_pred, "ANFIS")
plot_residuals_histogram(y_test, anfis_pred, "ANFIS")

# مهم: نحفظ الأوزان فقط حتى لا يتوقف
anfis_model.save_weights(os.path.join(OUTPUT_DIR, "anfis_model.weights.h5"))


# ============================================================
# 6) Random Forest
# ============================================================
print("\n=== TRAINING RANDOM FOREST ===")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train_s, y_train)
rf_pred = rf_model.predict(X_test_s)

rf_metrics = evaluate_model(y_test, rf_pred, "RandomForest")
save_predictions_excel(y_test, rf_pred, "RandomForest")

plot_actual_vs_pred(y_test, rf_pred, "RandomForest")
plot_actual_vs_pred_line(y_test, rf_pred, "RandomForest")
plot_actual_vs_pred_line(y_test, rf_pred, "RandomForest", n_points=100)
plot_residuals(y_test, rf_pred, "RandomForest")
plot_residuals_histogram(y_test, rf_pred, "RandomForest")
plot_rf_learning_curve(rf_model, X_train_s, y_train, model_name="RandomForest")

save_feature_importance(
    feature_names_all,
    rf_model.feature_importances_,
    "RandomForest"
)
plot_feature_importance(
    feature_names_all,
    rf_model.feature_importances_,
    "RandomForest"
)

joblib.dump(rf_model, os.path.join(OUTPUT_DIR, "random_forest_model.pkl"))


# ============================================================
# 7) المقارنة النهائية
# ============================================================
comparison_df = pd.DataFrame([
    ann_metrics,
    anfis_metrics,
    rf_metrics
]).sort_values(by="R2", ascending=False)

comparison_path = os.path.join(OUTPUT_DIR, "models_comparison.xlsx")
comparison_df.to_excel(comparison_path, index=False)

print("\n=== FINAL COMPARISON ===")
print(comparison_df)

preds_dict = {
    "ANN": ann_pred,
    "ANFIS": anfis_pred,
    "RandomForest": rf_pred
}

plot_comparison_overlay(y_test, preds_dict, title="Actual vs All Model Predictions")
plot_comparison_overlay(y_test, preds_dict, n_points=100, title="Actual vs All Model Predictions (First 100 Samples)")
plot_metrics_comparison_bars(comparison_df)
plot_metrics_comparison_lines(comparison_df)


# ============================================================
# 8) رسم إضافي
# ============================================================
plt.figure(figsize=(8, 5))
plt.bar(comparison_df["Model"], comparison_df["R2"])
plt.title("Model Comparison - R2")
plt.ylabel("R2 Score")
finalize_plot(os.path.join(OUTPUT_DIR, "comparison_R2_basic.png"))

print(f"\nAll results saved in folder: {OUTPUT_DIR}")