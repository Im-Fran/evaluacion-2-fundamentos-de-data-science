from config import out_dir
from loader import prepara_df
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Guarda una semilla para reproducibilidad
SEED = 27031410
np.random.seed(SEED)
random.seed(SEED)

def main():
    df = prepara_df()
    df["Calidad_Aire"] = np.where(df["MP2.5"] < 25, "Buena", "Mala")

    # Preparación de Datos para Modelos
    columnas = ['Temperatura', 'Humedad_Relativa', 'Viento']

    X_reg = df[columnas].copy() # Variables predictoras para la regresión
    y_reg = df['MP2.5'].copy() # Variable objetivo continua

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=SEED
    )

    # Semana 4: Modelos Predictivos - Regresión (Regresión Lineal Múltiple)
    # Justificación: Usamos regresión lineal múltiple para predecir el nivel continuo de MP2.5 basado en variables como temperatura, humedad, viento y hora. Esto es útil para estimar contaminación futura y evaluar el impacto de factores ambientales. Evaluamos con MAE (error absoluto medio) y R² (coeficiente de determinación) para medir precisión.
    # Preprocesamiento
    preprocess_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # manejo de nulos
        ("scaler", StandardScaler())  # útil si luego comparamos con otros modelos
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", preprocess_num, columnas)],
        remainder="drop"
    )

    model_reg = LinearRegression()

    reg_pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model_reg)
    ])

    # Validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    cv_results = cross_validate(reg_pipeline, X_train_reg, y_train_reg, cv=cv, scoring=scoring, return_train_score=False)

    mean_r2 = cv_results["test_r2"].mean()
    std_r2 = cv_results["test_r2"].std()
    mean_mae = -cv_results["test_mae"].mean()
    std_mae = cv_results["test_mae"].std()

    print("\nSemana 4 - Regresión Lineal (CV en train, 5-fold):")
    print(f"R² (media ± sd): {mean_r2:.3f} ± {std_r2:.3f}")
    print(f"MAE (media ± sd): {mean_mae:.2f} ± {std_mae:.2f}")

    # Entrenamiento y evaluacion en test
    reg_pipeline.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg_pipeline.predict(X_test_reg)

    mae_test = mean_absolute_error(y_test_reg, y_pred_reg)
    r2_test = r2_score(y_test_reg, y_pred_reg)

    print("\nSemana 4 - Evaluación en TEST:")
    print(f"MAE: {mae_test:.2f}")
    print(f"R²: {r2_test:.2f}")

    # Gráfico: Línea de regresión (predicho vs. real)
    # 1) Predicho vs Real
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    y_min, y_max = y_test_reg.min(), y_test_reg.max()
    plt.plot([y_min, y_max], [y_min, y_max], 'r--')
    plt.xlabel('PM2.5 Real')
    plt.ylabel('PM2.5 Predicho')
    plt.title('Regresión Lineal: Predicho vs. Real')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/regresion_pred_vs_real.png", dpi=150)
    plt.close()

    # 2) Residuales vs Predicho (diagnóstico)
    residuales = y_test_reg - y_pred_reg
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_reg, residuales, alpha=0.5)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('PM2.5 Predicho')
    plt.ylabel('Residuales (Real - Predicho)')
    plt.title('Regresión Lineal: Residuales vs. Predicho')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/regresion_residuales_vs_predicho.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    main()