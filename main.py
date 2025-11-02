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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
    plt.figure(figsize=(10, 7))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    y_min, y_max = y_test_reg.min(), y_test_reg.max()
    plt.plot([y_min, y_max], [y_min, y_max], 'r--', linewidth=2, label='Predicción Perfecta')
    plt.xlabel('PM2.5 Real (µg/m³)', fontsize=11, fontweight='bold')
    plt.ylabel('PM2.5 Predicho (µg/m³)', fontsize=11, fontweight='bold')
    plt.title('Regresión Lineal: Predicciones vs. Valores Reales\n(Conjunto de Test)', 
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Agregar información de rendimiento
    textstr = f'MAE: {mae_test:.2f} µg/m³\nR²: {r2_test:.3f}\nn_muestras: {len(y_test_reg)}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/regresion_pred_vs_real.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2) Residuales vs Predicho (diagnóstico)
    residuales = y_test_reg - y_pred_reg
    plt.figure(figsize=(10, 7))
    plt.scatter(y_pred_reg, residuales, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    plt.axhline(0, linestyle='--', linewidth=2, color='r', label='Residual = 0')
    plt.xlabel('PM2.5 Predicho (µg/m³)', fontsize=11, fontweight='bold')
    plt.ylabel('Residuales: Real - Predicho (µg/m³)', fontsize=11, fontweight='bold')
    plt.title('Diagnóstico de Residuales: Regresión Lineal\n(Conjunto de Test - Verificar Homocedasticidad)', 
              fontsize=12, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Agregar información sobre residuales
    textstr_res = f'Media: {residuales.mean():.3f}\nDesv.Est: {residuales.std():.3f}\n' + \
                  f'Min: {residuales.min():.2f}\nMax: {residuales.max():.2f}'
    plt.text(0.05, 0.95, textstr_res, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/regresion_residuales_vs_predicho.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Semana 5: Modelos Predictivos - Clasificación (Regresión Logística y K-NN)
    # Justificación: Usamos clasificación para predecir si la calidad del aire es "Buena" o "Mala" basado en las mismas variables ambientales. Esto es útil para generar alertas de calidad de aire. Comparamos dos algoritmos: Regresión Logística (modelo lineal) y K-NN (modelo basado en instancias) para evaluar su rendimiento.
    
    X_clf = df[columnas].copy()  # Variables predictoras para la clasificación
    y_clf = df['Calidad_Aire'].copy()  # Variable objetivo categórica (Buena/Mala)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=SEED
    )

    # Convertir etiquetas a numéricas para algunos cálculos
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_clf_encoded = le.fit_transform(y_train_clf)
    y_test_clf_encoded = le.transform(y_test_clf)

    # --- MODELO 1: REGRESIÓN LOGÍSTICA ---
    print("\n" + "="*60)
    print("Semana 5 - CLASIFICACIÓN: REGRESIÓN LOGÍSTICA")
    print("="*60)

    preprocess_clf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor_clf = ColumnTransformer(
        transformers=[("num", preprocess_clf, columnas)],
        remainder="drop"
    )

    model_logistic = LogisticRegression(max_iter=1000, random_state=SEED)

    logistic_pipeline = Pipeline(steps=[
        ("prep", preprocessor_clf),
        ("model", model_logistic)
    ])

    # Validación cruzada - Regresión Logística
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring_clf = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

    cv_results_logistic = cross_validate(
        logistic_pipeline, X_train_clf, y_train_clf_encoded, cv=cv, scoring=scoring_clf, return_train_score=False
    )

    print("\nRegresión Logística (CV en train, 5-fold):")
    print(f"Accuracy (media ± sd):  {cv_results_logistic['test_accuracy'].mean():.3f} ± {cv_results_logistic['test_accuracy'].std():.3f}")
    print(f"Precision (media ± sd): {cv_results_logistic['test_precision'].mean():.3f} ± {cv_results_logistic['test_precision'].std():.3f}")
    print(f"Recall (media ± sd):    {cv_results_logistic['test_recall'].mean():.3f} ± {cv_results_logistic['test_recall'].std():.3f}")
    print(f"F1-Score (media ± sd):  {cv_results_logistic['test_f1'].mean():.3f} ± {cv_results_logistic['test_f1'].std():.3f}")

    # Entrenamiento y evaluación en test - Regresión Logística
    logistic_pipeline.fit(X_train_clf, y_train_clf_encoded)
    y_pred_logistic = logistic_pipeline.predict(X_test_clf)

    acc_logistic = accuracy_score(y_test_clf_encoded, y_pred_logistic)
    precision_logistic = precision_score(y_test_clf_encoded, y_pred_logistic, zero_division=0)
    recall_logistic = recall_score(y_test_clf_encoded, y_pred_logistic, zero_division=0)
    f1_logistic = f1_score(y_test_clf_encoded, y_pred_logistic, zero_division=0)

    print("\nRegresión Logística - Evaluación en TEST:")
    print(f"Accuracy:  {acc_logistic:.3f}")
    print(f"Precision: {precision_logistic:.3f}")
    print(f"Recall:    {recall_logistic:.3f}")
    print(f"F1-Score:  {f1_logistic:.3f}")

    # Matriz de confusión - Regresión Logística
    cm_logistic = confusion_matrix(y_test_clf_encoded, y_pred_logistic)
    print("\nMatriz de Confusión (Regresión Logística):")
    print(cm_logistic)

    # --- MODELO 2: K-NN ---
    print("\n" + "="*60)
    print("Semana 5 - CLASIFICACIÓN: K-NEAREST NEIGHBORS (K-NN)")
    print("="*60)

    model_knn = KNeighborsClassifier(n_neighbors=5)

    knn_pipeline = Pipeline(steps=[
        ("prep", preprocessor_clf),
        ("model", model_knn)
    ])

    # Validación cruzada - K-NN
    cv_results_knn = cross_validate(
        knn_pipeline, X_train_clf, y_train_clf_encoded, cv=cv, scoring=scoring_clf, return_train_score=False
    )

    print("\nK-NN (CV en train, 5-fold, k=5):")
    print(f"Accuracy (media ± sd):  {cv_results_knn['test_accuracy'].mean():.3f} ± {cv_results_knn['test_accuracy'].std():.3f}")
    print(f"Precision (media ± sd): {cv_results_knn['test_precision'].mean():.3f} ± {cv_results_knn['test_precision'].std():.3f}")
    print(f"Recall (media ± sd):    {cv_results_knn['test_recall'].mean():.3f} ± {cv_results_knn['test_recall'].std():.3f}")
    print(f"F1-Score (media ± sd):  {cv_results_knn['test_f1'].mean():.3f} ± {cv_results_knn['test_f1'].std():.3f}")

    # Entrenamiento y evaluación en test - K-NN
    knn_pipeline.fit(X_train_clf, y_train_clf_encoded)
    y_pred_knn = knn_pipeline.predict(X_test_clf)

    acc_knn = accuracy_score(y_test_clf_encoded, y_pred_knn)
    precision_knn = precision_score(y_test_clf_encoded, y_pred_knn, zero_division=0)
    recall_knn = recall_score(y_test_clf_encoded, y_pred_knn, zero_division=0)
    f1_knn = f1_score(y_test_clf_encoded, y_pred_knn, zero_division=0)

    print("\nK-NN - Evaluación en TEST:")
    print(f"Accuracy:  {acc_knn:.3f}")
    print(f"Precision: {precision_knn:.3f}")
    print(f"Recall:    {recall_knn:.3f}")
    print(f"F1-Score:  {f1_knn:.3f}")

    # Matriz de confusión - K-NN
    cm_knn = confusion_matrix(y_test_clf_encoded, y_pred_knn)
    print("\nMatriz de Confusión (K-NN):")
    print(cm_knn)

    # --- GRÁFICOS DE COMPARACIÓN ---
    # 1) Comparación de métricas
    modelos = ['Regresión Logística', 'K-NN']
    accuracies = [acc_logistic, acc_knn]
    precisions = [precision_logistic, precision_knn]
    recalls = [recall_logistic, recall_knn]
    f1_scores = [f1_logistic, f1_knn]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Comparación de Rendimiento: Regresión Logística vs K-NN\n(Clasificación de Calidad del Aire - Conjunto de Test)', 
                 fontsize=14, fontweight='bold', y=0.995)

    # Accuracy
    bars1 = axes[0, 0].bar(modelos, accuracies, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Exactitud General\n(% de predicciones correctas)', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.03, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0, 0].set_facecolor('#f8f9fa')

    # Precision
    bars2 = axes[0, 1].bar(modelos, precisions, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_title('Precisión\n(De las predicciones positivas, % de correctas)', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.03, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0, 1].set_facecolor('#f8f9fa')

    # Recall
    bars3 = axes[1, 0].bar(modelos, recalls, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Recall (Sensibilidad)', fontweight='bold')
    axes[1, 0].set_title('Cobertura\n(% de casos positivos correctamente identificados)', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(recalls):
        axes[1, 0].text(i, v + 0.03, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1, 0].set_facecolor('#f8f9fa')

    # F1-Score
    bars4 = axes[1, 1].bar(modelos, f1_scores, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('F1-Score', fontweight='bold')
    axes[1, 1].set_title('Balance Precision-Recall\n(Media armónica de Precision y Recall)', fontweight='bold', fontsize=11)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(i, v + 0.03, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1, 1].set_facecolor('#f8f9fa')

    # Agregar leyenda global
    fig.text(0.99, 0.01, 'Nota: Los valores más altos indican mejor rendimiento del modelo', 
             ha='right', va='bottom', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/clasificacion_comparacion_modelos.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2) Matrices de confusión
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fig.suptitle('Matrices de Confusión: Predicciones de Calidad del Aire\n(Conjunto de Test)', 
                 fontsize=14, fontweight='bold', y=1.02)

    # Matriz de confusión - Regresión Logística
    sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True, 
                cbar_kws={'label': 'Cantidad'}, linewidths=2, linecolor='white')
    axes[0].set_title('Regresión Logística\n(Modelo Lineal)', fontweight='bold', fontsize=11, pad=15)
    axes[0].set_xlabel('Predicción del Modelo', fontweight='bold')
    axes[0].set_ylabel('Valor Real', fontweight='bold')
    axes[0].set_xticklabels(['Calidad Buena', 'Calidad Mala'], fontsize=10)
    axes[0].set_yticklabels(['Calidad Buena', 'Calidad Mala'], fontsize=10, rotation=0)
    
    # Agregar anotaciones adicionales para Regresión Logística
    tn_lr, fp_lr = cm_logistic[0]
    fn_lr, tp_lr = cm_logistic[1]
    textstr_lr = f'VP: {tp_lr} | FP: {fp_lr}\nFN: {fn_lr} | VN: {tn_lr}'
    axes[0].text(1.0, -0.35, textstr_lr, transform=axes[0].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Matriz de confusión - K-NN
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=True,
                cbar_kws={'label': 'Cantidad'}, linewidths=2, linecolor='white')
    axes[1].set_title('K-Nearest Neighbors (K=5)\n(Modelo basado en Instancias)', fontweight='bold', fontsize=11, pad=15)
    axes[1].set_xlabel('Predicción del Modelo', fontweight='bold')
    axes[1].set_ylabel('Valor Real', fontweight='bold')
    axes[1].set_xticklabels(['Calidad Buena', 'Calidad Mala'], fontsize=10)
    axes[1].set_yticklabels(['Calidad Buena', 'Calidad Mala'], fontsize=10, rotation=0)
    
    # Agregar anotaciones adicionales para K-NN
    tn_knn, fp_knn = cm_knn[0]
    fn_knn, tp_knn = cm_knn[1]
    textstr_knn = f'VP: {tp_knn} | FP: {fp_knn}\nFN: {fn_knn} | VN: {tn_knn}'
    axes[1].text(1.0, -0.35, textstr_knn, transform=axes[1].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Agregar leyenda general
    fig.text(0.5, -0.05, 'VP: Verdadero Positivo (detectó correctamente calidad mala) | FP: Falso Positivo (predijo mala cuando era buena)\n' +
                         'FN: Falso Negativo (no detectó calidad mala) | VN: Verdadero Negativo (predijo correctamente calidad buena)',
             ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/clasificacion_matrices_confusion.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("Gráficos guardados en la carpeta 'out/'")
    print("="*60)

if __name__ == '__main__':
    main()