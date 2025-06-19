"""train_test_meta_xgboost.py"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import json


def train_and_test_meta_xgboost(seed, logger):
    # === Chargement du dataset méta ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    meta_dataset_path = os.path.join(project_root, 'meta_model', 'dataset', 'features_and_target',
                                     f"meta_dataset_seed_{seed}.csv")
    df = pd.read_csv(meta_dataset_path)

    # === Target : prédiction correcte ou non ===
    df["meta_label"] = df["y_true"] == df["y_pred"]
    df["meta_label"] = df["meta_label"].astype(int)

    # Analyse du déséquilibre de classes
    class_ratio = df["meta_label"].value_counts(normalize=True)
    logger.info(
        f"Distribution des classes méta: Classe 0 (erreur): {class_ratio[0]:.2%}, Classe 1 (correct): {class_ratio[1]:.2%}")
    scale_pos_weight = class_ratio[0] / class_ratio[1]  # Calcul automatique du poids

    # === Séparation X / y ===
    drop_cols = ["y_true", "y_pred", "meta_label"]
    X = df.drop(columns=drop_cols)
    y = df["meta_label"]
    y_true_full = df["y_true"].to_numpy()
    y_pred_full = df["y_pred"].to_numpy()

    print(f"Inputs to train_test_split: {type(X)}, {type(y)}, {type(y_true_full)}, {type(y_pred_full)}")

    # === Split train / test (avec stratify sur meta_label) ===
    X_train, X_test, y_train, y_test, y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(
        X, y, y_true_full, y_pred_full, stratify=y, test_size=0.2, random_state=42
    )

    # === Configuration avancée du modèle XGBoost ===
    base_model = XGBClassifier(
        n_estimators=1000,  # Augmenté pour permettre l'early stopping
        max_depth=5,  # Réduit pour réduire l'overfitting
        learning_rate=0.05,  # Réduit pour une meilleure convergence
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",  # Optimisation directe pour l'AUC
        early_stopping_rounds=50,
        random_state=42,
        scale_pos_weight=scale_pos_weight,  # Gestion du déséquilibre
        use_label_encoder=False
    )

    # === Validation croisée pour l'optimisation du seuil ===
    logger.info("Optimisation du seuil de décision par validation croisée...")
    thresholds = []
    best_threshold = 0.5

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        base_model.fit(X_fold_train, y_fold_train,
                       eval_set=[(X_fold_val, y_fold_val)],
                       verbose=False)

        y_prob_val = base_model.predict_proba(X_fold_val)[:, 1]
        fpr, tpr, t = roc_curve(y_fold_val, y_prob_val)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = t[optimal_idx]
        thresholds.append(optimal_threshold)

    best_threshold = np.mean(thresholds)
    logger.info(f"Seuil optimal déterminé: {best_threshold:.4f}")

    # === Entraînement final avec calibration ===
    logger.info("Entraînement final du méta-modèle avec calibration...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    # === Évaluation avec seuil optimisé ===
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > best_threshold).astype(int)

    # Métriques complètes
    logger.info("Évaluation sur le set de test :")
    logger.info(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC AUC Score : {roc_auc:.6f}")
    logger.info("Matrice de confusion :\n%s", confusion_matrix(y_test, y_pred))

    # === Analyse feature importance ===
    feature_importance = base_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    logger.info("Top 10 des features importantes:")
    for i in sorted_idx[:10]:
        logger.info(f"  {X.columns[i]}: {feature_importance[i]:.4f}")

    # === Sauvegarde du modèle ===
    results_dir = os.path.join(project_root, 'meta_model', 'results', f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"xgboost_meta_model_seed_{seed}.joblib")
    joblib.dump(calibrated_model, model_path)
    logger.info(f"Modèle sauvegardé : {model_path}")

    # === Sauvegarde des résultats et visualisations ===
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC du méta-modèle')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    # Sauvegarde des prédictions
    np.save(os.path.join(results_dir, 'xgboost_meta_model_probs.npy'), y_prob)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_true.npy'), y_true_test)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_y_pred.npy'), y_pred_test)
    np.save(os.path.join(results_dir, 'xgboost_meta_model_X_test.npy'), X_test)

    # Paramètres du modèle
    with open(os.path.join(results_dir, 'model_params.json'), 'w') as f:
        json.dump({
            'scale_pos_weight': scale_pos_weight,
            'optimal_threshold': best_threshold,
            'roc_auc': roc_auc,
            'class_distribution': class_ratio.to_dict()
        }, f)

    return roc_auc