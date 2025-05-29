# run_multi_seed.py

import random
import numpy as np
import torch
import pandas as pd

from model.train.train_model import train_and_save_model
from model.eval.evaluate_model import evaluate_model
from backtesting.backtest import run_backtest

SEEDS = [42, 2023, 7, 99, 1234]
results = []

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

for seed in SEEDS:
    print(f"\n🔁 Run avec seed {seed}")
    set_seed(seed)

    # Entraînement
    model_path, scaler_path = train_and_save_model(seed=seed)

    # Évaluation
    metrics = evaluate_model(model_path, scaler_path)
    metrics['seed'] = seed
    results.append(metrics)

    # results_btt = run_backtest(
    #     y_pred_path=f"../model/dataset/splits/seed_{seed}/y_pred_seed_{seed}.npy",
    #     close_prices_path=f"../model/dataset/splits/seed_{seed}/close_prices.npy",
    #     transaction_fee=0.001,  # 0.1% par trade
    # )
    #
    # print(f"📊 Résultats seed {results_btt['seed']} :")
    # for k, v in results_btt.items():
    #     if k != "seed":
    #         print(f" - {k}: {v}")

# Résultats
df = pd.DataFrame(results)
print("\n📊 Moyenne des scores sur {} seeds :".format(len(SEEDS)))
print(df.mean(numeric_only=True).round(4))

print("\n📉 Écart-type des scores :")
print(df.std(numeric_only=True).round(4))

# Export CSV
df.to_csv("results_multi_seed.csv", index=False)
print("\n✅ Résultats enregistrés dans results_multi_seed.csv")
