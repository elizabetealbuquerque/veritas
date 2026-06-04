"""
pipeline.py — Avaliacao completa dos modelos Veritas com MLflow.

Implementa:
  1. Modelagem — comparacao dos 7 modelos com registro de hiperparametros OHEM
  2. Validacao — curva ROC, AUC, Recall da classe fake como metrica prioritaria
  3. Metricas — accuracy, precision, recall, F1, AUC e varredura de threshold
  4. Matriz de confusao por modelo
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import torch
import numpy as np
import joblib
import io
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from v3.dset import DirDataset
from v3.helper import set_model, transform
from v3.predict import predict

# ── Configuracoes ─────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
TEST_DIR   = Path("dataset/test")
THRESH     = 0.7
EXPERIMENT = "veritas-avaliacao-ciplab"
BATCH_SIZE = 32
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ─────────────────────────────────────────────────────────────────────────────

mlflow.set_experiment(EXPERIMENT)

modelos = sorted(MODELS_DIR.glob("*.pkl"))
if not modelos:
    print(f"ERRO: Nenhum .pkl em {MODELS_DIR.resolve()}")
    exit(1)

print(f"Modelos encontrados: {len(modelos)}")
for m in modelos:
    print(f"  - {m.name}")
print()

resultados = []

for model_path in modelos:
    print(f"\n{'='*55}")
    print(f"Avaliando: {model_path.name}")
    print(f"{'='*55}")

    with mlflow.start_run(run_name=model_path.stem):

        # ── Carrega modelo ────────────────────────────────────────────────
        try:
            set_model(str(model_path))
        except Exception as e:
            print(f"  ERRO ao carregar: {e}")
            mlflow.set_tag("status", "erro_carga")
            continue

        # ── Detecta arquitetura ───────────────────────────────────────────
        try:
            try:
                m = joblib.load(str(model_path))
            except Exception:
                with open(str(model_path), 'rb') as f:
                    m = torch.load(
                        io.BytesIO(f.read()),
                        map_location='cpu',
                        weights_only=False
                    )
            arch = m.default_cfg.get("architecture", "efficientnet_b0")
        except Exception:
            arch = "efficientnet_b0"

        # ── Loga parametros ───────────────────────────────────────────────
        mlflow.log_params({
            "modelo":        model_path.name,
            "arquitetura":   arch,
            "dataset":       "ciplab/real-and-fake-face-detection",
            "thresh":        THRESH,
            "batch_size":    BATCH_SIZE,
            "ohem_ohkeep":   0.5,
            "ohem_ohalpha":  0.5,
            "ohem_ohwarmup": 1,
            "lr_padrao":     "5e-4",
            "optimizer":     "AdamW",
            "scheduler":     "ReduceLROnPlateau",
            "augmentation":  "GaussianBlur_RandomRotation_HFlip_VFlip_ColorJitter",
        })

        # ── Coleta predicoes ──────────────────────────────────────────────
        try:
            test_ds = DirDataset(
                TEST_DIR, "test",
                transform=transform(train=False)
            )
            print(f"  Dataset: {len(test_ds)} imagens")

            probs_list  = []
            labels_list = []

            for b in range(0, len(test_ds), BATCH_SIZE):
                batch_imgs   = []
                batch_labels = []
                for i in range(b, min(b + BATCH_SIZE, len(test_ds))):
                    img, label = test_ds[i]
                    batch_imgs.append(img)
                    batch_labels.append(label)
                batch_tensor = torch.stack(batch_imgs)
                batch_probs  = predict(batch_tensor)
                probs_list.extend(batch_probs)
                labels_list.extend(batch_labels)

            probs   = np.array(probs_list,  dtype=float)
            ylabels = np.array(labels_list, dtype=int)

        except Exception as e:
            print(f"  ERRO ao avaliar: {e}")
            mlflow.set_tag("status", "erro_avaliacao")
            continue

        # ── 1. Metricas com threshold fixo ────────────────────────────────
        preds = (probs >= (1.0 - THRESH)).astype(int)

        acc         = accuracy_score(ylabels, preds)
        precision   = precision_score(ylabels, preds, zero_division=0)
        recall_real = recall_score(ylabels, preds, zero_division=0)
        f1          = f1_score(ylabels, preds, zero_division=0)
        recall_fake = recall_score(
            ylabels, preds, pos_label=0, zero_division=0
        )

        # ── 2. Matriz de confusao ─────────────────────────────────────────
        try:
            cm = confusion_matrix(ylabels, preds)

            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                ax=ax_cm,
                linewidths=0.5,
                linecolor='gray',
            )
            ax_cm.set_xlabel('Predito', fontsize=12)
            ax_cm.set_ylabel('Real', fontsize=12)
            ax_cm.set_title(
                f'Matriz de Confusão — {model_path.stem}\n'
                f'Threshold={THRESH} | Acc={acc:.3f}',
                fontsize=11
            )

            # Adiciona totais nas bordas
            total = cm.sum()
            tn, fp, fn, tp = cm.ravel()
            ax_cm.text(
                2.1, 0.5,
                f'TN={tn}\nFP={fp}\nFN={fn}\nTP={tp}\nTotal={total}',
                transform=ax_cm.transData,
                fontsize=9,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            cm_path = f"confusion_matrix_{model_path.stem}.png"
            fig_cm.tight_layout()
            fig_cm.savefig(cm_path, dpi=120, bbox_inches='tight')
            plt.close(fig_cm)
            mlflow.log_artifact(cm_path)

            # Loga TN, FP, FN, TP como metricas
            mlflow.log_metrics({
                "true_negative":  int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive":  int(tp),
            })

            print(f"  Matriz de confusao:")
            print(f"    TN={tn}  FP={fp}")
            print(f"    FN={fn}  TP={tp}")

        except Exception as e:
            print(f"  Aviso: Matriz de confusao nao gerada — {e}")

        # ── 3. AUC e curva ROC ────────────────────────────────────────────
        try:
            auc = roc_auc_score(ylabels, probs)
            fpr, tpr, _ = roc_curve(ylabels, probs)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color='#1D9E75', lw=2,
                    label=f'ROC (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Curva ROC — {model_path.stem}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            roc_path = f"roc_{model_path.stem}.png"
            fig.savefig(roc_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            mlflow.log_artifact(roc_path)

        except Exception as e:
            print(f"  Aviso: AUC nao calculado — {e}")
            auc = 0.0

        # ── 4. Varredura de threshold ─────────────────────────────────────
        recall_fakes_list    = []
        precisions_fake_list = []
        f1s_list             = []
        thresh_metrics       = {}

        for t in THRESHOLDS:
            p   = (probs >= (1.0 - t)).astype(int)
            rf  = recall_score(ylabels, p, pos_label=0, zero_division=0)
            pf  = precision_score(ylabels, p, pos_label=0, zero_division=0)
            f1t = f1_score(ylabels, p, zero_division=0)
            recall_fakes_list.append(rf)
            precisions_fake_list.append(pf)
            f1s_list.append(f1t)
            t_str = str(t).replace(".", "")
            thresh_metrics[f"t{t_str}_recall_fake"] = round(rf, 4)
            thresh_metrics[f"t{t_str}_f1"]          = round(f1t, 4)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(THRESHOLDS, recall_fakes_list,    'o-', color='#D85A30',
                 label='Recall fake (seguranca)')
        ax2.plot(THRESHOLDS, precisions_fake_list, 's-', color='#378ADD',
                 label='Precision fake')
        ax2.plot(THRESHOLDS, f1s_list,             '^-', color='#1D9E75',
                 label='F1-score')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title(f'Varredura de Threshold — {model_path.stem}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        thresh_path = f"thresh_{model_path.stem}.png"
        fig2.savefig(thresh_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)
        mlflow.log_artifact(thresh_path)

        # ── Loga metricas no MLflow ───────────────────────────────────────
        mlflow.log_metrics({
            "accuracy":    round(acc, 4),
            "precision":   round(precision, 4),
            "recall_real": round(recall_real, 4),
            "recall_fake": round(recall_fake, 4),
            "f1_score":    round(f1, 4),
            "auc":         round(auc, 4),
        })
        mlflow.log_metrics(thresh_metrics)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.set_tag("status", "ok")

        resultados.append({
            "nome":        model_path.name,
            "accuracy":    acc,
            "precision":   precision,
            "recall_real": recall_real,
            "recall_fake": recall_fake,
            "f1":          f1,
            "auc":         auc,
        })

        print(f"  accuracy:     {acc:.3f}")
        print(f"  precision:    {precision:.3f}")
        print(f"  recall real:  {recall_real:.3f}")
        print(f"  recall fake:  {recall_fake:.3f}  <- seguranca prioritaria")
        print(f"  f1_score:     {f1:.3f}")
        print(f"  AUC:          {auc:.3f}")

# ── Ranking final ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("RANKING FINAL (por AUC)")
print(f"{'='*55}")
resultados.sort(key=lambda x: x["auc"], reverse=True)
print(f"{'Modelo':<22} {'Acc':>5} {'Prec':>5} {'RecR':>5} {'RecF':>5} {'F1':>5} {'AUC':>5}")
print("-"*55)
for i, r in enumerate(resultados):
    pos = f"{i+1}o"
    print(
        f"{pos} {r['nome']:<20} "
        f"{r['accuracy']:.3f} "
        f"{r['precision']:.3f} "
        f"{r['recall_real']:.3f} "
        f"{r['recall_fake']:.3f} "
        f"{r['f1']:.3f} "
        f"{r['auc']:.3f}"
    )

print(f"\nAbra o MLflow com:")
print(f"  mlflow ui --backend-store-uri ./mlruns")
print(f"  http://localhost:5000")