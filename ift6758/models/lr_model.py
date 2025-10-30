import os
import sys

sys.path.append(os.path.abspath("../.."))
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from ift6758.models.wandb_utils import WandbLogger
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.calibration import CalibrationDisplay

class LRModel:
    def __init__(self, project_name='IFT6758.2025-A03', run_name="LR_run", config=None):
        self.model = LogisticRegression()
        self.logger = WandbLogger(project_name=project_name, run_name=run_name, config=config)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        self.logger.log_metrics(metrics, as_summary=True)
        return metrics
    
    def log_model(self, artifact_name='logreg_model', artifact_type='model', description='Logistic Regression Model'):
        self.logger.log_model_artifact(self.model, artifact_name=artifact_name, artifact_type=artifact_type, description=description)


    def ROC_curve(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_prob)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label="Random 50%")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        self.logger.log_figure("ROC_curve", fig)
        plt.close(fig)

        return fpr, tpr, thresholds
    
    def log_confusion_matrix(self, X_test, y_test):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix")

        self.logger.log_figure("Confusion_matrix", fig=fig)
        plt.close(fig)

    def plot_goal_rate_by_percentile(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        df = pd.DataFrame({'y_true': y_test, 'y_prob':y_prob})
        df['centile'] = pd.qcut(df['y_prob'], q=100, labels=False)
        goal_rate = df.groupby('centile')['y_true'].mean()

        fig, ax = plt.subplots()
        ax.plot(goal_rate.index, goal_rate.values, color='royalblue', label='Model 1')
        ax.invert_xaxis()
        ax.set_xlabel("Centile de probabilité du modèle de tir")
        ax.set_ylabel("Buts / (Tirs + Buts)")
        ax.set_title("Taux de buts par centile de probabilité")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.set_ylim(0, 1)
        self.logger.log_figure("TauxBut_centile", fig)
        plt.close(fig)

    def plot_cumulative_goal_curve(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        df = pd.DataFrame({'y_true': y_test, 'y_prob':y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['But cumulatifs'] = df['y_true'].cumsum()
        df['Rato cumulatifs'] = df['But cumulatifs'] / df['y_true'].sum()
        df['Proportion de tirs'] = np.arange(1, len(df) + 1) / len(df)

        fig, ax = plt.subplots()
        ax.plot(df["Proportion de tirs"], df["Rato cumulatifs"], label="Model 1")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1-x)*100:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

        ax.set_xlabel("Centile de probabilité du modèle de tir")
        ax.set_ylabel("Proportion cumulatives de buts")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.legend()
        plt.title("Courbe cumulative des buts")

        self.logger.log_figure("Cumulative_Goals", fig)
        plt.close(fig)

    def plot_calibration_curve(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        print("Prob range:", y_prob.min(), "→", y_prob.max())
        print("Unique y_test:", np.unique(y_test))
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10, strategy='uniform', ax=ax)
        plt.title('Courbe de calibration')
        
        self.logger.log_figure('Calibration_curve', fig)
        plt.close(fig)


    def finish(self):
        """Ferme la session W&B proprement"""
        self.logger.finish()

if __name__ == "__main__":
    print("Logistic Regression Model Module")
    data = pd.read_csv('../../games_data/feature_dataset_1_train.csv')
    X = data[['Distance']]
    y = data['Goal']
    # scaler = StandardScaler()
    # data['Distance_scaled'] = scaler.fit_transform(data[['Distance']])
    # X = data[['Distance_scaled']]
    # y = data['Goal']

    X_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LRModel(run_name='LR_model_probabilist_with_distance_class_unbalanced')

    model.train(X_train=X_train, y_train=y_train)
    metrics = model.evaluate(X_test=x_valid, y_test=y_valid)
    print(metrics)
    model.log_confusion_matrix(X_test=x_valid, y_test=y_valid)

    model.ROC_curve(x_valid, y_valid)
    model.plot_goal_rate_by_percentile(x_valid, y_valid)
    model.plot_cumulative_goal_curve(x_valid, y_valid)
    model.plot_calibration_curve(x_valid, y_valid)
    model.log_model(artifact_name='LR_model_probabiliste', description='LR model for part 3 of milestone 2 question 2 (distance, probabiité)')
