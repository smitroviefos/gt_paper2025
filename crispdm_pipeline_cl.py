# This code is a CRISP-DM pipeline for analyzing datasets using Random Forest and SHAP for interpretability.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from fairlearn.metrics import demographic_parity_difference

warnings.filterwarnings("ignore")
os.makedirs("results_rf/shap", exist_ok=True)

class CRISP_DM:
    def __init__(self):
        self.results = []
        self.shap_classification = {}

    def load_prepare_dataset(self, data_id, target_name):
        data = fetch_openml(data_id=data_id, as_frame=True)
        df = data.frame.dropna()
        if df.shape[0] > 5000:
            df = df.sample(n=5000, random_state=42)

        if target_name not in df.columns:
            raise ValueError(f"Target '{target_name}' not found.")

        y = df[target_name].astype(str)
        y, uniques = pd.factorize(y)

        X = df.drop(columns=[target_name])
        X = X.apply(lambda col: col.astype(str) if str(col.dtype).startswith("category") else col)

        X_encoded = pd.get_dummies(X, drop_first=True)
        X_imputed = SimpleImputer(strategy='most_frequent').fit_transform(X_encoded)
        X_scaled = StandardScaler().fit_transform(X_imputed)
        X_reduced = PCA(n_components=min(10, X_scaled.shape[1])).fit_transform(X_scaled)

        return X_reduced, y, df, uniques

    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test[:, 0])
        except:
            dpd = np.nan
        return acc, f1, dpd, y_pred

    def shap_kmeans_analysis(self, model, X_train, X_test, dataset_name):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test[:100])
        shap_array = shap_values.values

        # 3D SHAP matrica kod klasifikacije
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]  # uzmi SHAP za klasu 0

        shap.summary_plot(shap_values, features=X_test[:100],
                          feature_names=[f'PC{i+1}' for i in range(X_test.shape[1])],
                          show=False)
        plt.title(f"SHAP Summary – {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"results_rf/shap/shap_{dataset_name}.png")
        plt.close()

        shap_matrix = shap_array.T
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(shap_matrix)

        result = {}
        for i in range(3):
            pcs = [f"PC{j+1}" for j in np.where(clusters == i)[0]]
            result[f"Player {i+1}"] = pcs
        return result

    def save_results(self):
        pd.DataFrame(self.results).to_csv("results_rf/evaluation_results.csv", index=False)
        with open("results_rf/shap_clusters.txt", "w", encoding="utf-8") as f:
            for dataset, clusters in self.shap_classification.items():
                f.write(f"\nDataset: {dataset}\n")
                for igrac, pcs in clusters.items():
                    f.write(f"  {igrac}: {', '.join(pcs)}\n")
    
    def result interpretation(self):
        # 1. Učitaj rezultate evaluacije
        df = pd.read_csv("results_rf/evaluation_results.csv")

        # 2. Bar graf - Accuracy i F1
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(data=df, x="Dataset", y="Accuracy", ax=ax[0])
        ax[0].set_title("Accuracy per dataset")
        ax[0].set_ylim(0, 1)

        sns.barplot(data=df, x="Dataset", y="F1 Score", ax=ax[1])
        ax[1].set_title("F1 Score per dataset")
        ax[1].set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

        # 3. Klasifikacija po F1 Score
        df["F1 Kategorija"] = pd.qcut(df["F1 Score"], q=3, labels=["Low", "Medium", "High"])
        print("Klasifikacija datasetova po F1 Score :")
        print(df[["Dataset", "F1 Score", "F1 Category"]])

        # 4. Pareto analiza F1 Score
        df_sorted = df.sort_values("F1 Score", ascending=False)
        df_sorted["Kumulativni udio (%)"] = df_sorted["F1 Score"].cumsum() / df_sorted["F1 Score"].sum() * 100

        plt.figure(figsize=(8, 5))
        sns.barplot(x="Dataset", y="F1 Score", data=df_sorted)
        plt.plot(df_sorted["Dataset"], df_sorted["Kumulativni udio (%)"], color="red", marker="o", label="Kumulativno (%)")
        plt.title("Pareto analiza F1 Score po datasetovima")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 5. SHAP igrači
        with open("results_rf/shap_clusters.txt", "r", encoding="utf-8") as f:
            shap_clusters = f.read()

        print("📊 SHAP klasifikacija značajki po igračima:")
        print(shap_clusters)   

# === GLAVNI PROGRAM ===
if __name__ == "__main__":
    datasets = [
        ("Adult Income", 1590, "class"),
        ("Credit-G", 31, "class"),
        ("Bank Marketing", 1461, "Class")
    ]

    crisp_dm = CRISP_DM()

    for name, data_id, target in datasets:
        print(f"\n📊 Dataset: {name} (ID: {data_id})")
        try:
            X, y, _, uniques = crisp_dm.load_prepare_dataset(data_id, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = crisp_dm.train_model(X_train, y_train)
            acc, f1, dpd, _ = crisp_dm.evaluate_model(model, X_test, y_test)

            crisp_dm.results.append({
                "Dataset": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "Demographic Parity Diff.": dpd
            })

            shap_result = crisp_dm.shap_kmeans_analysis(
                model,
                X_train.astype(float),
                X_test.astype(float),
                name.replace(" ", "_")
            )
            crisp_dm.shap_classification[name] = shap_result

        except Exception as e:
            print(f"❌ Error in dataset {name}: {e}")
            crisp_dm.results.append({
                "Dataset": name,
                "Accuracy": None,
                "F1 Score": None,
                "Demographic Parity Diff.": str(e)
            })

    crisp_dm.save_results()
    print("\nAnaliza završena. Rezultati spremljeni u mapu 'results_rf/'")




