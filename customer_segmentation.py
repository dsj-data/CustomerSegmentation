# Customer Segmentation Analysis
# ------------------------------

# ==========================================
# 1. Import libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==========================================
# 2. Load dataset
# ==========================================
#!git clone https://github.com/AnalyticsKnight/yemoonsaBigdata

x_train = pd.read_csv('/content/yemoonsaBigdata/datasets/Part5/404_x_train.csv')
y_train = pd.read_csv('/content/yemoonsaBigdata/datasets/Part5/404_y_train.csv')

data = pd.merge(x_train, y_train, on="ID")
df = data.drop(columns=["ID"], axis=1)

# ==========================================
# 3. EDA
# ==========================================
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x="Profession", data=df)
plt.xticks(rotation=45)
plt.title("Profession Distribution")
plt.show()

# ==========================================
# 4. Preprocessing
# ==========================================
label_mappings = {}
cat_cols = ["Gender", "Ever_Married", "Graduated", "Segmentation"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

df = pd.get_dummies(df, columns=["Profession"], dtype=int)
df["Spending_Score"] = df["Spending_Score"].map({"Low": 0, "Average": 1, "High": 2})

# ==========================================
# 5. Supervised Classification
# ==========================================
X = df.drop(columns=["Segmentation"])
y = df["Segmentation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("----")
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ==========================================
# 6. K-Means Clustering
# ==========================================
X_cluster = df.drop(columns=["Segmentation"])
inertia, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    inertia.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels))

plt.plot(K_range, inertia, "bo-")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_cluster)

# ==========================================
# 7. PCA Visualization
# ==========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["KMeans_Cluster"], palette="viridis")
plt.title("K-Means Clusters (PCA Projection)")
plt.show()

# ==========================================
# 8. Compare with Existing Segmentation
# ==========================================
cross = pd.crosstab(df["Segmentation"], df["KMeans_Cluster"], normalize="index")
cross.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Paired")
plt.title("Comparison: Existing Segments vs K-Means Clusters")
plt.show()

