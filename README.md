# Customer Segmentation Analysis

## 🎯 Project Overview
- This project improves upon an existing customer segmentation (A/B/C/D) by applying machine learning–based clustering.  
- The original segmentation lacked clear differences between groups, especially within overlapping professions (e.g., "Artist" appearing in multiple segments).  
- By applying **K-Means clustering** and comparing it with the legacy segmentation, we aim to provide more actionable insights for personalized marketing.

---

## 📊 Dataset
- Source: [Yemoonsa Bigdata Part5 Dataset](https://github.com/AnalyticsKnight/yemoonsaBigdata)  
- ~6,700 customers with the following fields:
  - `Gender`, `Ever_Married`, `Age`, `Graduated`, `Profession`,  `Work_Experience`, `Spending_Score`, `Family_Size`, `Segmentation`

---

## 🛠️ Methodology
1. **EDA**  
   - Distribution of demographics (age, profession, family size, etc.)  
   - Existing segmentation analysis  

2. **Preprocessing**  
   - Label encoding: `Gender`, `Ever_Married`, `Graduated`, `Segmentation`  
   - One-hot encoding: `Profession`  
   - Ordinal mapping: `Spending_Score` → {Low=0, Average=1, High=2}  

3. **Classification (Supervised)**  
   - RandomForestClassifier & XGBoost trained to predict **existing segmentation**  
   - Achieved ~49% accuracy  

4. **Clustering (Unsupervised)**  
   - K-Means applied with **Elbow Method + Silhouette Score** to determine optimal `k`  
   - Selected `k=4` clusters  
   - PCA projection for visualization  

5. **Evaluation**  
   - Compared accuracy of predicting **K-Means clusters** (~99.8%) vs. existing segmentation (~49%)  
   - Visual comparison with PCA and stacked bar plots  

---

## 💡 Key Insights
- **Legacy segmentation** was highly overlapping and hard to interpret.  
- **K-Means clusters** formed clearer groups:
  - **Cluster A**: 50s, Artists, high spending → Premium cultural offers  
  - **Cluster B**: 30s, Artists, low spending → Discount/promotion sensitive  
  - **Cluster C**: 70s, Lawyers, high spending → VIP services  
  - **Cluster D**: 20s, Healthcare, low spending → Budget-friendly products  

---

## 📈 Business Impact
- New segmentation enables **targeted marketing strategies** per cluster.  
- Example: Offering premium memberships to Cluster A yielded simulated **+12% ROI improvement**.  
- At-risk clusters (B, D) can be reactivated throu
