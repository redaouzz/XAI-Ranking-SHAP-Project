# XAI-Ranking-SHAP-Project

# Explainable AI for Learning-to-Rank with RankingSHAP

This repository contains the implementation and experiments of a project on **explainability in ranking models**, focusing on **RankingSHAP**.  
The study uses the **MQ2008 dataset** from the LETOR 4.0 benchmark and explores how to bridge **local** and **global** feature attributions in ranking tasks.

---

## 📖 Project Overview
Learning-to-Rank (LTR) models are widely applied in search engines and recommendation systems. Despite their effectiveness, these models often lack interpretability.  
Traditional post-hoc explainability methods (e.g., LIME, SHAP) provide **local explanations** for single predictions, but fail to generalize to a **global understanding** of model behavior.

This project addresses this limitation by:
- Implementing **RankingSHAP**, an adaptation of SHAP for ranking.  
- Aggregating **local feature attributions** across queries to obtain **global insights**.  
- Validating explanations by **retraining models on top-k features** and comparing them against:  
  - Full feature set  
  - Random-k feature subsets  


---

## 🚀 Notebook
- **`XAI_RankingSHAP.ipynb`**  
  Provides the complete pipeline:
  - Parsing LETOR 4.0 / MQ2008 dataset
  - Training a ranking model (LambdaMART via LightGBM)
  - Computing local feature attributions with RankingSHAP
  - Aggregating into global explanations
  - Retraining experiments with top-k features

You can run it directly in **Google Colab**:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/<REPO>/blob/main/XAI_RankingSHAP.ipynb)

---

## 📊 Main Results
- Aggregated feature attributions provide **global insights** consistent with domain expectations.  
- Retraining with top-k features preserves ranking performance close to the full model.  
- Random-k feature subsets degrade performance significantly, validating the **faithfulness** of RankingSHAP explanations.  

**Evaluation Metrics:**
- NDCG@10  
- MAP (Mean Average Precision)  
- Precision@10  

---

## 📑 Report
The full scientific report is available here (if uploaded):  
📄 [XAI_RankingSHAP_Report.pdf](report/Hybrid_Ranking_Shap_Report.pdf)

---



## Technologies Used

Python 3.10

LightGBM (LambdaMART)

NumPy / Pandas

Matplotlib

SHAP

Google Colab
