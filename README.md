# Clickstream Customer Conversion 

An end-to-end **Machine Learning + Streamlit project** for analyzing clickstream data and predicting customer conversions.  
This project demonstrates **clustering (segmentation), classification (purchase prediction), and regression (revenue prediction)** using raw click-level e-commerce data.

Live Demo: [Clickstream Customer Conversion App](https://ishaaq09-clickstream-customer-conversion-app-fgbewh.streamlit.app/)

---

## Problem Statement
E-commerce businesses track **clickstream data** (every click made by a customer).  
However, raw click-level logs are messy and hard to interpret.  
The goal of this project is to transform clickstream data into **actionable insights**:

1. **Clustering** → Segment users into groups based on browsing behavior.  
2. **Classification** → Predict whether a session will lead to a **purchase**.  
3. **Regression** → Estimate the **expected revenue** from a session.  

---

## Project Goals
- Convert raw click-level logs → **session-level features**.  
- Build ML models for:
  - **Clustering (KMeans)** → customer segmentation.  
  - **Classification (Random Forest)** → purchase prediction.  
  - **Regression (Linear Regression)** → revenue estimation.  
- Build a **Streamlit app** where:
  - Users can **upload raw CSVs** and get predictions.  
  - Users can **manually input session details** for prediction.  
  - Historical **visual insights** are displayed.  
- Deploy the app on **Streamlit Cloud**.  

---

## Dataset
Each row = a **click event**.  

**Columns in raw data:**
- `session_id` → unique ID for each browsing session  
- `order` → sequential click number  
- `price` → price of clicked item  
- `year, month, day` → timestamp info  
- `country` → user location  
- `page1_main_category` → category clicked  
- `page2_clothing_model` → product model ID  
- `colour`, `location`, `model_photography` → product details  
- `price_2` → encoded price band  
- `page` → page number visited  

**Derived session-level features:**
- `num_of_clicks`  
- `Total_revenue`  
- `unique_categories`  
- `unique_product_models`  
- `num_revisits`  
- `session_duration_min`  
- `is_weekend`, `is_end_of_month`  
- `cluster_label`, `purchased`, etc.

**Dataset Source**
- Source: [Link](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping)
- Train dataset: [Link](https://drive.google.com/file/d/1gcw7H1MJUeG91Wp-0h3AGyVabvnLDJiy/view?usp=drive_link)
- Test dataset: [Link](https://drive.google.com/file/d/1JFO3eQbUwPpwngzzdBLqWMNQ84HgPJ4E/view?usp=drive_link)

---

## Workflow
1. **Data Preprocessing**
   - Group raw clicks → session-level aggregation.  
   - Feature engineering: revisits, session duration, temporal flags.  

2. **Model Building**
   - **Clustering**: KMeans (k=2) with StandardScaler.  
   - **Classification**: Random Forest (after SMOTE for imbalance).  
   - **Regression**: Linear Regression baseline.  

3. **Model Serialization**
   - Models saved as `.pkl`.  
   - Feature lists saved as `.json` to fix ordering issues.  

4. **App Development**
   - `app.py` built with Streamlit.  
   - Sections: **Insights**, **CSV Upload Prediction**, **Manual Input Prediction**.  

5. **Deployment**
   - Deployed on **Streamlit Cloud**.  
   - Python 3.10 environment for compatibility.  

---

## Tech Stack
- **Language:** Python 3.10  
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - ML: `scikit-learn`, `xgboost`, `imbalanced-learn`  
  - App: `streamlit`  
- **Version Control:** Git + GitHub  
- **Deployment:** Streamlit Cloud  

---

## App Features
**Visual Insights** → charts for purchases, clicks, clusters.  
**Upload Mode** → upload raw click-level CSV, app aggregates & predicts.  
**Manual Input Mode** → enter session details → instant predictions.  
**Outputs:**  
- Cluster label (customer segment)  
- Purchase prediction (yes/no)  
- Predicted revenue (numeric)  

---

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Ishaaq09/Clickstream-customer-conversion.git
cd Clickstream-customer-conversion
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

## Challenges Faced

- Feature mismatch errors → fixed with JSON feature lists.
- Missing derived columns (price_2_flag) → added preprocessing in app.py.
- Deployment errors (matplotlib missing) → fixed by redeploying with Python 3.10.

## Results

- Clustering → Silhouette score ≈ 0.35 (acceptable for segmentation).
- Classification → Random Forest chosen (best F1-score).
- Regression → Linear Regression chosen (simple + interpretable).

## Future Improvements

- Add deep learning models (LSTMs for sequential clicks).
- Integrate real-time streaming data (Kafka, Spark).
- Include Explainable AI (SHAP, LIME) in app.
- Expand clustering to more groups for deeper segmentation.

## Author

**Name:** Ishaaq MM | Data Science Enthusiast | ML & AI Projects

**GitHub:** [Ishaaq09](https://github.com/Ishaaq09)

**LinkedIn:** [Ishaaq M M](https://www.linkedin.com/in/ishaaq-m-m)

## References

[Streamlit Docs](https://docs.streamlit.io/)

[Scikit-learn Docs](https://scikit-learn.org/)

[XGBoost Docs](https://xgboost.readthedocs.io/)

[Imbalanced-learn Docs](https://imbalanced-learn.org/)
