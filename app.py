import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

with open("models/classification.pkl", "rb") as f:
    cls_pipeline = pickle.load(f)

with open("models/regression.pkl", "rb") as f:
    reg_pipeline = pickle.load(f)

with open("models/clustering.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

with open("models/cls_features.json", "r") as f:
    cls_features = json.load(f)

with open("models/reg_features.json", "r") as f:
    reg_features = json.load(f)

with open("models/cluster_features.json", "r") as f:
    cluster_features = json.load(f)

st.set_page_config(layout="wide")
st.title("Clickstream Customer Conversion App")
st.divider()

visuals = pd.read_csv("data/df_for_visuals.csv")
st.header("Insights from past sessions")

st.write("")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(data=visuals, x="purchased", ax=ax)
    ax.set_title("Purchased vs Not Purchased")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=visuals, x='day_of_week', hue='purchased', ax=ax)
    ax.set_title("Purchases by Day of Week")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(data=visuals, x="purchased", y="num_of_clicks", ax=ax)
    ax.set_title("Number of Clicks vs Purchased")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(data=visuals, x='session_duration_min', y='num_of_clicks', hue='purchased', ax=ax)
    ax.set_title("Session Duration vs Number of Clicks")
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    sns.boxplot(data=visuals, x="purchased", y="session_duration_min", ax=ax)
    ax.set_title("Session Duration vs Purchased")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=visuals, x='cluster_label', hue='purchased', ax=ax)
    ax.set_title("Cluster vs Purchased")
    st.pyplot(fig)


st.divider()
st.header("Predictions")
st.subheader("Manual Session Input for Prediction")

# --- File upload ---
uploaded_file = st.file_uploader("Upload raw click-level test data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # --- SESSION-LEVEL AGGREGATION ---
    
    if 'price_2_flag' not in df.columns:
        df['price_2_flag'] = df['price_2'].map({1:1, 2:0})
    
    session_df = df.groupby('session_id').agg({
        'order': 'count',
        'price': 'sum',
        'year': 'first',
        'month': 'first',
        'day': 'first',
        'country': 'first',
        'page1_main_category': pd.Series.nunique,
        'page2_clothing_model': pd.Series.nunique,
        'colour': pd.Series.nunique,
        'location': pd.Series.nunique,
        'model_photography': pd.Series.nunique,
        'price_2': lambda x: 1 if (1 in x.values) else 2,
        'price_2_flag': 'mean',
        'page': pd.Series.nunique
    }).reset_index()

    # Rename columns to match training session_data
    session_df.columns = [
        'session_id','num_of_clicks','Total_revenue','year','month','day','country_of_user',
        'unique_categories','unique_product_models','colour_of_product','location_of_product',
        'model_photography','price_above_avg','ratio_of_above_avg_price','page_number'
    ]

    session_df['date'] = pd.to_datetime(df[['year','month','day']])
    session_df['day_of_week'] = session_df['date'].dt.weekday
    session_df['is_weekend'] = session_df['day_of_week'].isin([5,6]).astype(int)
    session_df['is_end_of_month'] = (session_df['day'] >= 25).astype(int)
    session_df['week_num'] = session_df['date'].dt.isocalendar().week

    df['click_time'] = pd.to_datetime(df[['year','month','day']]) + pd.to_timedelta(df['order'], unit='m')
    session_times = df.groupby('session_id')['click_time'].agg(['min','max']).reset_index()
    session_times['session_duration_min'] = (session_times['max'] - session_times['min']).dt.total_seconds()/60
    session_df = pd.merge(session_df, session_times[['session_id','session_duration_min']], on='session_id', how='left')

    model_repeat = df.groupby(['session_id','page2_clothing_model']).size().reset_index(name='count')
    revisited = model_repeat[model_repeat['count']>1].groupby('session_id').size().reset_index(name='num_revisits')
    session_df = pd.merge(session_df, revisited, on='session_id', how='left')
    session_df['num_revisits'] = session_df['num_revisits'].fillna(0)

    category_counts = df.groupby(['session_id','page1_main_category']).size().unstack(fill_value=0)
    category_counts.columns = [f'category_{int(col)}_clicks' for col in category_counts.columns]
    session_df = pd.merge(session_df, category_counts, on='session_id', how='left')
    for col in category_counts.columns:
        if col not in session_df.columns:
            session_df[col] = 0

    X_cluster = session_df[cluster_features]
    session_df['cluster_label'] = cluster_pipeline.predict(X_cluster)

    X_cls = session_df[cls_features]
    session_df['purchased_pred'] = cls_pipeline.predict(X_cls)
        
    X_reg = session_df[reg_features]
    session_df['predicted_revenue'] = reg_pipeline.predict(X_reg)
    
    st.header("Predictions")
    combined_cols = ['session_id', 'predicted_revenue', 'cluster_label', 'purchased_pred']
    st.dataframe(session_df[combined_cols])

    st.success("Predictions completed successfully!")

st.divider()
st.subheader("Manual Session Input for Prediction")

session_input = {}
session_input['order'] = st.number_input("Number of clicks (order)", min_value=1, step=1, value=1)
session_input['price'] = st.number_input("Total price of clicks", min_value=0, step=1, value=28)
session_input['year'] = st.number_input("Year", min_value=2008, max_value=2030, step=1, value=2025)
session_input['month'] = st.number_input("Month", min_value=1, max_value=12, step=1, value=1)
session_input['day'] = st.number_input("Day", min_value=1, max_value=31, step=1, value=1)
session_input['country_of_user'] = st.number_input("Country (encoded as number)", min_value=1, max_value=47, step=1)
session_input['page1_main_category'] = st.number_input("Main category viewed (encoded)", min_value=1, max_value=4, step=1)
session_input['page2_clothing_model'] = st.number_input("Product models viewed", min_value=1, max_value=217, step=1)
session_input['colour'] = st.number_input("Unique colours", min_value=1, max_value=14, step=1)
session_input['location'] = st.number_input("Unique locations", min_value=1, max_value=6, step=1)
session_input['model_photography'] = st.number_input("Unique model photography", min_value=0, max_value=1, step=1, value=0)
session_input['price_2_flag'] = st.number_input("Price above avg flag (0 or 1)", min_value=0, max_value=1, step=1, value=0)
session_input['page'] = st.number_input("Page number", min_value=1, max_value=5, step=1, value=1)

user_df = pd.DataFrame([session_input])

user_df['Total_revenue'] = user_df['price']
user_df['num_of_clicks'] = user_df['order']
user_df['unique_product_models'] = user_df['page2_clothing_model']
user_df['num_revisits'] = 0
user_df['session_duration_min'] = 1
user_df['category_1_clicks'] = 0
user_df['category_2_clicks'] = 0
user_df['category_3_clicks'] = 0
user_df['category_4_clicks'] = 0
user_df['page_number'] = user_df['page'] 

user_df['date'] = pd.to_datetime(user_df[['year','month','day']])
user_df['day_of_week'] = user_df['date'].dt.weekday
user_df['is_weekend'] = user_df['day_of_week'].isin([5,6]).astype(int)
user_df['is_end_of_month'] = (user_df['day'] >= 25).astype(int)

if st.button("Predict for this session"):
    X_cluster = user_df[cluster_features]
    user_df['cluster_label'] = cluster_pipeline.predict(X_cluster)
    
    X_cls = user_df[cls_features]
    user_df['purchased_pred'] = cls_pipeline.predict(X_cls)
    
    X_reg = user_df[reg_features]
    user_df['predicted_revenue'] = reg_pipeline.predict(X_reg)
    
    st.header("Predictions")
    combined_cols = ['cluster_label', 'predicted_revenue', 'purchased_pred']
    st.dataframe(user_df[combined_cols])

    st.success("Predictions completed successfully!")