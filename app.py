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
st.subheader("Automatic Session Input for Prediction")

uploaded_file = st.file_uploader("Upload raw click-level test data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())
    
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
        'price_2': lambda x: 1 if (1 in x.values) else 0,
        'price_2_flag': 'mean',
        'page': pd.Series.nunique
    }).reset_index()

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
    category_counts.columns = [f'{(col)}_category_clicks' for col in category_counts.columns]
    expected_category_cols = ['blouses_category_clicks', 'sale_category_clicks', 
                         'skirts_category_clicks', 'trousers_category_clicks']
    for col in expected_category_cols:
        if col not in category_counts.columns:
            category_counts[col] = 0
    session_df = pd.merge(session_df, category_counts, on='session_id', how='left')
    
    categorical_cols = ['month', 'day', 'country_of_user', 'page_number']
    for col in categorical_cols:
        if col in session_df.columns:
            session_df[col] = session_df[col].astype(str)     
    if 'model_photography' in session_df.columns:
        session_df['model_photography'] = session_df['model_photography'].astype(float)
        
    numeric_cols = ['model_photography', 'unique_product_models', 'blouses_category_clicks',
                'skirts_category_clicks', 'num_revisits']
    for col in numeric_cols:
        if col in session_df.columns:
            session_df[col] = pd.to_numeric(session_df[col], errors='coerce').fillna(0)

    
    st.subheader("The following can be predicted ")
    tab1, tab2, tab3 = st.tabs([
        "Customer Purchase Prediction",
        "Total Price of Products Prediction",
        "Customer Type Prediction"
    ])
    
    with tab1:
        if st.button("Predict Customer Type"):
            X_cluster = session_df[cluster_features]
            session_df['cluster_label'] = cluster_pipeline.predict(X_cluster)
            cluster_results = session_df[['session_id', 'cluster_label']].copy()
            cluster_results['cluster_label'] = cluster_results['cluster_label'].map({
                0: "The person is window shopper",
                1: "This person is a regular customer"
            })
            st.dataframe(cluster_results)
        
    with tab2:
        if st.button("Predict Purchase"):
            X_cls = session_df[cls_features]
            session_df['purchased_pred'] = cls_pipeline.predict(X_cls)
            purchase_results = session_df[['session_id', 'purchased_pred']].copy()
            purchase_results['purchased_pred'] = purchase_results['purchased_pred'].map({
                1: "This customer purchased",
                0: "This customer did not purchase"
            })
            st.dataframe(purchase_results)
            
    with tab3:
        if st.button("Predict Total Price"):
            X_reg = session_df[reg_features]
            session_df['predicted_revenue'] = reg_pipeline.predict(X_reg)
            reg_results = session_df[['session_id', 'predicted_revenue']].copy()
            reg_results['predicted_revenue'] = reg_results['predicted_revenue'].round(2)
            st.dataframe(reg_results.rename(columns={'predicted_revenue': 'Total Price of Products'}))
    
st.divider()
st.header("Manual Session Input for Prediction")    

if "manual_sessions" not in st.session_state:
    st.session_state.manual_sessions = pd.DataFrame()

with st.form("manual_click_input", clear_on_submit=True):
    click_input = {}
    click_input['session_id'] = st.number_input(
        "Session ID", min_value=1, step=1,
        value=len(st.session_state.manual_sessions) + 1
    )
    click_input['order'] = st.number_input("Click order", min_value=1, step=1, value=1)
    click_input['price'] = st.number_input("Price", min_value=0, step=1, value=28)
    click_input['price_2'] = st.selectbox("Price above avg flag", [1, 2])
    click_input['year'] = st.number_input("Year", min_value=2008, max_value=2030, step=1, value=2025)
    click_input['month'] = st.selectbox("Month", list(range(1, 13)), index=0)
    click_input['day'] = st.selectbox("Day", list(range(1, 32)), index=0)
    click_input['country'] = st.selectbox("Country", [
        "Australia", "Austria", "Belgium", "British Virgin Islands", "Cayman Islands",
        "Christmas Island", "Croatia", "Cyprus", "Czech Republic", "Denmark",
        "Estonia", "unidentified", "Faroe Islands", "Finland", "France",
        "Germany", "Greece", "Hungary", "Iceland", "India",
        "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Mexico", "Netherlands", "Norway", "Poland", "Portugal",
        "Romania", "Russia", "San Marino", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Ukraine", "United Arab Emirates",
        "United Kingdom", "USA"
    ])
    click_input['page1_main_category'] = st.selectbox(
        "Main category viewed", ["trousers", "skirts", "blouses", "sale"]
    )
    click_input['page2_clothing_model'] = st.number_input(
        "Product model ID", min_value=1, max_value=217, step=1, value=1
    )
    click_input['colour'] = st.selectbox("Colour", [
        "beige", "black", "blue", "brown", "burgundy", "gray", "green",
        "navy blue", "of many colours", "olive", "pink", "red", "violet", "white"
    ])
    click_input['location'] = st.selectbox("Location", [
        "Top Left", "Top Middle", "Top Right", "Bottom Left", "Bottom Middle", "Bottom Right"
    ])
    click_input['model_photography'] = st.selectbox("Model photography", ["En Face", "Profile"])
    click_input['page'] = st.number_input("Page number", min_value=1, max_value=5, step=1, value=1)

    submitted = st.form_submit_button("Add detail")
    if submitted:
        new_click = pd.DataFrame([click_input])
        st.session_state.manual_sessions = pd.concat(
            [st.session_state.manual_sessions, new_click], ignore_index=True
        )

if not st.session_state.manual_sessions.empty:
    st.write("Collected Click-Level Data:")
    st.dataframe(st.session_state.manual_sessions)

    df = st.session_state.manual_sessions.copy()

    if 'price_2_flag' not in df.columns:
        df['price_2_flag'] = df['price_2'].map({1: 1, 2: 0})

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
        'price_2': lambda x: 1 if (1 in x.values) else 0,
        'price_2_flag': 'mean',
        'page': pd.Series.nunique
    }).reset_index()

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
    category_counts.columns = [f'{col}_category_clicks' for col in category_counts.columns]
    expected_category_cols = ['blouses_category_clicks', 'sale_category_clicks', 'skirts_category_clicks', 'trousers_category_clicks']
    for col in expected_category_cols:
        if col not in category_counts.columns:
            category_counts[col] = 0
    session_df = pd.merge(session_df, category_counts, on='session_id', how='left')

    st.subheader("Predictions on Manual Data")
    tab1, tab2, tab3 = st.tabs([
        "Customer Purchase Prediction",
        "Total Price of Products Prediction",
        "Customer Type Prediction"
    ])

    with tab1:
        if st.button("Predict Purchase (Manual)"):
            X_cls = session_df[cls_features]
            session_df['purchased_pred'] = cls_pipeline.predict(X_cls)
            results = session_df[['session_id', 'purchased_pred']].copy()
            results['purchased_pred'] = results['purchased_pred'].map({
                1: "This customer purchased",
                0: "This customer did not purchase"
            })
            st.dataframe(results)

    with tab2:
        if st.button("Predict Total Price (Manual)"):
            X_reg = session_df[reg_features]
            session_df['predicted_revenue'] = reg_pipeline.predict(X_reg)
            results = session_df[['session_id', 'predicted_revenue']].copy()
            results['predicted_revenue'] = results['predicted_revenue'].round(2)
            st.dataframe(results.rename(columns={'predicted_revenue': 'Total Price of Products'}))

    with tab3:
        if st.button("Predict Customer Type (Manual)"):
            X_cluster = session_df[cluster_features]
            session_df['cluster_label'] = cluster_pipeline.predict(X_cluster)
            results = session_df[['session_id', 'cluster_label']].copy()
            results['cluster_label'] = results['cluster_label'].map({
                0: "The person is window shopper",
                1: "This person is a regular customer"
            })
            st.dataframe(results)
