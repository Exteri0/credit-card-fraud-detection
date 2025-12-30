"""
Credit Card Fraud Detection - Streamlit Dashboard
=================================================
Interactive visualization for ML models and MongoDB queries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_path = 'models'
    
    try:
        models['kmeans'] = joblib.load(f'{model_path}/kmeans_model.pkl')
        models['kmeans_threshold'] = joblib.load(f'{model_path}/kmeans_threshold.pkl')
        models['logistic_regression'] = joblib.load(f'{model_path}/logistic_regression_model.pkl')
        models['naive_bayes'] = joblib.load(f'{model_path}/naive_bayes_model.pkl')
        models['decision_tree'] = joblib.load(f'{model_path}/decision_tree_model.pkl')
        models['scaler'] = joblib.load(f'{model_path}/scaler.pkl')
        models['label_encoder'] = joblib.load(f'{model_path}/label_encoder.pkl')
        models['feature_cols'] = joblib.load(f'{model_path}/feature_cols.pkl')
        models['category_mapping'] = joblib.load(f'{model_path}/category_mapping.pkl')
        
        if os.path.exists(f'{model_path}/model_comparison.pkl'):
            models['comparison'] = pd.read_pickle(f'{model_path}/model_comparison.pkl')
        
        if os.path.exists(f'{model_path}/apriori_rules.pkl'):
            models['apriori_rules'] = pd.read_pickle(f'{model_path}/apriori_rules.pkl')
            
        if os.path.exists(f'{model_path}/test_data.pkl'):
            models['test_data'] = pd.read_pickle(f'{model_path}/test_data.pkl')
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the notebook first to train and save the models.")
        
    return models

@st.cache_resource
def get_mongodb_connection():
    """Create MongoDB connection"""
    try:
        username = os.getenv('MONGODB_USERNAME')
        password = os.getenv('MONGODB_PASSWORD')
        cluster = os.getenv('MONGODB_CLUSTER')
        database = os.getenv('MONGODB_DATABASE')
        collection_name = os.getenv('MONGODB_COLLECTION', 'transactions')
        
        if not all([username, password, cluster, database]):
            return None, None, "MongoDB credentials not found in .env file"
        
        # Determine protocol (mongodb+srv:// for Atlas, mongodb:// for local/standard)
        protocol = "mongodb+srv"
        if ":" in cluster or "localhost" in cluster:
            protocol = "mongodb"
            
        if protocol == "mongodb+srv":
            uri = f"mongodb+srv://{username}:{password}@{cluster}/{database}?retryWrites=true&w=majority"
            client = MongoClient(uri, server_api=ServerApi('1'))
        else:
            if username and password:
                uri = f"mongodb://{username}:{password}@{cluster}/{database}?authSource=admin"
            else:
                uri = f"mongodb://{cluster}/{database}"
            client = MongoClient(uri)
        
        # Test connection
        client.admin.command('ping')
        
        db = client[database]
        collection = db[collection_name]
        
        return client, collection, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_csv_data():
    """Load data from CSV"""
    try:
        df = pd.read_csv('Dataset/credit_card_fraud_10k.csv')
        return df
    except:
        return None

def decode_merchant_category(df, models):
    """Decode merchant category from encoded values to original names"""
    if df is None or len(df) == 0:
        return df
        
    category_mapping = models.get('category_mapping', {})
    if not category_mapping:
        return df
    
    # Create reverse mapping (encoded -> original)
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    
    # Check which column exists and decode
    if 'merchant_category_encoded' in df.columns:
        # Check if values are scaled (floats in range like -1 to 1) or integers (0-4)
        sample_vals = df['merchant_category_encoded'].dropna().head(100)
        if len(sample_vals) > 0:
            # If values are integers in valid range, decode them
            if sample_vals.dtype in ['int64', 'int32'] or (sample_vals.abs().max() <= 10 and (sample_vals % 1 == 0).all()):
                df['merchant_category'] = df['merchant_category_encoded'].astype(int).map(reverse_mapping)
            # Otherwise they're scaled values - cannot decode
            else:
                # Note: These are StandardScaler normalized values, cannot map back to categories
                df['merchant_category'] = 'Scaled_Cat_' + df['merchant_category_encoded'].astype(str).str[:4]
    elif 'merchant_category' in df.columns:
        # Check if merchant_category contains encoded integers
        if df['merchant_category'].dtype in ['int64', 'int32', 'float64']:
            # Try to convert to int and map
            try:
                df['merchant_category'] = df['merchant_category'].astype(int).map(reverse_mapping)
            except:
                pass
        # If it's already strings, keep them as is
    
    return df

# ============================================
# Tab Content Functions
# ============================================

def render_overview_tab():
    """Render the overview/home tab"""
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    models = load_models()
    
    # Try to load balanced data from MongoDB first, fallback to CSV
    client, collection, error = get_mongodb_connection()
    
    if not error and collection is not None:
        try:
            # Load balanced data from MongoDB
            results = list(collection.find().limit(20000))
            df_balanced = pd.DataFrame(results)
            if '_id' in df_balanced.columns:
                df_balanced = df_balanced.drop('_id', axis=1)
            
            st.info(f"📊 Showing **Balanced Training Data** from MongoDB ({len(df_balanced):,} records)")
            df = df_balanced
        except Exception as e:
            st.warning(f"Could not load from MongoDB: {e}. Loading original CSV data.")
            df = load_csv_data()
    else:
        # Fallback to CSV data
        df = load_csv_data()
        if df is not None:
            st.info("📊 Showing **Original Unbalanced Data** from CSV (MongoDB not connected)")
    
    # Display feature encoding reference
    with st.expander("📋 Feature Encoding Reference (Click to expand)"):
        st.markdown("### Categorical Feature Encodings")
        col_enc1, col_enc2 = st.columns(2)
        
        with col_enc1:
            st.markdown("**Merchant Category Encoding:**")
            category_mapping = models.get('category_mapping', {})
            if category_mapping:
                enc_df = pd.DataFrame([
                    {"Original Value": cat, "Encoded Value": code} 
                    for cat, code in sorted(category_mapping.items(), key=lambda x: x[1])
                ])
                st.dataframe(enc_df, use_container_width=True, hide_index=True)
        
        with col_enc2:
            st.markdown("**Binary Feature Encodings:**")
            binary_enc = pd.DataFrame([
                {"Feature": "foreign_transaction", "No": 0, "Yes": 1},
                {"Feature": "location_mismatch", "No": 0, "Yes": 1},
                {"Feature": "is_fraud", "Not Fraud": 0, "Fraud": 1}
            ])
            st.dataframe(binary_enc, use_container_width=True, hide_index=True)
            
            st.markdown("**Note:** Balanced data has **StandardScaler normalized** numerical features")
    
    if df is not None:
        # Decode merchant categories to show original names
        df = decode_merchant_category(df, models)
        
        # Debug: Show available columns
        with st.expander("🔍 Debug: Available Data Columns"):
            st.write("Columns in dataset:", list(df.columns))
            if 'merchant_category' in df.columns:
                st.write("Sample merchant_category values:", df['merchant_category'].unique()[:10])
            if 'merchant_category_encoded' in df.columns:
                st.write("Sample merchant_category_encoded values:", df['merchant_category_encoded'].unique()[:10])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df['is_fraud'].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            avg_amount = df['amount'].mean()
            st.metric("Avg Transaction", f"${avg_amount:.2f}")
        
        st.markdown("---")
        
        # Data distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Distribution")
            fig = px.pie(
                values=df['is_fraud'].value_counts().values,
                names=['Not Fraud', 'Fraud'],
                color_discrete_sequence=['#3498db', '#e74c3c'],
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Transaction Amount Distribution")
            fig = px.histogram(
                df, x='amount', 
                color='is_fraud',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                labels={'is_fraud': 'Fraud'},
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud by Merchant Category")
            if 'merchant_category' in df.columns:
                # Check if categories are decoded names or scaled values
                sample_cat = df['merchant_category'].iloc[0] if len(df) > 0 else None
                is_scaled = isinstance(sample_cat, str) and 'Scaled_Cat_' in str(sample_cat)
                
                if not is_scaled:
                    fraud_by_cat = df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
                    fraud_df = pd.DataFrame({'Category': fraud_by_cat.index, 'Fraud Rate': fraud_by_cat.values * 100})
                    fig = px.bar(fraud_df, x='Category', y='Fraud Rate',
                                color='Fraud Rate',
                                color_continuous_scale='Reds')
                    fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
                    fig.update_yaxes(title='Fraud Rate (%)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("💡 Merchant categories in balanced data are StandardScaler normalized. Cannot decode to original names. Use CSV data (when MongoDB unavailable) to see category names.")
            elif 'merchant_category_encoded' in df.columns:
                # Use encoded values directly
                st.info("💡 Showing scaled merchant category values from balanced training data (StandardScaler normalized)")
                fraud_by_cat = df.groupby('merchant_category_encoded')['is_fraud'].mean().sort_values(ascending=False).head(10)
                fraud_df = pd.DataFrame({'Category (Scaled)': fraud_by_cat.index, 'Fraud Rate': fraud_by_cat.values * 100})
                fig = px.bar(fraud_df, x='Category (Scaled)', y='Fraud Rate',
                            color='Fraud Rate',
                            color_continuous_scale='Reds')
                fig.update_layout(height=400, showlegend=False)
                fig.update_yaxes(title='Fraud Rate (%)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Merchant category data not available")
        
        with col2:
            st.subheader("Fraud by Transaction Hour")
            if 'transaction_hour' in df.columns:
                fraud_by_hour = df.groupby('transaction_hour')['is_fraud'].mean()
                fig = px.line(
                    x=fraud_by_hour.index,
                    y=fraud_by_hour.values * 100,
                    labels={'x': 'Hour', 'y': 'Fraud Rate (%)'},
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Transaction hour data not available")
    else:
        st.warning("Could not load data. Please ensure the CSV file exists.")


def render_kmeans_tab(models):
    """Render K-Means clustering visualization"""
    st.header("🔮 K-Means Anomaly Detection")
    
    if 'kmeans' not in models:
        st.warning("K-Means model not loaded. Please run the notebook first.")
        return
    
    st.markdown("""
    K-Means clustering is used for **unsupervised anomaly detection**. 
    Transactions that are far from cluster centers are flagged as potential fraud.
    """)
    
    df = load_csv_data()
    if df is not None:
        # Get predictions
        le = models.get('label_encoder')
        scaler = models.get('scaler')
        feature_cols = models.get('feature_cols')
        
        df_temp = df.copy()
        df_temp['merchant_category_encoded'] = le.transform(df_temp['merchant_category'])
        
        X = df_temp[feature_cols]
        X_scaled = scaler.transform(X)
        
        kmeans = models['kmeans']
        threshold = models['kmeans_threshold']
        
        distances = kmeans.transform(X_scaled).min(axis=1)
        predictions = (distances > threshold).astype(int)
        
        df_temp['cluster'] = kmeans.predict(X_scaled)
        df_temp['distance'] = distances
        df_temp['predicted_fraud'] = predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Distribution")
            fig = px.scatter(
                df_temp, 
                x='amount', 
                y='device_trust_score',
                color='cluster',
                symbol='is_fraud',
                opacity=0.6,
                labels={'cluster': 'Cluster', 'is_fraud': 'Actual Fraud'},
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Anomaly Score Distribution")
            fig = px.histogram(
                df_temp, 
                x='distance',
                color='is_fraud',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                labels={'is_fraud': 'Actual Fraud'},
                barmode='overlay',
                opacity=0.7
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold: {threshold:.2f}")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.subheader("Model Performance")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(df_temp['is_fraud'], predictions):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(df_temp['is_fraud'], predictions, zero_division=0):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(df_temp['is_fraud'], predictions, zero_division=0):.4f}")
        with col4:
            st.metric("F1-Score", f"{f1_score(df_temp['is_fraud'], predictions, zero_division=0):.4f}")


def render_apriori_tab(models):
    """Render Apriori association rules"""
    st.header("🔗 Apriori - Association Rules")
    
    st.markdown("""
    Apriori algorithm discovers **frequent patterns** in fraudulent transactions.
    These rules help identify combinations of factors that commonly appear in fraud cases.
    """)
    
    if 'apriori_rules' not in models:
        st.warning("Apriori rules not found. Please run the notebook first.")
        
        # Show sample patterns from data
        df = load_csv_data()
        if df is not None:
            st.subheader("Fraud Pattern Analysis (from raw data)")
            
            fraud_df = df[df['is_fraud'] == 1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Most Common Merchant Categories in Fraud:**")
                cat_counts = fraud_df['merchant_category'].value_counts()
                fig = px.bar(x=cat_counts.index, y=cat_counts.values,
                           labels={'x': 'Category', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Foreign Transaction in Fraud Cases:**")
                foreign_counts = fraud_df['foreign_transaction'].value_counts()
                fig = px.pie(values=foreign_counts.values, 
                           names=['Domestic', 'Foreign'],
                           color_discrete_sequence=['#3498db', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
        return
    
    rules = models['apriori_rules']
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Min Support", 0.0, 1.0, 0.1, 0.05)
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
    with col3:
        min_lift = st.slider("Min Lift", 0.0, 5.0, 1.0, 0.1)
    
    filtered_rules = rules[
        (rules['support'] >= min_support) & 
        (rules['confidence'] >= min_confidence) & 
        (rules['lift'] >= min_lift)
    ]
    
    st.subheader(f"Association Rules ({len(filtered_rules)} rules)")
    
    if len(filtered_rules) > 0:
        # Display rules
        display_rules = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.dataframe(display_rules.head(20), use_container_width=True)
        
        # Visualization
        fig = px.scatter(
            filtered_rules,
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            color_continuous_scale='Viridis',
            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
        )
        fig.update_layout(title="Rules: Support vs Confidence (size = Lift)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rules match the current filters. Try adjusting the thresholds.")


def render_logistic_regression_tab(models):
    """Render Logistic Regression results"""
    st.header("📈 Logistic Regression")
    
    if 'logistic_regression' not in models:
        st.warning("Logistic Regression model not loaded. Please run the notebook first.")
        return
    
    st.markdown("""
    Logistic Regression is a **linear classification model** that predicts the probability 
    of fraud based on transaction features.
    """)
    
    model = models['logistic_regression']
    feature_cols = models.get('feature_cols', [])
    
    # Feature coefficients
    if hasattr(model, 'coef_'):
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig = px.bar(
            coef_df, 
            x='Coefficient', 
            y='Feature',
            orientation='h',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if 'comparison' in models:
        st.subheader("Model Performance")
        metrics = models['comparison'].loc['Logistic Regression']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        if 'ROC-AUC' in metrics:
            st.metric("ROC-AUC Score", f"{metrics['ROC-AUC']:.4f}")
    
    # Interactive prediction
    st.subheader("🔮 Make a Prediction")
    render_prediction_form(models, 'logistic_regression')


def render_naive_bayes_tab(models):
    """Render Naive Bayes results"""
    st.header("📊 Naïve Bayes Classifier")
    
    if 'naive_bayes' not in models:
        st.warning("Naive Bayes model not loaded. Please run the notebook first.")
        return
    
    st.markdown("""
    Naïve Bayes is a **probabilistic classifier** based on Bayes' theorem. 
    It assumes feature independence and works well with high-dimensional data.
    """)
    
    model = models['naive_bayes']
    
    # Model parameters
    if hasattr(model, 'var_'):
        st.subheader("Feature Variances by Class")
        feature_cols = models.get('feature_cols', [])
        
        var_df = pd.DataFrame({
            'Feature': feature_cols,
            'Variance (Not Fraud)': model.var_[0],
            'Variance (Fraud)': model.var_[1]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Not Fraud', x=feature_cols, y=model.var_[0], marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Fraud', x=feature_cols, y=model.var_[1], marker_color='#e74c3c'))
        fig.update_layout(barmode='group', height=400, title='Feature Variances by Class')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if 'comparison' in models:
        st.subheader("Model Performance")
        metrics = models['comparison'].loc['Naïve Bayes']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
    
    # Interactive prediction
    st.subheader("🔮 Make a Prediction")
    render_prediction_form(models, 'naive_bayes')


def render_decision_tree_tab(models):
    """Render Decision Tree results"""
    st.header("🌳 Decision Tree Classifier")
    
    if 'decision_tree' not in models:
        st.warning("Decision Tree model not loaded. Please run the notebook first.")
        return
    
    st.markdown("""
    Decision Trees create **interpretable rules** for classification. 
    They're particularly useful for understanding which features drive fraud predictions.
    """)
    
    model = models['decision_tree']
    feature_cols = models.get('feature_cols', [])
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tree parameters
    st.subheader("Tree Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Depth", model.max_depth if model.max_depth else "None")
    with col2:
        st.metric("Min Samples Split", model.min_samples_split)
    with col3:
        st.metric("Min Samples Leaf", model.min_samples_leaf)
    
    # Model performance
    if 'comparison' in models:
        st.subheader("Model Performance")
        metrics = models['comparison'].loc['Decision Tree']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
    
    # Interactive prediction
    st.subheader("🔮 Make a Prediction")
    render_prediction_form(models, 'decision_tree')


def render_prediction_form(models, model_key):
    """Render prediction form for a model"""
    category_mapping = models.get('category_mapping', {})
    categories = list(category_mapping.keys())
    
    # Display category encoding reference
    with st.expander("📋 Category Encoding Reference"):
        encoding_df = pd.DataFrame([
            {"Category": cat, "Encoded Value": code} 
            for cat, code in category_mapping.items()
        ])
        st.dataframe(encoding_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0, key=f"{model_key}_amount")
        transaction_hour = st.slider("Transaction Hour", 0, 23, 12, key=f"{model_key}_hour")
        # Show category with its encoded value
        merchant_category = st.selectbox(
            "Merchant Category", 
            categories, 
            format_func=lambda x: f"{x} (encoded: {category_mapping.get(x, 0)})",
            key=f"{model_key}_cat"
        )
        foreign_transaction = st.selectbox("Foreign Transaction", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)", key=f"{model_key}_foreign")
    
    with col2:
        location_mismatch = st.selectbox("Location Mismatch", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)", key=f"{model_key}_location")
        device_trust_score = st.slider("Device Trust Score", 0, 100, 50, key=f"{model_key}_trust")
        velocity_last_24h = st.slider("Velocity Last 24h", 0, 10, 2, key=f"{model_key}_velocity")
        cardholder_age = st.slider("Cardholder Age", 18, 90, 35, key=f"{model_key}_age")
    
    if st.button("Predict", key=f"{model_key}_predict"):
        # Prepare input
        input_data = pd.DataFrame({
            'amount': [amount],
            'transaction_hour': [transaction_hour],
            'merchant_category_encoded': [category_mapping.get(merchant_category, 0)],
            'foreign_transaction': [foreign_transaction],
            'location_mismatch': [location_mismatch],
            'device_trust_score': [device_trust_score],
            'velocity_last_24h': [velocity_last_24h],
            'cardholder_age': [cardholder_age]
        })
        
        scaler = models.get('scaler')
        if scaler:
            input_scaled = scaler.transform(input_data)
            
            model = models.get(model_key)
            if model:
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
                
                if prediction == 1:
                    st.error(f"⚠️ **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **Transaction appears legitimate**")
                
                if proba is not None:
                    st.write(f"Fraud Probability: **{proba[1]*100:.2f}%**")


def render_comparison_tab(models):
    """Render model comparison"""
    st.header("📊 Model Comparison")
    
    if 'comparison' not in models:
        st.warning("Model comparison data not found. Please run the notebook first.")
        return
    
    comparison_df = models['comparison']
    
    st.subheader("Performance Metrics")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrics Comparison")
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df.index,
                y=comparison_df[metric],
                marker_color=colors[i]
            ))
        
        fig.update_layout(barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC-AUC Comparison")
        roc_df = comparison_df[comparison_df['ROC-AUC'].notna()]
        
        if len(roc_df) > 0:
            fig = px.bar(
                x=roc_df.index,
                y=roc_df['ROC-AUC'],
                color=roc_df['ROC-AUC'],
                color_continuous_scale='Viridis',
                labels={'x': 'Model', 'y': 'ROC-AUC Score'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Best model recommendation
    st.subheader("🏆 Best Model Recommendation")
    best_f1 = comparison_df['F1-Score'].idxmax()
    best_recall = comparison_df['Recall'].idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Best by F1-Score:** {best_f1} ({comparison_df.loc[best_f1, 'F1-Score']:.4f})")
    with col2:
        st.info(f"**Best by Recall:** {best_recall} ({comparison_df.loc[best_recall, 'Recall']:.4f})")
    
    st.markdown("""
    > **Note:** For fraud detection, **Recall** is often more important than precision 
    > because missing a fraud case (false negative) is typically more costly than 
    > flagging a legitimate transaction (false positive).
    """)


def render_mongodb_tab():
    """Render MongoDB query interface"""
    st.header("🗄️ MongoDB Query Interface")
    
    # Load models for encoding reference
    models = load_models()
    
    # Display encoding reference for MongoDB queries
    with st.expander("📋 Data Encoding Reference (Important for Queries)"):
        st.markdown("""
        **Note:** The data stored in MongoDB is the **balanced, preprocessed training data** with encoded values.
        Use the encoded values below when constructing queries.
        """)
        
        col_enc1, col_enc2 = st.columns(2)
        
        with col_enc1:
            st.markdown("**Merchant Category (Encoded):**")
            category_mapping = models.get('category_mapping', {})
            if category_mapping:
                enc_df = pd.DataFrame([
                    {"Category": cat, "Encoded Value": code} 
                    for cat, code in sorted(category_mapping.items(), key=lambda x: x[1])
                ])
                st.dataframe(enc_df, use_container_width=True, hide_index=True)
        
        with col_enc2:
            st.markdown("**Binary Features:**")
            st.markdown("- `foreign_transaction`: 0 = No, 1 = Yes")
            st.markdown("- `location_mismatch`: 0 = No, 1 = Yes")
            st.markdown("- `is_fraud`: 0 = Not Fraud, 1 = Fraud")
            st.markdown("")
            st.markdown("**Note:** Numerical features are **StandardScaler normalized**")
    
    client, collection, error = get_mongodb_connection()
    
    if error:
        st.error(f"MongoDB Connection Error: {error}")
        st.info("Please ensure your MongoDB credentials are set in the .env file.")
        
        # Fallback to CSV data
        st.subheader("📊 Query from Local Data (CSV Fallback)")
        df = load_csv_data()
        if df is not None:
            render_local_query_interface(df, models)
        return
    
    st.success("✅ Connected to MongoDB!")
    
    # Query options
    st.subheader("Query Options")
    
    query_type = st.selectbox(
        "Select Query Type",
        ["All Transactions", "Fraud Cases Only", "Custom Filter", "Aggregation"]
    )
    
    if query_type == "All Transactions":
        limit = st.slider("Limit Results", 10, 1000, 100)
        
        if st.button("Execute Query"):
            with st.spinner("Querying MongoDB..."):
                results = list(collection.find().limit(limit))
                df = pd.DataFrame(results)
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Decode merchant categories
                df = decode_merchant_category(df, models)
                
                st.dataframe(df, use_container_width=True)
                st.write(f"Retrieved {len(df)} records")
    
    elif query_type == "Fraud Cases Only":
        limit = st.slider("Limit Results", 10, 500, 100)
        
        if st.button("Execute Query"):
            with st.spinner("Querying MongoDB..."):
                results = list(collection.find({"is_fraud": 1}).limit(limit))
                df = pd.DataFrame(results)
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Decode merchant categories
                df = decode_merchant_category(df, models)
                
                st.dataframe(df, use_container_width=True)
                st.write(f"Retrieved {len(df)} fraud cases")
    
    elif query_type == "Custom Filter":
        st.subheader("Filter Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount filter
            st.write("**Amount Range**")
            min_amount = st.number_input("Min Amount", 0.0, 10000.0, 0.0)
            max_amount = st.number_input("Max Amount", 0.0, 10000.0, 10000.0)
            
            # Fraud filter
            fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud Only", "Non-Fraud Only"])
        
        with col2:
            # Category filter
            categories = ["All", "Electronics", "Travel", "Grocery", "Food", "Clothing"]
            category_filter = st.selectbox("Merchant Category", categories)
            
            # Trust score
            st.write("**Device Trust Score**")
            min_trust = st.slider("Min Trust Score", 0, 100, 0)
            max_trust = st.slider("Max Trust Score", 0, 100, 100)
        
        # Foreign transaction
        foreign_filter = st.selectbox("Foreign Transaction", ["All", "Yes", "No"])
        
        limit = st.slider("Limit Results", 10, 1000, 100)
        
        if st.button("Execute Query"):
            # Build query
            query = {
                "amount": {"$gte": min_amount, "$lte": max_amount},
                "device_trust_score": {"$gte": min_trust, "$lte": max_trust}
            }
            
            if fraud_filter == "Fraud Only":
                query["is_fraud"] = 1
            elif fraud_filter == "Non-Fraud Only":
                query["is_fraud"] = 0
            
            if category_filter != "All":
                query["merchant_category"] = category_filter
            
            if foreign_filter == "Yes":
                query["foreign_transaction"] = 1
            elif foreign_filter == "No":
                query["foreign_transaction"] = 0
            
            st.code(f"Query: {query}")
            
            with st.spinner("Querying MongoDB..."):
                results = list(collection.find(query).limit(limit))
                df = pd.DataFrame(results)
                if len(df) > 0 and '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Decode merchant categories if they're encoded
                models = load_models()
                df = decode_merchant_category(df, models)
                
                st.dataframe(df, use_container_width=True)
                st.write(f"Retrieved {len(df)} records")
    
    elif query_type == "Aggregation":
        agg_type = st.selectbox(
            "Aggregation Type",
            ["Fraud Count by Category", "Average Amount by Fraud Status", "Hourly Transaction Count"]
        )
        
        if st.button("Execute Aggregation"):
            with st.spinner("Running aggregation..."):
                if agg_type == "Fraud Count by Category":
                    pipeline = [
                        {"$group": {
                            "_id": {"category": "$merchant_category", "fraud": "$is_fraud"},
                            "count": {"$sum": 1}
                        }},
                        {"$sort": {"_id.category": 1, "_id.fraud": 1}}
                    ]
                elif agg_type == "Average Amount by Fraud Status":
                    pipeline = [
                        {"$group": {
                            "_id": "$is_fraud",
                            "avg_amount": {"$avg": "$amount"},
                            "count": {"$sum": 1}
                        }}
                    ]
                else:  # Hourly Transaction Count
                    pipeline = [
                        {"$group": {
                            "_id": "$transaction_hour",
                            "count": {"$sum": 1},
                            "fraud_count": {"$sum": "$is_fraud"}
                        }},
                        {"$sort": {"_id": 1}}
                    ]
                
                results = list(collection.aggregate(pipeline))
                df = pd.DataFrame(results)
                
                # Decode merchant categories if present in aggregation results
                models = load_models()
                if len(df) > 0 and agg_type == "Fraud Count by Category":
                    category_mapping = models.get('category_mapping', {})
                    if category_mapping:
                        reverse_mapping = {v: k for k, v in category_mapping.items()}
                        # Check if '_id' contains category as integer
                        if '_id' in df.columns and isinstance(df.iloc[0]['_id'], dict):
                            df['_id_decoded'] = df['_id'].apply(lambda x: {
                                'category': reverse_mapping.get(x.get('category', x.get('category')), x.get('category')),
                                'fraud': x.get('fraud')
                            })
                            df = df.drop('_id', axis=1).rename(columns={'_id_decoded': '_id'})
                
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                if agg_type == "Hourly Transaction Count" and len(df) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Total', x=df['_id'], y=df['count'], marker_color='#3498db'))
                    fig.add_trace(go.Bar(name='Fraud', x=df['_id'], y=df['fraud_count'], marker_color='#e74c3c'))
                    fig.update_layout(barmode='group', title='Transactions by Hour')
                    st.plotly_chart(fig, use_container_width=True)


def render_local_query_interface(df, models):
    """Render query interface for local CSV data"""
    st.subheader("Filter Options")
    
    # Decode merchant categories first
    df = decode_merchant_category(df, models)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount filter
        amount_range = st.slider(
            "Amount Range",
            float(df['amount'].min()),
            float(df['amount'].max()),
            (float(df['amount'].min()), float(df['amount'].max()))
        )
        
        # Fraud filter
        fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud Only", "Non-Fraud Only"])
    
    with col2:
        # Category filter
        categories = ["All"] + list(df['merchant_category'].unique())
        category_filter = st.selectbox("Merchant Category", categories)
        
        # Trust score
        trust_range = st.slider("Device Trust Score", 0, 100, (0, 100))
    
    # Hour range
    hour_range = st.slider("Transaction Hour Range", 0, 23, (0, 23))
    
    # Foreign transaction
    foreign_filter = st.selectbox("Foreign Transaction", ["All", "Yes", "No"])
    
    if st.button("Apply Filters"):
        filtered_df = df.copy()
        
        # Apply filters
        filtered_df = filtered_df[
            (filtered_df['amount'] >= amount_range[0]) &
            (filtered_df['amount'] <= amount_range[1])
        ]
        
        filtered_df = filtered_df[
            (filtered_df['device_trust_score'] >= trust_range[0]) &
            (filtered_df['device_trust_score'] <= trust_range[1])
        ]
        
        filtered_df = filtered_df[
            (filtered_df['transaction_hour'] >= hour_range[0]) &
            (filtered_df['transaction_hour'] <= hour_range[1])
        ]
        
        if fraud_filter == "Fraud Only":
            filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
        elif fraud_filter == "Non-Fraud Only":
            filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
        
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['merchant_category'] == category_filter]
        
        if foreign_filter == "Yes":
            filtered_df = filtered_df[filtered_df['foreign_transaction'] == 1]
        elif foreign_filter == "No":
            filtered_df = filtered_df[filtered_df['foreign_transaction'] == 0]
        
        st.write(f"Found {len(filtered_df)} matching records")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Summary stats
        if len(filtered_df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Amount", f"${filtered_df['amount'].mean():.2f}")
            with col2:
                fraud_rate = filtered_df['is_fraud'].mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col3:
                st.metric("Total Records", len(filtered_df))


# ============================================
# Main App
# ============================================

def main():
    # Load models
    models = load_models()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to",
        [
            "🏠 Overview",
            "🔮 K-Means",
            "🔗 Apriori",
            "📈 Logistic Regression",
            "📊 Naïve Bayes",
            "🌳 Decision Tree",
            "📊 Model Comparison",
            "🗄️ MongoDB Query"
        ]
    )
    
    # Render selected page
    if selection == "🏠 Overview":
        render_overview_tab()
    elif selection == "🔮 K-Means":
        render_kmeans_tab(models)
    elif selection == "🔗 Apriori":
        render_apriori_tab(models)
    elif selection == "📈 Logistic Regression":
        render_logistic_regression_tab(models)
    elif selection == "📊 Naïve Bayes":
        render_naive_bayes_tab(models)
    elif selection == "🌳 Decision Tree":
        render_decision_tree_tab(models)
    elif selection == "📊 Model Comparison":
        render_comparison_tab(models)
    elif selection == "🗄️ MongoDB Query":
        render_mongodb_tab()


if __name__ == "__main__":
    main()
