import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPU Specifications Dashboard",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('gpu_specs_v6.csv')
    # Drop pixelShader and vertexShader columns as requested
    df = df.drop(['pixelShader', 'vertexShader'], axis=1, errors='ignore')
    
    # Filter to only include NVIDIA, ATI, AMD, Intel
    valid_manufacturers = ['NVIDIA', 'ATI', 'AMD', 'Intel']
    df = df[df['manufacturer'].isin(valid_manufacturers)]
    
    return df

df = load_data()

# Impute missing values with average whole numbers for numerical columns
numerical_cols = ['releaseYear', 'memSize', 'memBusWidth', 'gpuClock', 'memClock', 
                 'unifiedShader', 'tmu', 'rop']

for col in numerical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        avg_value = int(df[col].mean())
        df[col].fillna(avg_value, inplace=True)

# Create GPU Performance Score (1-100) based on specifications - UPDATED VERSION
def calculate_gpu_score(row):
    """Calculate GPU performance score from 1-100 with balanced feature contributions"""
    score = 0
    
    # Memory Size contribution (0-20 points) - Scaled for 1-100 range
    if row['memSize'] <= 2:
        score += 5
    elif row['memSize'] <= 4:
        score += 10
    elif row['memSize'] <= 8:
        score += 15
    else:
        score += 20
    
    # GPU Clock contribution (0-20 points) - Enhanced and scaled
    if row['gpuClock'] >= 2500:
        score += 20
    elif row['gpuClock'] >= 2000:
        score += 15
    elif row['gpuClock'] >= 1500:
        score += 10
    elif row['gpuClock'] >= 1000:
        score += 5
    else:
        score += 2.5
    
    # Memory Bus Width contribution (0-20 points) - Enhanced and scaled
    if row['memBusWidth'] >= 384:
        score += 20
    elif row['memBusWidth'] >= 256:
        score += 15
    elif row['memBusWidth'] >= 192:
        score += 10
    elif row['memBusWidth'] >= 128:
        score += 7.5
    elif row['memBusWidth'] >= 64:
        score += 5
    else:
        score += 2.5
    
    # Memory Clock contribution (0-10 points) - Scaled
    if 'memClock' in row:
        if row['memClock'] >= 2000:
            score += 10
        elif row['memClock'] >= 1500:
            score += 7.5
        elif row['memClock'] >= 1000:
            score += 5
        else:
            score += 2.5
    
    # Unified Shaders contribution (0-20 points) - Enhanced and scaled
    if 'unifiedShader' in row:
        if row['unifiedShader'] >= 8000:
            score += 20
        elif row['unifiedShader'] >= 5000:
            score += 15
        elif row['unifiedShader'] >= 3000:
            score += 10
        elif row['unifiedShader'] >= 1500:
            score += 7.5
        elif row['unifiedShader'] >= 800:
            score += 5
        else:
            score += 2.5
    
    # TMU contribution (0-10 points) - Scaled
    if 'tmu' in row:
        if row['tmu'] >= 200:
            score += 10
        elif row['tmu'] >= 100:
            score += 7.5
        elif row['tmu'] >= 50:
            score += 5
        else:
            score += 2.5
    
    # ROP contribution (0-10 points) - Scaled
    if 'rop' in row:
        if row['rop'] >= 80:
            score += 10
        elif row['rop'] >= 40:
            score += 7.5
        elif row['rop'] >= 20:
            score += 5
        else:
            score += 2.5
    
    # Release Year bonus (0-10 points) - Scaled
    if 'releaseYear' in row:
        current_year = 2024
        year_diff = current_year - row['releaseYear']
        if year_diff <= 2:
            score += 10
        elif year_diff <= 5:
            score += 5
        elif year_diff <= 8:
            score += 2.5
    
    # Normalize to 1-100 scale
    max_possible_score = 120  # 20 + 20 + 20 + 10 + 20 + 10 + 10 + 10 = 120
    normalized_score = min(100, max(1, (score / max_possible_score) * 100))
    
    return round(normalized_score, 1)

# Apply GPU score calculation
df['gpu_score'] = df.apply(calculate_gpu_score, axis=1)

# ML Model Training for GPU Score Prediction - FIXED VERSION
@st.cache_resource
def train_ml_model(df):
    # Prepare features for ML
    feature_cols = ['memSize', 'memBusWidth', 'gpuClock', 'memClock', 'unifiedShader', 'tmu', 'rop', 'releaseYear']
    
    # Filter only columns that exist in the dataset
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['gpu_score']
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    # Add feature engineering to create interactions and reduce memSize dominance
    if 'memSize' in X.columns and 'memBusWidth' in X.columns:
        X['mem_bandwidth_score'] = X['memSize'] * X['memBusWidth'] / 64
    
    if 'gpuClock' in X.columns and 'unifiedShader' in X.columns:
        X['compute_power'] = X['gpuClock'] * X['unifiedShader'] / 1000
    
    if 'tmu' in X.columns and 'rop' in X.columns:
        X['texture_pixel_power'] = (X['tmu'] + X['rop']) / 10
    
    # Update available features with engineered ones
    available_features = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with better hyperparameters to balance feature importance
    model = RandomForestRegressor(
        n_estimators=200, 
        random_state=42, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Don't always use all features
        bootstrap=True
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate comprehensive metrics for regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, available_features, mse, rmse, mae, r2, X_test_scaled, y_test, y_pred

# Train the model
model, scaler, available_features, mse, rmse, mae, r2, X_test, y_test, y_pred = train_ml_model(df)

# Sidebar for navigation
st.sidebar.title("🖥️ GPU Dashboard")
page = st.sidebar.radio("Navigate to:", ["🏠 Home", "📊 Overview", "🔧 Filters & Analysis", "📈 Trends", "🤖 ML Predictor", "📋 Data Explorer"])

# Common filters for all pages
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Global Filters")

# Manufacturer filter
manufacturers = st.sidebar.multiselect(
    "Manufacturers",
    options=df['manufacturer'].unique(),
    default=df['manufacturer'].unique()
)

# Release year range
min_year = int(df['releaseYear'].min())
max_year = int(df['releaseYear'].max())
year_range = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# GPU Score range - UPDATED TO 1-100
score_range = st.sidebar.slider(
    "GPU Score Range",
    min_value=1.0,
    max_value=100.0,
    value=(1.0, 100.0),
    step=1.0
)

# Apply filters
filtered_df = df[
    (df['manufacturer'].isin(manufacturers)) &
    (df['releaseYear'].between(year_range[0], year_range[1])) &
    (df['gpu_score'].between(score_range[0], score_range[1]))
]

# HOME PAGE
if page == "🏠 Home":
    # Display image only on home page - YOUR ORIGINAL IMAGE
    st.image("MyGPU.JPG")
    
    st.title("🖥️ GPU Specifications Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Welcome to the GPU Analysis Dashboard")
        st.markdown("""
        This interactive dashboard provides comprehensive analysis of GPU specifications across major manufacturers:
        - **NVIDIA**
        - **AMD** 
        - **Intel**

        
        ### 📊 What you can explore:
        - **Overview**: Key metrics and manufacturer distributions
        - **Filters & Analysis**: Detailed visualizations and comparisons
        - **Trends**: Historical performance and technology evolution
        - **🤖 ML Predictor**: Predict GPU Performance Score (1-100) based on specs
        - **Data Explorer**: Raw data and detailed statistics
        
        ### 🎯 Key Features:
        - Real-time filtering and updates
        - Interactive charts and visualizations
        - Machine Learning performance predictions
        - Comparative analysis between manufacturers
        - Performance trend analysis
        - **Balanced Feature Importance**: All specifications contribute meaningfully to performance scores
        """)
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total GPUs", len(filtered_df))
        st.metric("Manufacturers", filtered_df['manufacturer'].nunique())
        st.metric("Years Covered", f"{min_year} - {max_year}")
        st.metric("Avg GPU Score", f"{filtered_df['gpu_score'].mean():.1f}/
