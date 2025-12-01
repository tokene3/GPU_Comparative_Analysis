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
        st.metric("Avg GPU Score", f"{filtered_df['gpu_score'].mean():.1f}/100")
        st.metric("ML Model R² Score", f"{r2:.3f}")
    
    st.markdown("---")
    
    # Top GPUs preview
    st.subheader("🏆 Top Performing GPUs")
    top_gpus = filtered_df.nlargest(5, 'gpu_score')[['manufacturer', 'productName', 'gpu_score', 'memSize', 'gpuClock', 'memBusWidth', 'unifiedShader']]
    st.dataframe(top_gpus, use_container_width=True)

# OVERVIEW PAGE
elif page == "📊 Overview":
    st.title("📊 Overview Dashboard")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total GPUs", len(filtered_df))
        
    with col2:
        st.metric("Manufacturers", filtered_df['manufacturer'].nunique())
        
    with col3:
        st.metric("Avg GPU Score", f"{filtered_df['gpu_score'].mean():.1f}/100")
        
    with col4:
        st.metric("Avg Memory (GB)", f"{filtered_df['memSize'].mean():.1f}")
    
    st.markdown("---")
    
    # First row of charts - USING PLOTLY FOR INTERACTIVITY
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 GPUs by Manufacturer")
        manufacturer_counts = filtered_df['manufacturer'].value_counts()
        
        fig = px.bar(
            x=manufacturer_counts.index,
            y=manufacturer_counts.values,
            labels={'x': 'Manufacturer', 'y': 'Count'},
            color=manufacturer_counts.index,
            color_discrete_sequence=['#76b7b2', '#edc948', '#af7aa1', '#ff9da7']
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            text=manufacturer_counts.values,
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ GPU Score Distribution")
        # Updated score bins for 1-100 scale
        score_bins = [1, 20, 40, 60, 80, 100]
        score_labels = ['1-20 (Low)', '20-40 (Fair)', '40-60 (Good)', '60-80 (High)', '80-100 (Excellent)']
        filtered_df['score_category'] = pd.cut(filtered_df['gpu_score'], bins=score_bins, labels=score_labels)
        score_counts = filtered_df['score_category'].value_counts().sort_index()
        
        fig = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            labels={'x': 'Score Category', 'y': 'Count'},
            color=score_counts.values,
            color_continuous_scale='reds'
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            text=score_counts.values,
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Memory Type Distribution")
        mem_type_counts = filtered_df['memType'].value_counts().head(6)
        
        fig = px.pie(
            values=mem_type_counts.values,
            names=mem_type_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textinfo='percent+label'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 GPU Score vs Multiple Features")
        color_by = st.selectbox("Color points by:", ['gpuClock', 'memBusWidth', 'unifiedShader', 'releaseYear'])
        
        fig = px.scatter(
            filtered_df,
            x='memSize',
            y='gpu_score',
            color=color_by,
            hover_data=['manufacturer', 'productName'],
            labels={
                'memSize': 'Memory Size (GB)',
                'gpu_score': 'GPU Score (1-100)',
                color_by: color_by
            },
            title=f'GPU Score vs Memory Size (colored by {color_by})'
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>Memory: %{x} GB<br>Score: %{y}/100<br>" + 
                         f"{color_by}: " + "%{marker.color}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

# FILTERS & ANALYSIS PAGE
elif page == "🔧 Filters & Analysis":
    st.title("🔧 Detailed Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ GPU Clock vs Memory Bus Width")
        fig = px.scatter(
            filtered_df,
            x='memBusWidth',
            y='gpuClock',
            color='gpu_score',
            hover_data=['manufacturer', 'productName'],
            labels={
                'memBusWidth': 'Memory Bus Width (bits)',
                'gpuClock': 'GPU Clock (MHz)',
                'gpu_score': 'GPU Score'
            },
            color_continuous_scale='viridis'
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>Bus Width: %{x} bits<br>Clock: %{y} MHz<br>Score: %{marker.color}/100<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Manufacturer Comparison")
        compare_metric = st.selectbox("Select metric to compare:", 
                                    ['memSize', 'gpuClock', 'memClock', 'memBusWidth', 'unifiedShader', 'gpu_score'])
        
        fig = px.box(
            filtered_df,
            x='manufacturer',
            y=compare_metric,
            color='manufacturer',
            color_discrete_sequence=['#76b7b2', '#edc948', '#af7aa1', '#ff9da7']
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" + f"{compare_metric}: " + "%{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("📈 Performance Metrics by Manufacturer")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        avg_mem_by_mfr = filtered_df.groupby('manufacturer')['memSize'].mean().round(1)
        for mfr, avg in avg_mem_by_mfr.items():
            st.metric(f"{mfr} Avg Memory", f"{avg} GB")
    
    with metrics_col2:
        avg_clock_by_mfr = filtered_df.groupby('manufacturer')['gpuClock'].mean().round(0)
        for mfr, avg in avg_clock_by_mfr.items():
            st.metric(f"{mfr} Avg Clock", f"{int(avg)} MHz")
    
    with metrics_col3:
        avg_score_by_mfr = filtered_df.groupby('manufacturer')['gpu_score'].mean().round(1)
        for mfr, avg in avg_score_by_mfr.items():
            st.metric(f"{mfr} Avg Score", f"{avg}/100")
    
    with metrics_col4:
        avg_shader_by_mfr = filtered_df.groupby('manufacturer')['unifiedShader'].mean().round(0)
        for mfr, avg in avg_shader_by_mfr.items():
            st.metric(f"{mfr} Avg Shaders", f"{int(avg)}")

# TRENDS PAGE
elif page == "📈 Trends":
    st.title("📈 Historical Trends")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 GPUs by Release Year")
        year_counts = filtered_df['releaseYear'].value_counts().sort_index()
        
        fig = px.line(
            x=year_counts.index,
            y=year_counts.values,
            markers=True,
            labels={'x': 'Release Year', 'y': 'Number of GPUs Released'}
        )
        fig.update_traces(
            line=dict(width=4, color='#d62728'),
            marker=dict(size=8, symbol='circle', line=dict(width=2, color='white'))
        )
        fig.update_layout(
            hovermode='x unified',
            showlegend=False
        )
        fig.update_traces(
            hovertemplate="<b>Year: %{x}</b><br>GPUs Released: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚀 Performance Evolution")
        trend_metric = st.selectbox("Select performance metric:", 
                                  ['memSize', 'gpuClock', 'memClock', 'memBusWidth', 'unifiedShader', 'gpu_score'])
        
        yearly_avg = filtered_df.groupby(['releaseYear', 'manufacturer'])[trend_metric].mean().reset_index()
        
        fig = px.line(
            yearly_avg,
            x='releaseYear',
            y=trend_metric,
            color='manufacturer',
            markers=True,
            labels={'releaseYear': 'Release Year', trend_metric: f'Average {trend_metric}'}
        )
        fig.update_traces(
            hovertemplate="<b>Year: %{x}</b><br>" + f"Avg {trend_metric}: " + "%{y}<br>Manufacturer: %{fullData.name}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Technology adoption timeline
    st.subheader("🔄 Memory Technology Adoption")
    
    if 'memType' in filtered_df.columns:
        # Pivot table for memory type by year
        memtype_year = pd.crosstab(filtered_df['releaseYear'], filtered_df['memType'])
        
        fig = px.bar(
            memtype_year,
            x=memtype_year.index,
            y=memtype_year.columns,
            labels={'x': 'Release Year', 'y': 'Number of GPUs', 'variable': 'Memory Type'}
        )
        fig.update_traces(
            hovertemplate="<b>Year: %{x}</b><br>Memory Type: %{fullData.name}<br>Count: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

# ML PREDICTOR PAGE
elif page == "🤖 ML Predictor":
    st.title("🤖 GPU Performance Score Predictor")
    st.markdown("---")
    
    # Comprehensive Model Metrics Section
    st.subheader("📊 Comprehensive Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
        
    with col2:
        st.metric("RMSE", f"{rmse:.3f}")
        
    with col3:
        st.metric("MAE", f"{mae:.3f}")
        
    with col4:
        st.metric("MSE", f"{mse:.3f}")
    
    # Feature importance
    st.subheader("🔍 Feature Importance for Score Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            y='feature',
            x='importance',
            orientation='h',
            labels={'feature': 'Feature', 'importance': 'Importance'},
            color='importance',
            color_continuous_scale='blues'
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Actual vs Predicted plot
        st.subheader("📈 Actual vs Predicted Scores")
        
        # Create a DataFrame for plotting
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig = px.scatter(
            results_df,
            x='Actual',
            y='Predicted',
            labels={'Actual': 'Actual Scores', 'Predicted': 'Predicted Scores'},
            opacity=0.6
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='black'),
                name='Perfect Prediction'
            )
        )
        fig.update_traces(
            hovertemplate="<b>Actual: %{x:.2f}</b><br>Predicted: %{y:.2f}<extra></extra>",
            selector=dict(type='scatter', mode='markers')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interactive Prediction Section
    st.subheader("🔮 Predict GPU Performance Score")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mem_size = st.slider("Memory Size (GB)", min_value=1, max_value=24, value=8, step=1)
        mem_bus = st.slider("Memory Bus Width (bits)", min_value=32, max_value=512, value=128, step=32)
    
    with col2:
        gpu_clock = st.slider("GPU Clock (MHz)", min_value=100, max_value=3000, value=1500, step=50)
        mem_clock = st.slider("Memory Clock (MHz)", min_value=500, max_value=2500, value=1500, step=50)
    
    with col3:
        unified_shader = st.slider("Unified Shaders", min_value=100, max_value=15000, value=2000, step=100)
        tmu_count = st.slider("TMU Count", min_value=10, max_value=500, value=100, step=10)
    
    with col4:
        rop_count = st.slider("ROP Count", min_value=4, max_value=200, value=32, step=4)
        release_year = st.slider("Release Year", min_value=2000, max_value=2024, value=2022, step=1)
    
    # Create input array for prediction - FIXED VERSION
    input_features = {}
    
    # Create a mapping from feature names to variable names
    feature_mapping = {
        'memSize': mem_size,
        'memBusWidth': mem_bus,
        'gpuClock': gpu_clock,
        'memClock': mem_clock,
        'unifiedShader': unified_shader,
        'tmu': tmu_count,
        'rop': rop_count,
        'releaseYear': release_year
    }
    
    # Handle both original and engineered features
    for feature in available_features:
        if feature in feature_mapping:
            input_features[feature] = feature_mapping[feature]
        elif feature == 'mem_bandwidth_score':
            input_features[feature] = mem_size * mem_bus / 64
        elif feature == 'compute_power':
            input_features[feature] = gpu_clock * unified_shader / 1000
        elif feature == 'texture_pixel_power':
            input_features[feature] = (tmu_count + rop_count) / 10
        else:
            # For any other features, use default values
            input_features[feature] = 0
    
    # Filter only available features and ensure correct order
    input_array = np.array([[input_features[feature] for feature in available_features]])
    
    # Make prediction
    if st.button("🎯 Predict GPU Score", type="primary"):
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Get prediction
        predicted_score = model.predict(input_scaled)[0]
        predicted_score = max(1, min(100, predicted_score))  # Clamp between 1-100
        
        # Display results with visual rating
        st.success(f"**Predicted GPU Performance Score: {predicted_score:.1f}/100**")
        
        # Visual score indicator
        st.subheader("📊 Performance Rating")
        
        # Create a visual progress bar for the score
        score_percentage = (predicted_score / 100) * 100
        
        # Determine color based on score
        if predicted_score >= 80:
            color = "#00ff00"  # Green
            rating = "Excellent"
        elif predicted_score >= 60:
            color = "#90ee90"  # Light Green
            rating = "Very Good"
        elif predicted_score >= 40:
            color = "#ffff00"  # Yellow
            rating = "Good"
        elif predicted_score >= 20:
            color = "#ffa500"  # Orange
            rating = "Average"
        else:
            color = "#ff0000"  # Red
            rating = "Basic"
        
        # Display visual rating
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {color} {score_percentage}%, #f0f0f0 {score_percentage}%); 
                        height: 30px; border-radius: 15px; position: relative;">
                <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); 
                           font-weight: bold; color: {'white' if predicted_score >= 60 else 'black'};">
                    {predicted_score:.1f}/100 - {rating}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Rating", rating)
        
        with col3:
            st.metric("Percentile", f"Top {100 - predicted_score:.0f}%")
        
        # Show similar GPUs from dataset
        st.subheader("🔍 Similar GPUs in Dataset")
        similar_gpus = df.copy()
        
        # Calculate similarity score using multiple features
        similarity_features = ['memSize', 'gpuClock', 'memBusWidth', 'unifiedShader', 'memClock']
        for feature in similarity_features:
            if feature in available_features:
                similar_gpus[f'{feature}_diff'] = abs(similar_gpus[feature] - input_features.get(feature, 0))
        
        available_similarity_features = [f'{f}_diff' for f in similarity_features if f in available_features]
        if available_similarity_features:
            similar_gpus['similarity_score'] = similar_gpus[available_similarity_features].sum(axis=1)
            similar_gpus = similar_gpus.nsmallest(5, 'similarity_score')
            
            display_cols = ['manufacturer', 'productName', 'gpu_score', 'memSize', 'gpuClock', 'memBusWidth']
            display_cols = [col for col in display_cols if col in similar_gpus.columns]
            st.dataframe(similar_gpus[display_cols], use_container_width=True)
        else:
            st.info("No similar GPUs found with current feature set.")

# DATA EXPLORER PAGE
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Filtered Dataset")
        st.dataframe(filtered_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Quick Statistics")
        st.write(f"**Dataset Shape:** {filtered_df.shape}")
        st.write(f"**Columns:** {len(filtered_df.columns)}")
        st.write(f"**Manufacturers:** {', '.join(filtered_df['manufacturer'].unique())}")
        
        st.download_button(
            label="📥 Download Filtered Data",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_gpu_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Detailed statistics
    st.subheader("📈 Detailed Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.write("**Numerical Columns Summary**")
        numerical_summary = filtered_df[numerical_cols + ['gpu_score']].describe()
        st.dataframe(numerical_summary, use_container_width=True)
    
    with stats_col2:
        st.write("**Categorical Columns Summary**")
        categorical_cols = ['manufacturer', 'memType', 'igp', 'bus']
        for col in categorical_cols:
            if col in filtered_df.columns:
                st.write(f"**{col}:**")
                st.write(filtered_df[col].value_counts())
