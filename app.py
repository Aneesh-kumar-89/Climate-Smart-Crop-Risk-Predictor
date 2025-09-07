import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os
from src.data_processor import WeatherDataProcessor

# Configure page
st.set_page_config(
    page_title="Climate-Smart Crop Risk Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data():
    """Generate synthetic training data for demonstration"""
    np.random.seed(42)
    n_samples = 2000
    
    data = []
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    growth_stages = ['Seedling', 'Vegetative', 'Flowering', 'Maturation']
    
    for _ in range(n_samples):
        # Generate weather features
        temp = np.random.normal(25, 8)  # Temperature (°C)
        rainfall = np.random.exponential(50)  # Rainfall (mm/week)
        humidity = np.random.normal(70, 15)  # Humidity (%)
        wind_speed = np.random.exponential(10)  # Wind speed (km/h)
        
        # Generate other features
        crop = np.random.choice(crops)
        growth_stage = np.random.choice(growth_stages)
        soil_moisture = np.random.normal(50, 20)
        
        # Calculate risk based on thresholds
        risk_score = 0
        
        # Temperature stress
        if crop == 'Rice' and (temp > 35 or temp < 15):
            risk_score += 2
        elif crop == 'Wheat' and (temp > 30 or temp < 5):
            risk_score += 2
        elif crop == 'Maize' and (temp > 32 or temp < 10):
            risk_score += 2
        elif temp > 40 or temp < 0:
            risk_score += 3
            
        # Rainfall stress
        if rainfall > 200:  # Too much rain
            risk_score += 2
        elif rainfall < 10:  # Drought
            risk_score += 2
            
        # Humidity stress
        if humidity > 90 and growth_stage == 'Flowering':
            risk_score += 2  # Fungal disease risk
        elif humidity < 30:
            risk_score += 1
            
        # Wind stress
        if wind_speed > 25:
            risk_score += 1
            
        # Growth stage sensitivity
        if growth_stage in ['Seedling', 'Flowering']:
            risk_score *= 1.2
            
        # Convert to categorical risk
        if risk_score >= 4:
            risk_level = 'High'
        elif risk_score >= 2:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
            
        data.append({
            'temperature': temp,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'crop': crop,
            'growth_stage': growth_stage,
            'soil_moisture': soil_moisture,
            'risk_level': risk_level,
            'risk_score': risk_score
        })
    
    return pd.DataFrame(data)

@st.cache_resource
def train_model():
    """Train the crop risk prediction model"""
    df = generate_synthetic_data()
    
    # Prepare features
    df_encoded = pd.get_dummies(df[['temperature', 'rainfall', 'humidity', 'wind_speed', 
                                   'crop', 'growth_stage', 'soil_moisture']], 
                               columns=['crop', 'growth_stage'])
    
    X = df_encoded
    y = df['risk_level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    return model, scaler, X.columns.tolist(), train_score, test_score, df

def get_risk_recommendations(risk_level, crop, weather_conditions):
    """Generate actionable recommendations based on risk level"""
    recommendations = {
        'High': {
            'Rice': [
                "🚨 Immediate action required",
                "Consider drainage systems if heavy rainfall predicted",
                "Apply fungicide preventively if humidity >90%",
                "Harvest early if crop is near maturation",
                "Secure crop insurance claims documentation"
            ],
            'Wheat': [
                "🚨 High risk detected",
                "Protect from frost if temperature <5°C",
                "Avoid irrigation if heavy rain predicted",
                "Apply windbreaks if strong winds expected",
                "Monitor for rust diseases in humid conditions"
            ],
            'Maize': [
                "🚨 Critical weather conditions",
                "Ensure adequate drainage",
                "Protect from hail with netting if possible",
                "Delay harvesting if storms predicted",
                "Check soil compaction after heavy rain"
            ]
        },
        'Medium': {
            'Rice': [
                "⚠️ Monitor closely",
                "Check water levels daily",
                "Prepare drainage equipment",
                "Monitor pest activity",
                "Have backup irrigation ready"
            ],
            'Wheat': [
                "⚠️ Moderate caution needed",
                "Monitor soil moisture",
                "Watch for disease symptoms",
                "Prepare for temperature fluctuations",
                "Check grain quality regularly"
            ],
            'Maize': [
                "⚠️ Stay vigilant",
                "Monitor silking stage carefully",
                "Check for pest damage",
                "Ensure adequate pollination",
                "Prepare for weather changes"
            ]
        },
        'Low': {
            'Rice': [
                "✅ Favorable conditions",
                "Continue regular care routine",
                "Optimal time for fertilizer application",
                "Good conditions for growth",
                "Monitor for optimal harvest timing"
            ],
            'Wheat': [
                "✅ Good growing conditions",
                "Ideal for normal farming operations",
                "Good time for field visits",
                "Continue scheduled activities",
                "Monitor for early pest signs"
            ],
            'Maize': [
                "✅ Excellent conditions",
                "Perfect for field operations",
                "Good time for side-dressing",
                "Optimal growing weather",
                "Plan for harvest preparation"
            ]
        }
    }
    
    default_rec = [
        "Continue monitoring weather conditions",
        "Follow standard crop management practices",
        "Consult local agricultural extension services"
    ]
    
    return recommendations.get(risk_level, {}).get(crop, default_rec)

def predict_risk(model, scaler, feature_columns, temperature, rainfall, humidity, wind_speed, crop, growth_stage, soil_moisture):
    """Make risk prediction for given inputs"""
    # Create input dataframe
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'rainfall': [rainfall],
        'humidity': [humidity],
        'wind_speed': [wind_speed],
        'crop': [crop],
        'growth_stage': [growth_stage],
        'soil_moisture': [soil_moisture]
    })
    
    # Encode categorical variables
    input_encoded = pd.get_dummies(input_data, columns=['crop', 'growth_stage'])
    
    # Align with training features
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_encoded)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get class probabilities
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))
    
    return prediction, prob_dict

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Climate-Smart Crop Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Predict crop risks before they happen • Protect your harvest • Make informed decisions**")
    
    # Initialize weather processor
    weather_processor = WeatherDataProcessor()
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model, scaler, feature_columns, train_score, test_score, training_data = train_model()
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Training Accuracy", f"{train_score:.1%}")
        st.metric("Testing Accuracy", f"{test_score:.1%}")
        st.metric("Training Samples", len(training_data))
        
        st.header("🌍 Supported Regions")
        st.write("• India (Primary)")
        st.write("• South Asia")
        st.write("• Similar Climate Zones")
        
        st.header("📖 How It Works")
        st.write("1. Input current weather & crop details")
        st.write("2. AI analyzes climate stress factors")
        st.write("3. Get risk level & recommendations")
        st.write("4. Take preventive action")
    
    # Real-time weather section
    st.header("🌍 Real-Time Weather Data")
    
    # Location input for real-time weather
    location_col1, location_col2, location_col3 = st.columns([2, 1, 1])
    
    with location_col1:
        city_name = st.text_input(
            "Enter City Name", 
            value="Mumbai", 
            help="Enter your city name to fetch real-time weather data"
        )
    
    with location_col2:
        country_code = st.selectbox(
            "Country", 
            ["IN", "US", "GB", "AU", "CA", "DE", "FR", "JP", "BR", "ZA"],
            help="Select your country code"
        )
    
    with location_col3:
        fetch_weather = st.button("🌤️ Get Live Weather", type="secondary")
    
    # Initialize weather variables
    real_time_weather = None
    use_real_time = False
    
    if fetch_weather and city_name:
        with st.spinner(f"Fetching real-time weather for {city_name}..."):
            real_time_weather = weather_processor.get_real_time_weather_for_location(city_name, country_code)
            if real_time_weather and real_time_weather.get('location') != 'Demo Location':
                use_real_time = True
                st.success(f"✅ Weather data fetched for {real_time_weather['location']}, {real_time_weather['country']}")
                
                # Display current weather
                weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
                
                with weather_col1:
                    st.metric("🌡️ Temperature", f"{real_time_weather['temperature']:.1f}°C", 
                             f"Feels like {real_time_weather['feels_like']:.1f}°C")
                
                with weather_col2:
                    st.metric("💧 Humidity", f"{real_time_weather['humidity']:.0f}%")
                
                with weather_col3:
                    st.metric("💨 Wind Speed", f"{real_time_weather['wind_speed']:.1f} km/h")
                
                with weather_col4:
                    st.metric("🌧️ Rainfall", f"{real_time_weather['rainfall']:.1f} mm/day")
                
                st.info(f"Current conditions: {real_time_weather['description'].title()}")
            else:
                st.error("❌ Could not fetch weather data. Please check city name or try again later.")
    
    # Toggle between real-time and manual input
    input_mode = st.radio(
        "Choose Input Mode:",
        ["📍 Use Real-Time Weather Data" if use_real_time else "📍 Real-Time Data (Fetch weather first)", "✋ Manual Input"],
        index=0 if use_real_time else 1,
        help="Select whether to use fetched real-time data or input manually"
    )
    
    # Main input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_mode.startswith("📍") and use_real_time:
            st.header("🌦️ Real-Time Weather Conditions")
            st.info("Using live weather data from OpenWeatherMap API")
            
            # Use real-time weather data
            temperature = real_time_weather['temperature']
            rainfall = real_time_weather['rainfall']
            humidity = real_time_weather['humidity']
            wind_speed = real_time_weather['wind_speed']
            
            # Display the values (read-only)
            temp_col, rain_col = st.columns(2)
            with temp_col:
                st.metric("Temperature (°C)", f"{temperature:.1f}")
            with rain_col:
                st.metric("Rainfall (mm/day)", f"{rainfall:.1f}")
            
            humid_col, wind_col = st.columns(2)
            with humid_col:
                st.metric("Humidity (%)", f"{humidity:.0f}")
            with wind_col:
                st.metric("Wind Speed (km/h)", f"{wind_speed:.1f}")
                
        else:
            st.header("🌦️ Enter Current Conditions")
            
            # Weather inputs for manual mode
            st.subheader("Weather Forecast (Next 7 Days)")
            temp_col, rain_col = st.columns(2)
            
            with temp_col:
                temperature = st.slider(
                    "Average Temperature (°C)", 
                    min_value=-10, max_value=50, value=25, step=1,
                    help="Expected average temperature for the next week"
                )
            
            with rain_col:
                rainfall = st.slider(
                    "Total Rainfall (mm)", 
                    min_value=0, max_value=500, value=50, step=5,
                    help="Expected total rainfall for the next week"
                )
            
            humid_col, wind_col = st.columns(2)
            
            with humid_col:
                humidity = st.slider(
                    "Average Humidity (%)", 
                    min_value=10, max_value=100, value=70, step=5,
                    help="Expected average humidity level"
                )
            
            with wind_col:
                wind_speed = st.slider(
                    "Wind Speed (km/h)", 
                    min_value=0, max_value=50, value=10, step=1,
                    help="Expected average wind speed"
                )
        
        # Crop details
        st.subheader("Crop Information")
        crop_col, stage_col = st.columns(2)
        
        with crop_col:
            crop = st.selectbox(
                "Crop Type", 
                ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'],
                help="Select your primary crop"
            )
        
        with stage_col:
            growth_stage = st.selectbox(
                "Growth Stage", 
                ['Seedling', 'Vegetative', 'Flowering', 'Maturation'],
                help="Current stage of crop development"
            )
        
        soil_moisture = st.slider(
            "Soil Moisture (%)", 
            min_value=0, max_value=100, value=50, step=5,
            help="Current soil moisture level"
        )
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Weather stress indicators
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Temperature Status", 
                 "Normal" if 15 <= temperature <= 35 else "Stress",
                 delta=f"{temperature - 25:.1f}°C from optimal")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rainfall Status", 
                 "Normal" if 25 <= rainfall <= 150 else "Abnormal",
                 delta=f"{rainfall - 75:.0f}mm from ideal")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Humidity Status", 
                 "Normal" if 40 <= humidity <= 80 else "Stress",
                 delta=f"{humidity - 60:.0f}% from optimal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.header("🎯 Risk Assessment")
    
    if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing climate conditions and crop vulnerability..."):
            # Make prediction
            risk_level, probabilities = predict_risk(
                model, scaler, feature_columns,
                temperature, rainfall, humidity, wind_speed, 
                crop, growth_stage, soil_moisture
            )
            
            # Display risk level
            if risk_level == 'High':
                st.markdown(f'''
                <div class="risk-high">
                    <h2>🚨 HIGH RISK</h2>
                    <p><strong>Immediate attention required!</strong> Weather conditions pose significant threat to {crop} crop.</p>
                    <p><strong>Confidence:</strong> {probabilities[risk_level]:.1%}</p>
                </div>
                ''', unsafe_allow_html=True)
            elif risk_level == 'Medium':
                st.markdown(f'''
                <div class="risk-medium">
                    <h2>⚠️ MEDIUM RISK</h2>
                    <p><strong>Monitor closely.</strong> Some weather stress factors detected for {crop} crop.</p>
                    <p><strong>Confidence:</strong> {probabilities[risk_level]:.1%}</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="risk-low">
                    <h2>✅ LOW RISK</h2>
                    <p><strong>Favorable conditions!</strong> Weather is suitable for {crop} growth.</p>
                    <p><strong>Confidence:</strong> {probabilities[risk_level]:.1%}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Display probability breakdown
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Detailed Risk Breakdown")
                
                # Create probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(probabilities.keys()),
                        y=list(probabilities.values()),
                        marker_color=['#f44336' if x=='High' else '#ff9800' if x=='Medium' else '#4caf50' 
                                     for x in probabilities.keys()]
                    )
                ])
                fig.update_layout(
                    title="Risk Probability Distribution",
                    xaxis_title="Risk Level",
                    yaxis_title="Probability",
                    height=400,
                    showlegend=False
                )
                fig.update_yaxis(tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💡 Recommendations")
                recommendations = get_risk_recommendations(risk_level, crop, {
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                })
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
    
    # Historical analysis section
    st.header("📊 Historical Climate Patterns")
    
    # Show training data insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_dist = training_data['risk_level'].value_counts()
        fig = px.pie(values=risk_dist.values, names=risk_dist.index, 
                     title="Historical Risk Distribution",
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        crop_risk = training_data.groupby(['crop', 'risk_level']).size().reset_index(name='count')
        fig = px.bar(crop_risk, x='crop', y='count', color='risk_level',
                     title="Risk Levels by Crop Type",
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        stage_risk = training_data.groupby(['growth_stage', 'risk_level']).size().reset_index(name='count')
        fig = px.bar(stage_risk, x='growth_stage', y='count', color='risk_level',
                     title="Risk by Growth Stage",
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather correlation analysis
    st.subheader("🌡️ Weather Impact Analysis")
    
    # Temperature vs Risk
    temp_risk = training_data.groupby(['risk_level'])['temperature'].mean().reset_index()
    rain_risk = training_data.groupby(['risk_level'])['rainfall'].mean().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(temp_risk, x='risk_level', y='temperature', 
                     title="Average Temperature by Risk Level",
                     color='risk_level',
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(rain_risk, x='risk_level', y='rainfall', 
                     title="Average Rainfall by Risk Level",
                     color='risk_level',
                     color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🌱 Climate-Smart Crop Risk Predictor** | Built for farmers, by data science
    
    **Disclaimer**: This tool provides risk estimates based on weather patterns and should be used alongside local agricultural expertise. 
    Always consult with agricultural extension services for critical decisions.
    
    **Data Sources**: Synthetic data for demonstration. In production, integrate with NOAA, local meteorological services.
    """)

if __name__ == "__main__":
    main()