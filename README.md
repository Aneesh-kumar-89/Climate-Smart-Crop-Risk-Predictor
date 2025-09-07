# 🌾 Climate-Smart Crop Risk Predictor

A machine learning-powered web application that helps farmers predict crop risks based on weather forecasts and make informed decisions to protect their harvests.

## 💡 Core Concept

**Input**: Weather forecast (temperature, rainfall, humidity, wind) + crop type + growth stage  
**Output**: Risk score (Low/Medium/High) of crop damage in the coming week

This solution provides zero-budget feasibility with huge real-world impact, helping farmers prevent crop losses from unpredictable climate changes.

### Example Flow:
👩‍🌾 **Farmer inputs**: Rice, 30°C avg, 80% humidity, 200mm rainfall forecast  
🤖 **Model predicts**: High Risk – Flooding likely. Suggests water drainage systems

### Why This Is Unique:
- Most plant AI apps only analyze leaf images (reactive)
- This system combines **climate + crop stage** to predict disaster risk **before it happens** (preventive)

## 🎯 Features

- **Real-time Weather Integration**: Live data from OpenWeatherMap API
- **Multiple Location Detection**: IP-based, GPS, and manual location input
- **Risk Prediction**: Analyze weather conditions and predict Low/Medium/High crop risks
- **Multi-Crop Support**: Rice, Wheat, Maize, Cotton, Sugarcane
- **Growth Stage Sensitivity**: Accounts for crop vulnerability at different stages
- **Actionable Recommendations**: Specific advice for each risk level
- **Interactive Dashboard**: Visual risk analysis and historical patterns
- **Geolocation Support**: Automatic location detection for seamless user experience

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Aneesh-kumar-89/Climate-Smart-Crop-Risk-Predictor.git
cd Climate-Smart-Crop-Risk-Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment (optional)**
```bash
cp .env.template .env
# Add your OpenWeatherMap API key to .env file
```

4. **Run the application**
```bash
# Enhanced version with real-time weather
streamlit run app_realtime.py

# Or basic version
streamlit run app.py
```

5. **Access the app**
Open http://localhost:8501 in your browser

## 📊 How It Works

1. **Location Detection**: Auto-detect via IP/GPS or manual entry
2. **Weather Data**: Fetch real-time conditions or manual input
3. **Crop Input**: Select crop type, growth stage, soil conditions
4. **ML Analysis**: Random Forest model analyzes climate stress factors
5. **Risk Prediction**: Returns risk level with confidence score
6. **Recommendations**: Provides specific actionable advice

## 🛠️ Project Structure

```
Climate-Smart-Crop-Risk-Predictor/
├── app.py                 # Basic Streamlit application
├── app_realtime.py        # Enhanced app with real-time weather
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env.template         # Environment variables template
├── .gitignore            # Git ignore patterns
├── Dockerfile            # Docker configuration
├── data/                 # Data storage
├── models/               # Trained ML models
├── src/                  # Source code modules
│   ├── __init__.py       # Package initialization
│   ├── data_processor.py # Weather data & processing
│   ├── model_trainer.py  # ML model training
│   └── risk_calculator.py # Risk calculation logic
├── config/               # Configuration files
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 🌐 Deployment Options

### Local Development
```bash
streamlit run app_realtime.py
```

### Streamlit Cloud (Free)
1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy directly from repository

### Docker Deployment
```bash
docker build -t crop-risk-predictor .
docker run -p 8501:8501 crop-risk-predictor
```

## 📈 Model Performance

- **Accuracy**: ~85% on synthetic data
- **Features**: Weather variables + crop characteristics
- **Algorithm**: Random Forest Classifier
- **Training Data**: 2000+ synthetic samples based on agricultural research
- **Real-time Integration**: Live weather data processing

## 🌱 Supported Crops

| Crop | Optimal Temp | Rainfall Range | Special Considerations |
|------|-------------|----------------|----------------------|
| Rice | 20-30°C | 100-200mm | Flood tolerance, humidity sensitivity |
| Wheat | 15-25°C | 40-100mm | Cold tolerance, drought sensitivity |
| Maize | 18-28°C | 50-150mm | Wind sensitivity, moderate water needs |
| Cotton | 21-32°C | 60-120mm | Heat tolerance, pest considerations |
| Sugarcane | 25-35°C | 150-300mm | High water needs, long growing season |

## 🗺️ Location Features

- **IP Geolocation**: Automatic location detection using IP address
- **GPS Integration**: Browser-based precise coordinate detection
- **Manual Input**: City name or coordinate entry
- **Global Support**: Works worldwide with weather API coverage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- 📱 Issues: [GitHub Issues](https://github.com/Aneesh-kumar-89/Climate-Smart-Crop-Risk-Predictor/issues)
- 📖 Documentation: Check `/docs` folder
- 🌐 Live Demo: Available via Streamlit deployment

## 🙏 Acknowledgments

- **Weather Data**: OpenWeatherMap API, ipapi.co
- **Agricultural Research**: FAO, local agricultural universities
- **ML Libraries**: scikit-learn, pandas, numpy
- **Web Framework**: Streamlit
- **Geolocation**: Browser Geolocation API, IP geolocation services

---

**Made with ❤️ for farmers worldwide** 🌾
