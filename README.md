# 🌾 Climate-Smart Crop Risk Predictor

A machine learning-powered web application that helps farmers predict crop risks based on weather forecasts and make informed decisions to protect their harvests.

## 🎯 Features

- **Risk Prediction**: Analyze weather conditions and predict Low/Medium/High crop risks
- **Multi-Crop Support**: Rice, Wheat, Maize, Cotton, Sugarcane
- **Growth Stage Sensitivity**: Accounts for crop vulnerability at different stages
- **Actionable Recommendations**: Specific advice for each risk level
- **Interactive Dashboard**: Visual risk analysis and historical patterns
- **Real-time Weather**: Optional integration with weather APIs

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/climate-smart-crop-predictor.git
cd climate-smart-crop-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
Open http://localhost:8501 in your browser

## 📊 How It Works

1. **Input**: Weather forecast, crop type, growth stage
2. **Analysis**: ML model analyzes climate stress factors
3. **Prediction**: Returns risk level with confidence score
4. **Action**: Provides specific recommendations

## 🛠️ Development Setup

### Project Structure
```
climate-smart-crop-predictor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env.template         # Environment variables template
├── data/                 # Data storage
├── models/               # Trained ML models
├── src/                  # Source code modules
│   ├── data_processor.py # Data handling
│   ├── model_trainer.py  # ML training
│   └── risk_calculator.py # Risk logic
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
└── docs/                 # Documentation
```

### Model Training
```bash
python -c "from src.model_trainer import CropRiskModelTrainer; from src.data_processor import generate_historical_data; trainer = CropRiskModelTrainer(); df = generate_historical_data(); X, y, cols = trainer.prepare_features(df); trainer.train_models(X, y); trainer.save_model()"
```

## 🌐 Deployment Options

### Local Development
```bash
streamlit run app.py
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

## 🌱 Supported Crops

| Crop | Optimal Temp | Rainfall Range | Special Considerations |
|------|-------------|----------------|----------------------|
| Rice | 20-30°C | 100-200mm | Flood tolerance, humidity sensitivity |
| Wheat | 15-25°C | 40-100mm | Cold tolerance, drought sensitivity |
| Maize | 18-28°C | 50-150mm | Wind sensitivity, moderate water needs |
| Cotton | 21-32°C | 60-120mm | Heat tolerance, pest considerations |
| Sugarcane | 25-35°C | 150-300mm | High water needs, long growing season |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- 📧 Email: support@cropriskpredictor.com
- 📱 Issues: GitHub Issues page
- 📖 Docs: `/docs` folder

## 🙏 Acknowledgments

- Weather data: OpenWeatherMap, NOAA
- Agricultural research: FAO, local agricultural universities
- ML libraries: scikit-learn, pandas, numpy
- Web framework: Streamlit

---

**Made with ❤️ for farmers worldwide**
