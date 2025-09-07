import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
from typing import Dict, List, Tuple, Optional, Any

class WeatherDataProcessor:
    """Handle weather data collection and processing"""
    
    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = os.getenv('OPENWEATHER_API_KEY', '')
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0"
    
    def get_coordinates_by_city(self, city_name: str, country_code: str = "") -> Tuple[float, float]:
        """Get latitude and longitude for a city name"""
        try:
            url = f"{self.geocoding_url}/direct"
            params = {
                'q': f"{city_name},{country_code}" if country_code else city_name,
                'limit': 1,
                'appid': self.api_key
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0]['lat'], data[0]['lon']
            return 0.0, 0.0
        except:
            return 0.0, 0.0
    
    def fetch_current_weather(self, lat: float, lon: float) -> Dict:
        """Fetch current weather data from OpenWeatherMap API"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def fetch_current_weather_by_city(self, city_name: str, country_code: str = "") -> Dict:
        """Fetch current weather data by city name"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': f"{city_name},{country_code}" if country_code else city_name,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def fetch_weather_forecast(self, lat: float, lon: float) -> Dict:
        """Fetch 7-day weather forecast"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}
    
    def process_current_weather_data(self, weather_data: Dict) -> Dict:
        """Extract relevant features from current weather data"""
        if not weather_data or 'main' not in weather_data:
            return self.get_default_weather()
        
        # Extract current weather data
        main = weather_data['main']
        wind = weather_data.get('wind', {})
        rain = weather_data.get('rain', {})
        
        return {
            'temperature': main.get('temp', 25.0),
            'rainfall': rain.get('1h', 0) * 24,  # Convert hourly to daily estimate
            'humidity': main.get('humidity', 70.0),
            'wind_speed': wind.get('speed', 0) * 3.6,  # Convert m/s to km/h
            'pressure': main.get('pressure', 1013),
            'feels_like': main.get('feels_like', main.get('temp', 25.0)),
            'location': weather_data.get('name', 'Unknown'),
            'country': weather_data.get('sys', {}).get('country', ''),
            'description': weather_data.get('weather', [{}])[0].get('description', 'Clear sky')
        }
    
    def process_forecast_data(self, forecast_data: Dict) -> Dict:
        """Extract relevant features from forecast data"""
        if not forecast_data or 'list' not in forecast_data:
            return self.get_default_weather()
        
        forecasts = forecast_data['list'][:21]  # Next 7 days (3-hour intervals)
        
        temps = [f['main']['temp'] for f in forecasts]
        humidity_vals = [f['main']['humidity'] for f in forecasts]
        wind_speeds = [f['wind']['speed'] * 3.6 for f in forecasts]  # Convert to km/h
        
        # Calculate rainfall
        rainfall = sum([f.get('rain', {}).get('3h', 0) for f in forecasts])
        
        return {
            'temperature': np.mean(temps),
            'rainfall': rainfall,
            'humidity': np.mean(humidity_vals),
            'wind_speed': np.mean(wind_speeds)
        }
    
    def get_default_weather(self) -> Dict:
        """Return default weather values for demo"""
        return {
            'temperature': 25.0,
            'rainfall': 50.0,
            'humidity': 70.0,
            'wind_speed': 10.0,
            'pressure': 1013,
            'feels_like': 25.0,
            'location': 'Demo Location',
            'country': 'IN',
            'description': 'Clear sky'
        }
    
    def get_coordinates_by_ip(self) -> Optional[Dict[str, Any]]:
        """
        Get user's approximate location using IP geolocation
        
        Returns:
            Dictionary containing lat, lon, city, country or None if failed
        """
        try:
            # Use ipapi.co for IP geolocation (free service)
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'lat': float(data.get('latitude', 0)),
                    'lon': float(data.get('longitude', 0)),
                    'city': data.get('city', 'Unknown'),
                    'country': data.get('country_code', 'XX'),
                    'region': data.get('region', ''),
                    'country_name': data.get('country_name', '')
                }
        except Exception as e:
            print(f"IP geolocation failed: {e}")
        
        return None
    
    def get_weather_by_ip_location(self) -> Optional[Dict[str, Any]]:
        """
        Get real-time weather data for user's IP-based location
        
        Returns:
            Dictionary containing processed weather data or None if failed
        """
        try:
            # Get location by IP
            location = self.get_coordinates_by_ip()
            if not location:
                print("Could not determine location from IP")
                return self.get_default_weather()
            
            # Get weather data using coordinates
            weather_data = self.fetch_current_weather(location['lat'], location['lon'])
            if not weather_data:
                print(f"Could not fetch weather data for IP location")
                return self.get_default_weather()
            
            # Process and return the weather data
            processed_data = self.process_current_weather_data(weather_data)
            processed_data['location'] = location['city']
            processed_data['country'] = location['country']
            processed_data['region'] = location.get('region', '')
            processed_data['country_name'] = location.get('country_name', '')
            
            return processed_data
            
        except Exception as e:
            print(f"Error getting weather by IP location: {e}")
            return self.get_default_weather()

    def get_real_time_weather_for_location(self, city_name: str, country_code: str = "IN") -> Optional[Dict[str, Any]]:
        """
        Get real-time weather data for a specific location
        
        Args:
            city_name: Name of the city
            country_code: ISO country code (default: IN for India)
            
        Returns:
            Dictionary containing processed weather data or None if failed
        """
        try:
            # Get coordinates for the city
            coords = self.get_coordinates_by_city(city_name, country_code)
            if not coords:
                print(f"Could not get coordinates for {city_name}, {country_code}")
                return self.get_default_weather()
            
            # Get weather data using coordinates
            weather_data = self.fetch_current_weather(coords[0], coords[1])
            if not weather_data:
                print(f"Could not fetch weather data for coordinates {coords[0]}, {coords[1]}")
                return self.get_default_weather()
            
            # Process and return the weather data
            processed_data = self.process_current_weather_data(weather_data)
            processed_data['location'] = city_name.title()
            processed_data['country'] = country_code.upper()
            
            return processed_data
            
        except Exception as e:
            print(f"Error getting weather for location: {e}")
            return self.get_default_weather()

class CropDataProcessor:
    """Handle crop-specific data and risk calculations"""
    
    def __init__(self):
        self.crop_thresholds = {
            'Rice': {
                'temp_min': 15, 'temp_max': 35,
                'rain_min': 100, 'rain_max': 200,
                'humidity_optimal': 80
            },
            'Wheat': {
                'temp_min': 5, 'temp_max': 30,
                'rain_min': 40, 'rain_max': 100,
                'humidity_optimal': 60
            },
            'Maize': {
                'temp_min': 10, 'temp_max': 32,
                'rain_min': 50, 'rain_max': 150,
                'humidity_optimal': 65
            },
            'Cotton': {
                'temp_min': 18, 'temp_max': 35,
                'rain_min': 60, 'rain_max': 120,
                'humidity_optimal': 70
            },
            'Sugarcane': {
                'temp_min': 20, 'temp_max': 38,
                'rain_min': 150, 'rain_max': 300,
                'humidity_optimal': 75
            }
        }
        
        self.growth_sensitivity = {
            'Seedling': 1.5,
            'Vegetative': 1.0,
            'Flowering': 2.0,
            'Maturation': 1.2
        }
    
    def calculate_stress_factors(self, weather: Dict, crop: str, growth_stage: str) -> Dict:
        """Calculate various stress factors for the crop"""
        thresholds = self.crop_thresholds.get(crop, self.crop_thresholds['Rice'])
        sensitivity = self.growth_sensitivity[growth_stage]
        
        # Temperature stress
        temp_stress = 0
        if weather['temperature'] < thresholds['temp_min']:
            temp_stress = (thresholds['temp_min'] - weather['temperature']) / 5
        elif weather['temperature'] > thresholds['temp_max']:
            temp_stress = (weather['temperature'] - thresholds['temp_max']) / 5
        
        # Rainfall stress
        rain_stress = 0
        if weather['rainfall'] < thresholds['rain_min']:
            rain_stress = (thresholds['rain_min'] - weather['rainfall']) / 50
        elif weather['rainfall'] > thresholds['rain_max']:
            rain_stress = (weather['rainfall'] - thresholds['rain_max']) / 100
        
        # Humidity stress
        humidity_stress = abs(weather['humidity'] - thresholds['humidity_optimal']) / 20
        
        # Wind stress
        wind_stress = max(0, (weather['wind_speed'] - 20) / 10)
        
        return {
            'temperature_stress': temp_stress * sensitivity,
            'rainfall_stress': rain_stress * sensitivity,
            'humidity_stress': humidity_stress * sensitivity,
            'wind_stress': wind_stress * sensitivity,
            'total_stress': (temp_stress + rain_stress + humidity_stress + wind_stress) * sensitivity
        }

def generate_historical_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic historical data for model training"""
    np.random.seed(42)
    
    data = []
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    growth_stages = ['Seedling', 'Vegetative', 'Flowering', 'Maturation']
    
    processor = CropDataProcessor()
    
    for _ in range(n_samples):
        # Generate weather conditions
        season = np.random.choice(['Summer', 'Monsoon', 'Winter', 'Spring'])
        
        if season == 'Summer':
            temp = np.random.normal(35, 5)
            rainfall = np.random.exponential(20)
            humidity = np.random.normal(50, 15)
        elif season == 'Monsoon':
            temp = np.random.normal(28, 3)
            rainfall = np.random.exponential(150)
            humidity = np.random.normal(85, 10)
        elif season == 'Winter':
            temp = np.random.normal(18, 8)
            rainfall = np.random.exponential(30)
            humidity = np.random.normal(60, 20)
        else:  # Spring
            temp = np.random.normal(25, 5)
            rainfall = np.random.exponential(60)
            humidity = np.random.normal(65, 15)
        
        wind_speed = np.random.exponential(8) + 2
        crop = np.random.choice(crops)
        growth_stage = np.random.choice(growth_stages)
        soil_moisture = np.random.normal(50, 20)
        
        weather = {
            'temperature': temp,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
        
        # Calculate stress factors
        stress = processor.calculate_stress_factors(weather, crop, growth_stage)
        
        # Determine risk level based on total stress
        if stress['total_stress'] >= 3:
            risk_level = 'High'
        elif stress['total_stress'] >= 1.5:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        data.append({
            **weather,
            'crop': crop,
            'growth_stage': growth_stage,
            'soil_moisture': soil_moisture,
            'risk_level': risk_level,
            'total_stress': stress['total_stress']
        })
    
    return pd.DataFrame(data)