import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class CropRiskCalculator:
    """Calculate crop risk based on weather conditions and crop characteristics"""
    
    def __init__(self):
        # Crop-specific optimal conditions
        self.crop_profiles = {
            'Rice': {
                'temp_range': (20, 30),
                'rainfall_range': (100, 200),
                'humidity_range': (70, 90),
                'critical_stages': ['Flowering', 'Maturation'],
                'stress_multipliers': {'heat': 1.5, 'drought': 2.0, 'flood': 1.8}
            },
            'Wheat': {
                'temp_range': (15, 25),
                'rainfall_range': (40, 100),
                'humidity_range': (50, 70),
                'critical_stages': ['Flowering', 'Maturation'],
                'stress_multipliers': {'heat': 1.3, 'drought': 1.8, 'flood': 1.5}
            },
            'Maize': {
                'temp_range': (18, 28),
                'rainfall_range': (50, 150),
                'humidity_range': (60, 80),
                'critical_stages': ['Flowering'],
                'stress_multipliers': {'heat': 1.4, 'drought': 1.9, 'flood': 1.6}
            },
            'Cotton': {
                'temp_range': (21, 32),
                'rainfall_range': (60, 120),
                'humidity_range': (60, 75),
                'critical_stages': ['Flowering'],
                'stress_multipliers': {'heat': 1.2, 'drought': 1.7, 'flood': 1.4}
            },
            'Sugarcane': {
                'temp_range': (25, 35),
                'rainfall_range': (150, 300),
                'humidity_range': (70, 85),
                'critical_stages': ['Vegetative', 'Maturation'],
                'stress_multipliers': {'heat': 1.1, 'drought': 2.2, 'flood': 1.3}
            }
        }
        
        self.stage_vulnerability = {
            'Seedling': 1.4,
            'Vegetative': 1.0,
            'Flowering': 1.8,
            'Maturation': 1.3
        }
    
    def calculate_weather_stress(self, weather: Dict, crop: str) -> Dict:
        """Calculate individual weather stress components"""
        profile = self.crop_profiles.get(crop, self.crop_profiles['Rice'])
        
        stress_factors = {}
        
        # Temperature stress
        temp = weather['temperature']
        temp_min, temp_max = profile['temp_range']
        if temp < temp_min:
            stress_factors['cold_stress'] = (temp_min - temp) / 10
        elif temp > temp_max:
            stress_factors['heat_stress'] = (temp - temp_max) / 10
        else:
            stress_factors['temperature_stress'] = 0
        
        # Rainfall stress
        rain = weather['rainfall']
        rain_min, rain_max = profile['rainfall_range']
        if rain < rain_min:
            stress_factors['drought_stress'] = (rain_min - rain) / rain_min
        elif rain > rain_max:
            stress_factors['flood_stress'] = (rain - rain_max) / rain_max
        else:
            stress_factors['rainfall_stress'] = 0
        
        # Humidity stress
        humidity = weather['humidity']
        optimal_humidity = profile['humidity_range'][1]
        stress_factors['humidity_stress'] = abs(humidity - optimal_humidity) / 30
        
        # Wind stress
        wind = weather['wind_speed']
        stress_factors['wind_stress'] = max(0, (wind - 25) / 15)
        
        return stress_factors
    
    def calculate_overall_risk(self, weather: Dict, crop: str, growth_stage: str, 
                             soil_moisture: float) -> Tuple[str, float, Dict]:
        """Calculate overall risk level and detailed breakdown"""
        
        # Get weather stress factors
        stress_factors = self.calculate_weather_stress(weather, crop)
        
        # Apply growth stage sensitivity
        stage_multiplier = self.stage_vulnerability[growth_stage]
        
        # Soil moisture stress
        soil_stress = 0
        if soil_moisture < 30:
            soil_stress = (30 - soil_moisture) / 20
        elif soil_moisture > 80:
            soil_stress = (soil_moisture - 80) / 20
        
        # Calculate weighted total stress
        total_stress = 0
        for factor, value in stress_factors.items():
            if 'cold' in factor or 'heat' in factor:
                total_stress += value * 0.3
            elif 'drought' in factor or 'flood' in factor:
                total_stress += value * 0.4
            elif 'humidity' in factor:
                total_stress += value * 0.2
            elif 'wind' in factor:
                total_stress += value * 0.1
        
        total_stress += soil_stress * 0.2
        total_stress *= stage_multiplier
        
        # Determine risk level
        if total_stress >= 2.0:
            risk_level = 'High'
        elif total_stress >= 1.0:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Create detailed breakdown
        breakdown = {
            'total_stress_score': total_stress,
            'stage_sensitivity': stage_multiplier,
            'primary_stress_factors': [k for k, v in stress_factors.items() if v > 0.5],
            'stress_components': stress_factors,
            'soil_stress': soil_stress
        }
        
        return risk_level, total_stress, breakdown