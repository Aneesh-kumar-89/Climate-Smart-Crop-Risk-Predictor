cat > tests/test_risk_calculator.py << 'EOF'
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_calculator import CropRiskCalculator

class TestCropRiskCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calculator = CropRiskCalculator()
    
    def test_high_risk_conditions(self):
        """Test high risk weather conditions"""
        weather = {
            'temperature': 45,  # Very high temperature
            'rainfall': 300,    # Excessive rainfall
            'humidity': 95,     # Very high humidity
            'wind_speed': 40    # Strong winds
        }
        
        risk_level, stress_score, breakdown = self.calculator.calculate_overall_risk(
            weather, 'Rice', 'Flowering', 60
        )
        
        self.assertEqual(risk_level, 'High')
        self.assertGreater(stress_score, 2.0)
    
    def test_low_risk_conditions(self):
        """Test optimal weather conditions"""
        weather = {
            'temperature': 25,  # Optimal temperature
            'rainfall': 75,     # Good rainfall
            'humidity': 65,     # Normal humidity
            'wind_speed': 8     # Light winds
        }
        
        risk_level, stress_score, breakdown = self.calculator.calculate_overall_risk(
            weather, 'Rice', 'Vegetative', 55
        )
        
        self.assertEqual(risk_level, 'Low')
        self.assertLess(stress_score, 1.0)
    
    def test_crop_specific_thresholds(self):
        """Test that different crops have different thresholds"""
        weather = {'temperature': 35, 'rainfall': 50, 'humidity': 60, 'wind_speed': 10}
        
        # Rice should be more stressed by high temperature
        rice_risk, rice_stress, _ = self.calculator.calculate_overall_risk(
            weather, 'Rice', 'Vegetative', 50
        )
        
        # Sugarcane should be less stressed by same temperature
        cane_risk, cane_stress, _ = self.calculator.calculate_overall_risk(
            weather, 'Sugarcane', 'Vegetative', 50
        )
        
        self.assertGreater(rice_stress, cane_stress)

if __name__ == '__main__':
    unittest.main()
EOF