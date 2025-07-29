import unittest
import pandas as pd
from app import predict_stage, validate_data, generate_report

class TestLiverCirrhosisApp(unittest.TestCase):

    def test_data_upload_valid(self):
        """Test valid data upload"""
        df = pd.DataFrame({
            'Age': [45],
            'Gender': ['Male'],
            'Bilirubin': [2.5],
            'Albumin': [3.1],
            'INR': [1.2],
            'Stage': [3]
        })
        result = validate_data(df)
        self.assertTrue(result)

    def test_data_upload_invalid(self):
        """Test invalid data upload with missing columns"""
        df = pd.DataFrame({
            'Age': [45],
            'Bilirubin': [2.5]
        })
        result = validate_data(df)
        self.assertFalse(result)

    def test_prediction_valid_data(self):
        """Test prediction with valid data"""
        input_data = {
            'Age': 50,
            'Gender': 'Female',
            'Bilirubin': 3.2,
            'Albumin': 3.8,
            'INR': 1.5
        }
        result = predict_stage(input_data)
        self.assertIn(result, [1, 2, 3, 4])  # Assuming 4 stages

    def test_prediction_invalid_data(self):
        """Test prediction with invalid data"""
        input_data = {
            'Age': '',
            'Gender': 'Unknown',
            'Bilirubin': 'abc'
        }
        with self.assertRaises(ValueError):
            predict_stage(input_data)

    def test_report_generation(self):
        """Test report generation"""
        prediction_data = {
            'Age': 45,
            'Gender': 'Male',
            'Stage': 3
        }
        result = generate_report(prediction_data)
        self.assertTrue(result.endswith('.pdf'))

if __name__ == '__main__':
    unittest.main()
