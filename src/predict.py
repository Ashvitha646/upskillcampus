"""
Agriculture Crop Production Prediction — Prediction System
Usage: python predict.py
"""
import pickle
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load model and encoders
with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

le_crop   = encoders['crop']
le_state  = encoders['state']
le_season = encoders['season']

CROPS   = sorted(le_crop.classes_.tolist())
STATES  = sorted(le_state.classes_.tolist())
SEASONS = sorted(le_season.classes_.tolist())


def predict_yield(crop, state, season, area, cost_cultivation, cost_production, rainfall, temperature):
    """
    Predict crop yield given input features.

    Parameters
    ----------
    crop             : str   e.g. 'Rice'
    state            : str   e.g. 'Punjab'
    season           : str   e.g. 'Kharif'
    area             : float Cultivated area in hectares
    cost_cultivation : float Cost of cultivation per hectare (₹)
    cost_production  : float Cost of production per hectare (₹)
    rainfall         : float Annual rainfall in mm
    temperature      : float Average temperature in °C

    Returns
    -------
    float  Predicted yield in kg/ha
    """
    try:
        crop_enc   = le_crop.transform([crop])[0]
        state_enc  = le_state.transform([state])[0]
        season_enc = le_season.transform([season])[0]
    except ValueError as e:
        print(f"Encoding error: {e}")
        return None

    features = np.array([[crop_enc, state_enc, season_enc,
                          area, cost_cultivation, cost_production,
                          rainfall, temperature]])
    return model.predict(features)[0]


def interactive_predict():
    print("=" * 50)
    print("  CROP YIELD PREDICTION SYSTEM")
    print("=" * 50)
    print(f"\nAvailable Crops   : {', '.join(CROPS)}")
    print(f"Available States  : {', '.join(STATES)}")
    print(f"Available Seasons : {', '.join(SEASONS)}")
    print()

    crop   = input("Enter Crop   : ").strip().title()
    state  = input("Enter State  : ").strip().title()
    season = input("Enter Season : ").strip().title()
    area   = float(input("Enter Area (ha): "))
    cost_c = float(input("Cost of Cultivation (₹/ha): "))
    cost_p = float(input("Cost of Production  (₹/ha): "))
    rain   = float(input("Annual Rainfall (mm)       : "))
    temp   = float(input("Average Temperature (°C)   : "))

    result = predict_yield(crop, state, season, area, cost_c, cost_p, rain, temp)
    if result:
        print(f"\n✅ Predicted Yield : {result:.2f} kg/ha")
        print(f"   Estimated Production : {result * area / 1000:.2f} tonnes")


# ── Batch examples ─────────────────────────────────
if __name__ == '__main__':
    examples = [
        ('Rice',      'Punjab',          'Kharif', 200,  35000, 25000, 1000, 28),
        ('Wheat',     'Haryana',         'Rabi',   150,  28000, 20000, 600,  22),
        ('Cotton',    'Maharashtra',     'Kharif', 80,   42000, 32000, 700,  33),
        ('Sugarcane', 'Uttar Pradesh',   'Kharif', 50,   68000, 52000, 950,  30),
        ('Maize',     'Karnataka',       'Kharif', 120,  22000, 16000, 800,  27),
    ]

    print("=" * 70)
    print(f"  {'Crop':<12} {'State':<20} {'Season':<8} {'Predicted Yield (kg/ha)':>22}")
    print("=" * 70)
    for args in examples:
        y = predict_yield(*args)
        print(f"  {args[0]:<12} {args[1]:<20} {args[2]:<8} {y:>22.2f}")
    print("=" * 70)
    print("\nRun `interactive_predict()` in Python for manual input.\n")
