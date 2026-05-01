"""
Script to generate synthetic agriculture crop dataset for India.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Groundnut', 'Soybean', 'Sunflower', 'Barley', 'Jowar']
states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Madhya Pradesh', 'Gujarat', 'Rajasthan']
seasons = ['Kharif', 'Rabi', 'Zaid']

# Base yield values per crop (kg/ha)
base_yields = {
    'Rice': 2200, 'Wheat': 3100, 'Maize': 2000, 'Sugarcane': 65000, 'Cotton': 450,
    'Groundnut': 1200, 'Soybean': 1100, 'Sunflower': 900, 'Barley': 2400, 'Jowar': 950
}

n = 2000
records = []
for _ in range(n):
    crop = np.random.choice(crops)
    state = np.random.choice(states)
    season = np.random.choice(seasons)
    area = np.round(np.random.uniform(0.5, 1000), 2)
    cost_cultivation = np.round(np.random.uniform(15000, 70000), 2)
    cost_production = np.round(cost_cultivation * np.random.uniform(0.6, 0.9), 2)
    rainfall = np.round(np.random.uniform(300, 2000), 2)
    temperature = np.round(np.random.uniform(15, 40), 2)
    
    base = base_yields[crop]
    noise = np.random.normal(0, base * 0.1)
    yield_val = np.round(max(100, base + noise + (rainfall * 0.5) - (cost_cultivation * 0.002)), 2)
    production = np.round(area * yield_val / 1000, 2)

    records.append({
        'Crop': crop,
        'State': state,
        'Season': season,
        'Area': area,
        'Cost_of_Cultivation': cost_cultivation,
        'Cost_of_Production': cost_production,
        'Annual_Rainfall': rainfall,
        'Temperature': temperature,
        'Yield': yield_val,
        'Production': production
    })

df = pd.DataFrame(records)
df.to_csv('crop_data.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
