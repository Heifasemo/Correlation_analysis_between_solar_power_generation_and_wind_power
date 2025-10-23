# Correlation analysis between solar power generation and wind power

This repository analyzes the correlation between solar photovoltaic generation and wind characteristics, including preprocessing, training, and prediction modules.

## Repository Structure
```
Correlation_analysis_between_solar_power_generation_and_wind_power/
├─ src/
│  ├─ MLP.py          # Residual MLP + LightGBM hybrid model for solar power prediction
│  ├─ predict.py      # Predict solar power for new regions (transfer test support)
│  └─ standard.py     # Data preprocessing and standardization for 5B dataset
├─ data/
│  ├─ 5B/
│  │  └─ final_merged.csv        # Preprocessed dataset from 5B solar power station (training source)
│  └─ UNSW/
│     └─ 2024_Mech_Weather_Station_Data.xlsx  # UNSW laboratory environmental dataset (transfer test, unchanged)
├─ model/             # Trained models and scalers will be stored here
└─ outputs/           # Prediction results and analysis exports
```

## Usage
1. **Standardize and prepare training data**
```bash
python src/standard.py
```
2. **Train the model (5B station data)**
```bash
python src/MLP.py
```
3. **Predict for new regions or UNSW lab data**
Edit `predict.py` input/output paths and run:
```bash
python src/predict.py
```

## Notes
- `final_merged.csv` comes from the 5B solar power station.
- `2024_Mech_Weather_Station_Data.xlsx` is from UNSW lab and kept unchanged.
- StandardScaler and model artifacts are reused during transfer testing.
