# Traffic & CO₂ Emission Forecasting using ARO-Optimized LSTM

![python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![license](https://img.shields.io/badge/License-MIT-green.svg)
![status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![build](https://img.shields.io/badge/Model-LSTM%20%2B%20ARO-orange.svg)

---

<div align="center">

### **A reproducible multi-sector CO₂ emission forecasting framework based on LSTM + ARO optimization**

💡 *Smart highway decarbonization • Multivariate time-series • Metaheuristic tuning • Reproducible pipelines*

</div>

---

This repository contains a demo implementation of a **multi-sector CO₂ emission forecasting model** based on a Long Short-Term Memory (LSTM) network optimized using the **Artificial Rabbits Optimization (ARO)** algorithm. It demonstrates my research workflow in intelligent highway energy systems, including:

- Multivariate time-series forecasting  
- CO₂ emission estimation  
- Hyperparameter tuning using ARO  
- Structured preprocessing and reproducible pipelines  

> ⚠️ Synthetic data only.  
> No proprietary or project-specific data is used.

---

##  Background

Accurate forecasting of traffic-related CO₂ emissions is essential for smart highway operation and low-carbon transportation planning.  
Traditional models (ARIMA, SARIMA, SVR, GBM) often struggle with:

- Long temporal dependencies  
- Highly nonlinear multi-sector dynamics  
- Limited cross-sector modelling ability  

To address these challenges, this demo integrates:

- **LSTM deep sequence models**  
- **ARO metaheuristic optimization**  
- **Multi-output CO₂ forecasting design**  

---

##  Repository Structure

```plaintext
traffic-co2-aro-lstm/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── sample_data.csv
│
├── src/
│   ├── data_loader.py
│   ├── model_lstm.py
│   ├── aro_optimizer.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   └── demo_forecasting.ipynb
│
├── results/
│   └── full_sequence_vs_real.png
```

---

##  Data Format

| area    | co2       | sector           | date       |
|---------|-----------|------------------|------------|
| RegionA | 2.521743  | Ground Transport | 2024-01-01 |
| RegionA | 10.453925 | Industry         | 2024-01-01 |
| RegionA | 14.875310 | Power            | 2024-01-01 |
| RegionA | 4.348890  | Residential      | 2024-01-01 |
| ...     | ...       | ...              | ...        |

The dataset includes **7 predicted emission sectors**:

- Domestic Aviation  
- Ground Transport  
- Industry  
- International Aviation  
- Power  
- Residential  
- Total  

The dataset is automatically pivoted into wide-format for multi-task forecasting.

---

##  Model Overview

###  Multi-task LSTM (7 outputs)

- Input dimension: 7  
- Output dimension: 7  
- Learns long-range dependencies and cross-sector relationships  

###  ARO Hyperparameter Optimization

The ARO algorithm searches:

- Hidden size  
- Number of LSTM layers  
- Learning rate  

Lightweight implementation included in `aro_optimizer.py`.

---

##  How to Run

### 1️ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️ Train the model
```bash
python src/train.py
```

### 3️ Evaluate & generate prediction plot
```bash
python src/evaluate.py
```

### 4️ Run full demo Notebook
```
notebooks/demo_forecasting.ipynb
```

---

##  Results(Outputs)

Below is an example of the generated prediction plot (`results/full_sequence_vs_real.png`):

> *Real vs Predicted CO₂ — Total Sector*

![Full Sequence CO₂ Forecast](results/full_sequence_vs_real.png)





---

##  Technical Stack

| Category | Tools |
|---------|-------|
| Deep Learning | PyTorch, LSTM |
| Optimization | ARO (Artificial Rabbits Optimization) |
| Data | Pandas, NumPy, MinMaxScaler |
| Visualization | Matplotlib |
| Reproducibility | Jupyter Notebook |

---

##  Workflow Diagram

```plaintext
Raw Data (CSV)
      ↓
Pivot to wide format (7 sectors)
      ↓
Normalization & sequence generation
      ↓
Multi-task LSTM model
      ↓
ARO hyperparameter optimization
      ↓
Training → Validation → Testing
      ↓
Prediction Plot (results/full_sequence_vs_real.png)
```

---

##  Citation

If you use or reference this repository:

```bibtex
@misc{liu2024trafficCO2,
  title        = {traffic-co2-aro-lstm: Multi-sector CO₂ Forecasting with LSTM + ARO},
  author       = {Liu, Xiaoya},
  year         = {2024},
  howpublished = {GitHub repository},
  url          = {https://github.com/ciciliu122/traffic-co2-aro-lstm}
}
```

---

##  Future Work

- Transformer-based CO₂ forecasting  
- Spatial–temporal GCN + LSTM hybrid models  
- Multi-region and multi-energy forecasting  
- Online adaptive learning  
- Integration with C-V2X traffic systems  

---

##  License

MIT License — free for research and non-commercial use.

---

