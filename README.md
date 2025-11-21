# Traffic & CO₂ Emission Forecasting using ARO-Optimized LSTM

A reproducible multi-sector CO₂ emission forecasting framework built with LSTM and ARO optimization.

This repository contains a demo implementation of a **multi-sector CO₂ emission forecasting model** based on a Long Short-Term Memory (LSTM) network optimized using the **Artificial Rabbits Optimization (ARO)** algorithm.  
It demonstrates my research workflow in intelligent highway energy systems, with a focus on:

- Multivariate time-series forecasting  
- CO₂ emission estimation  
- Hyperparameter optimization using metaheuristics  
- Data preprocessing and reproducible pipelines

> ⚠️ This repository uses synthetic example data for demonstration purposes only.  
> No proprietary or project-specific data is included.

---

## 🔍 Background

Accurate forecasting of traffic-related CO₂ emissions is essential for smart highway management and low-carbon operations.  
Traditional models (ARIMA, SARIMA, SVR, GBM) have limitations handling:

- Long-range temporal dependencies  
- Highly nonlinear emission dynamics  
- Multi-sector correlations  

Deep learning solves part of the problem, but the performance of models such as LSTM heavily depends on hyperparameter tuning.

To address these gaps, this demo repo implements:

- **An LSTM-based CO₂ forecasting model**  
- **ARO metaheuristic optimization** for key hyperparameters  
- **A clean data → model → evaluation workflow**

---

## 📁 Repository Structure

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
│   └── prediction_plot.png
```

---

## 📊 Data Format

The synthetic dataset follows this structure:

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

The dataset is automatically **pivoted into wide format** for multi-task forecasting.

---

## 🧠 Model Overview

### 🔹 Multi-task LSTM (7 output sectors)

- Input dimension: 7  
- Output dimension: 7  
- Captures long-term temporal dependencies  
- Learns cross-sector relationships  

### 🔹 ARO Hyperparameter Optimization

The ARO optimizer tunes:

- Hidden size  
- Number of LSTM layers  
- Learning rate  

A lightweight ARO implementation is provided for demonstration.

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model

```bash
python src/train.py
```

### 3️⃣ Evaluate and generate prediction plot

```bash
python src/evaluate.py
```

### 4️⃣ Run the end-to-end Notebook

Open:

```
notebooks/demo_forecasting.ipynb
```

---

## 📈 Example Output

The evaluation script produces a prediction plot comparing:

- Real vs. Predicted CO₂ (Total sector)

Saved to:

```
results/prediction_plot.png
```

To embed the plot directly in README:

```markdown
![Prediction Plot](results/prediction_plot.png)
```

---

## 🔮 Future Work

Potential extensions include:

- Transformer-based CO₂ forecasting  
- GCN–LSTM hybrid spatial–temporal models  
- Multi-region emission prediction  
- Multi-energy flow modeling for smart highways  
- Adaptive online learning for dynamic systems  
- Integration with C-V2X systems  

---

## 📜 License

MIT License  
Free for research and non-commercial use.
