<div align="center">
    <img src='assets\logo.jpg' style="border-radius: 15px">
</div>


# **Forecasting with Exogenous variables**

This project focuses on building a modularized and interpretable forecasting pipeline that incorporates exogenous variables, organized into categories like macroeconomic, climatic, holidays, and discounts. The pipeline supports the selection, configuration, and evaluation of forecasting models with various data preprocessing and interpretability techniques.

## **Table of Contents**
1. [Features](#features)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Run the Exogenous Data Selector](#1-run-the-exogenous-data-selector)
   - [Run the Full Pipeline](#2-run-the-full-pipeline)
5. [YAML Configuration](#yaml-configuration)
6. [Results](#results)
7. [Contributing](#contributing)
8. [Future Improvements](#future-improvements)

## **Features**
- **Exogenous Data Selection**:
  - Dynamically select exogenous variables, grouped by type (e.g., macroeconomic, climatic).
  - Persist variable configurations in a YAML file for reuse across experiments.
  - Interactive selection of variables by group or individually.

- **Configurable Data Pipeline**:
  - Modularized data preparation and integration.
  - Support for multiple preprocessing techniques (scaling, transformation, imputation).

- **Forecasting Models**:
  - Evaluate models such as Random Forest, XGBoost, and ARIMA with configurable hyperparameters.
  - Generate interpretable outputs, including SHAP values and Partial Dependence Plots (PDP).

- **Results Management**:
  - Save metrics, predictions, and interpretability plots in an organized directory structure.
  - Experiment-level results stored for easy comparison.



## **Folder Structure**

```
project/
├── src/
│   ├── data/                       # Data preparation and integration
│   │   ├── utils/
│   │   ├── base/
│   │   ├── dataset_makers/
│   │   └── exogenous_data/
│   │       ├── exogenous_data_extractor.py
│   │       └── exogenous_data_selector.py
│   │   └── ...
│   ├── models/                     # Forecasting models and interpretability tools
│   ├── misc/
│   └── run_pipeline.py             # Main script to run the forecasting pipeline
├── results/                        # Organized experiment outputs
│   ├── Experiment_1/
│   │   ├── RandomForest/
│   │   └── ...
├── data.config.yaml                # Configuration for data handling
├── model.config.yaml               # Configuration for model training and evaluation
├── exog.config.yaml                # Configuration for exogenous variables
└── README.md                       # Project documentation
```


## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the required configuration files:
   - `data.config.yaml`
   - `model.config.yaml`
   - `exog.config.yaml`



## **Usage**

### **1. Run the Exogenous Data Selector**
Interactive selection of exogenous variables:
```bash
python src/data/exogenous_data/exogenous_data_selector.py
```

### **2. Run the Full Pipeline**
Execute the forecasting pipeline:
```bash
python src/run_pipeline.py
```



## **YAML Configuration**

### **Example: `exog.config.yaml`**
```yaml
selected_variables: true
exogenous_data:
  macroeconomic:
    - IMACEC
    - PIB
  climatic:
    - tavg
    - tmin
  holidays:
    - es_festivo
  discounts:
    - cyber_monday
    - black_friday
```


## **Results**

Results are saved in the `results/` directory, organized by experiment and model. Each experiment contains:
- **Metrics**: `metrics.json`
- **Predictions**: `predictions.csv`
- **Plots**: SHAP and PDP visualizations
- **Model Configurations**: `config.json`


## **Future Improvements**
- [ ] TODO: bcchile credentials
- [ ] Add support for additional forecasting models (e.g., Prophet, LSTM).
- [ ] Extend interpretability tools for non-tree-based models.
- [ ] Optimize the integration of multiple exogenous variable sources.

---

