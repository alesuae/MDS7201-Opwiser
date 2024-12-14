<div align="center">
    <img src='assets\logo.jpg' style="border-radius: 15px">
</div>


# **Forecasting with Exogenous variables**

This project focuses on building a modularized and interpretable forecasting pipeline that incorporates exogenous variables, organized into categories like macroeconomic, climatic, holidays, and discounts. The pipeline supports the selection, configuration, and evaluation of forecasting models with various data preprocessing and interpretability techniques.

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example Usage](#example-usage)
5. [Future Improvements](#future-improvements)
6. [Contributing](#contributing)

## **Features**
- **Flexible Configuration with YAML files**
  - Define parameters for data, exogenous variables and models using YAML files for easy customization and reusability
  - Enables adjustments without modifying the core codebase
- **Exogenous Data Selection**:
  - Dynamically select exogenous variables, grouped by type (e.g., macroeconomic, climatic).
  - Persist variable configurations in a YAML file for reuse across experiments.
  - Interactive selection of variables by group or individually.
- **Modular and Automated Data Pipeline**
  - Automatic data preprocessing, including: cleaning, normalization and imputation, transformations like log scaling, imputation and feature engineering. Advanced techniques like SMOTENC to hanlde imbalanced categories.
- **Configurable Data Pipeline**:
  - Modularized data preparation and integration.
  - Support for multiple preprocessing techniques (scaling, transformation, imputation).
- **Tracking with MLFlow**
  1. Forecasting Models
    - Evaluate machine learning models such as Random Forest, XGBoost, and traditional models like ARIMA and Prophet, with configurable hyperparameters.
  2. Tracks and Logs
    - Enables tracking of hyperparameters, metrics (e.g., MAE, RMSE, WAPE), visualizations (residuals, real vs. predicted values, error distributions, etc) and SHAP plots to interpret feature contributions.
  3. Enables seamless comparison of experiments in a centralized environment.
- **Designed for Comparative Experiments**
  - Designed to evaluate the impact of adding exogenous variables compared to predictions based on historical sales and stock.

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv env
   source env/bin/activate  
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your **[Banco Central de Chile (BCChile)](https://si3.bcentral.cl/Siete/es/Siete/API?respuesta=)** API credentials**:
   - Create a file named ```creds.txt``` in the root directory of the project.

5. **Set up the required configuration files**:
   - Update the ```data.config.yaml```, ```exog.config.yaml```, and ```model.config.yaml``` files according to your dataset and modeling requirements. [See example on [Wiki](https://github.com/alesuae/MDS7201-Opwiser/wiki)]


## **Usage**

This project is managed using a ```Makefile``` to simplify common tasks. Below are the available commands and their descriptions:

**Track experiments with MLFlow**:

  To start the MLFlow tracking server and visualize experiment results:

  ```bash
  make mflow
  ```

**Running the machine learning pipeline**:

  To run the entire data preprocessing, model training and evaluation pipeline:
  >**Please make sure your MLFlow tracking server is running before running any pipelines**

  ```bash
  make run
  ```

**Cleaning the processed data**:

  To remove all processed data and split files in the ```data/processed``` and ```data/splits``` directories:

  ```bash
  make clean
  ```

**Cleaning the processed data**:

  To remove all processed data and split files in the ```data/processed``` and ```data/splits``` directories:

  ```bash
  make clean
  ```

**Additional Notes**
  1. **API Credentials**: Ensure you have added your Banco Central de Chile (BCChile) API credentials in a creds.txt file as described in the [installation](#installation) section.
  2. **YAML Configuration**: Adjust the YAML files to align with your dataset and modeling requirements.
  3. **MLFlow**: **Please make sure your MLFlow tracking server is running before running any pipelines**


## **Example Usage**
Below is a walkthrough for using the pipeline to predict weekly sales with exogenous variables. First, ensure all dependencies are installed and that you have a file named `creds.txt` in the root directory

#### 1. Configure Your Pipeline
Adjust the YAML configuration files to match your dataset and analysis needs:
- **`data.config.yaml`**: Define data preprocessing methods (e.g., scaling, imputations) and aggregation levels.
- **`exog.config.yaml`**: Select the exogenous variables to include and configure data sources like BCChile.
- **`model.config.yaml`**: Specify model hyperparameters and evaluation metrics.

#### 2. Start MLflow Tracking Server
Launch the MLflow tracking server to compare and analyze experiments. Access the server at [http://localhost:5000](http://localhost:5000).
```bash
make mlflow
```


#### 3. Run the Full Pipeline
Execute the entire pipeline, including data preparation, model training, and evaluation:
```bash
make run
```

This command will:
- Preprocess raw data and split it into training and testing sets.
- Aggregate data weekly by product categories.
- Fetch macroeconomic data from BCChile and merge it with the main dataset.
- Train baseline model and machine learning models (e.g., LightGBM, XGBoost).
- Log all metrics, plots, and trained models to MLflow for tracking.

#### 4. Clean Processed Data (Optional)
To reset and clean the processed data:
```bash
make clean-data
```

#### 5. Evaluate Results
After running the pipeline:
- Visit the MLflow interface to compare model performance metrics (e.g., RMSE, WAPE, R²).
- Use the generated SHAP plots to interpret feature importance.
- Check the prediction plots stored in the `mlruns` folder for insights into predicted vs. actual values.


This example workflow demonstrates how to prepare the data, configure the pipeline, and analyze results efficiently.

## **Future Improvements**
- [X] Add support for additional forecasting models (e.g., Prophet, LSTM).
- [X] Extend interpretability tools for non-tree-based models.
- [ ] Optimize the integration of multiple exogenous variable sources.

## Contribution Guidelines

We welcome contributions to improve this project! Whether it’s fixing bugs, adding new features, improving documentation, or sharing feedback, your input is greatly appreciated. To maintain code quality and consistency:
1. Follow the [Conventional Commits](https://www.conventionalcommits.org/) standard for commit messages.
2. Test your changes thoroughly before submitting a pull request.
3. Use `make clean-data` to reset the dataset if required.
4. For any questions, start a discussion in the **Issues** tab.



