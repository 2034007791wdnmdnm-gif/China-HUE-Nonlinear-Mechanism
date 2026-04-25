# Urban HUE Analysis with SHAP

## Project Overview

This project analyzes Urban Human Development Index (HUE) using Gradient Boosting Regression and SHAP (SHapley Additive exPlanations) explainability techniques.

## Key Features

- Data preprocessing and feature engineering
- Gradient Boosting Regression model training and evaluation
- SHAP feature importance analysis
- Interaction effect analysis
- 10 types of high-quality visualizations

## Performance Metrics

| Model | R² | Adj R² | RMSE | MAPE(%) |
|-------|-----|--------|------|---------|
| GradientBoosting | 0.897 | 0.874 | 0.086 | 8.51 |
| AdaBoost | 0.828 | 0.791 | 0.111 | 11.43 |
| ExtraTrees | 0.753 | 0.700 | 0.132 | 11.70 |
| XGBoost | 0.700 | 0.636 | 0.146 | 13.57 |
| DecisionTree | 0.697 | 0.632 | 0.147 | 13.18 |
| RandomForest | 0.675 | 0.606 | 0.152 | 13.68 |
| Ridge | 0.513 | 0.409 | 0.186 | 17.81 |
| ElasticNet | 0.505 | 0.399 | 0.187 | 17.91 |

## Project Structure

```
Urban-HUE-Analysis/
├── data/                    # Data directory
│   └── README.md           # Data documentation
├── output/                  # Analysis outputs
│   ├── Fig1_Prediction_Residuals.pdf
│   ├── Fig2_Global_Contribution.pdf
│   ├── Fig3_Dependence_Plots/
│   ├── Fig4_Main_vs_Interaction_All.pdf
│   ├── Fig5_Combined_Matrix.pdf
│   ├── Fig6_Pairwise_Interactions/
│   ├── Fig7_Impact_Interaction_Network.pdf
│   ├── Fig8_SHAP_Heatmap.pdf
│   ├── Fig9_Force_Plots/
│   ├── Fig10_2D_PDP/
│   ├── Dataset_SHAP_Results.xlsx
│   └── Feature_Importance.xlsx
├── src/                     # Source code
│   └── Cities_HUE_Analysis_Enhanced.py
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT License

```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Urban-HUE-Analysis.git
cd Urban-HUE-Analysis
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the analysis
```bash
cd src
python Cities_HUE_Analysis_Enhanced.py
```

### Output
- Model performance metrics
- Bootstrap confidence intervals
- 10 types of visualizations
- SHAP analysis results (Excel format)

## Dependencies

- Python >= 3.8
- scikit-learn >= 1.0
- shap >= 0.41
- matplotlib >= 3.5
- seaborn >= 0.11
- pandas >= 1.3
- numpy >= 1.21
- networkx >= 2.6
- statsmodels >= 0.13

## Dataset Description

The dataset contains the following features:
- **Temp**: Temperature
- **FloatPop**: Floating Population
- **EntPer10k**: Enterprises per 10,000 people
- **UrbRate**: Urbanization Rate
- **Green**: Green Space Coverage
- **RoadArea**: Road Area
- **DispInc**: Dispersion Index
- **AQI**: Air Quality Index
- **GDPpc**: GDP per Capita
- **GovExp**: Government Expenditure
- **EduExp**: Education Expenditure
- **Hosp**: Number of Hospitals
- **Museum**: Number of Museums
- **LandFin**: Land Finance Dependency
- **Pension**: Pension Insurance
- **TIR**: Traffic Accessibility Index
- **Pop**: Total Population

Target variable: **HUE** (Human Development Index)

## Generated Visualizations

1. **Fig1**: Predicted vs Actual Values + Residual Distribution
2. **Fig2**: Global Feature Contribution
3. **Fig3**: Partial Dependence Plots (17 features)
4. **Fig4**: Main Effect vs Interaction Effect Comparison
5. **Fig5**: Combined Interaction Matrix Heatmap
6. **Fig6**: Pairwise Interaction Scatter Plots
7. **Fig7**: Feature Interaction Network
8. **Fig8**: SHAP Heatmap
9. **Fig9**: Individual Sample Force Plots
10. **Fig10**: 2D Partial Dependence Plots

## Statistical Methods

- **Model**: Gradient Boosting Regression
- **SHAP Analysis**: TreeExplainer with interaction values
- **Cross-Validation**: Leave-One-Out Cross-Validation (LOOCV) - for small samples
- **Bootstrap**: 1000 resamples for 95% confidence intervals
- **Residual Tests**: Shapiro-Wilk normality test, Durbin-Watson autocorrelation test

## Results Interpretation

- **Adj R² = 0.874**: Model explains 87.4% of variance
- **LOOCV RMSE = 0.167**: Average prediction error
- **Most Important Features**: Temperature (Temp), Floating Population (FloatPop), Enterprises per 10k (EntPer10k)

## Important Notes

⚠️ This project uses small sample data (n=103) with LOOCV for cross-validation.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- Delin Huang
- Hongqiang Jiang
- Weishan Qin
- Xiaohui Wang

## Acknowledgments

- SHAP development team
- scikit-learn development team
