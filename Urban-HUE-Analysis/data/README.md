# Data Directory

This directory contains dataset files for the Urban HUE Analysis project.

## Data Files

The dataset should contain the following columns:

### Feature Variables
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

### Target Variable
- **HUE**: Human Development Index

## Sample Size

- **Minimum**: 100 samples
- **Recommended**: 200+ samples

## Data Format

Data should be in Excel format (.xlsx) or CSV format (.csv)

## Important Notes

⚠️ Since data may contain sensitive information, please ensure:
1. Data is anonymized before uploading
2. Personal identification information is removed
3. Data usage agreements are followed

## Usage Example

```python
import pandas as pd

# Load data
df = pd.read_excel('your_data.xlsx')

# Check data
print(df.head())
print(df.describe())
```

## Generating Sample Data

To generate sample data for testing, you can use the `generate_sample_data()` function in the main analysis code.
