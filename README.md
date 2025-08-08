# BMW & Mercedes-Benz Car Specifications Forecasting with Facebook Prophet

## 📊 Project Overview

This project demonstrates the application of Facebook Prophet for forecasting automotive technology trends, specifically focusing on BMW and Mercedes-Benz vehicle specifications from 1945-2020. The analysis predicts future trends in maximum horsepower and minimum acceleration up to 2050.

## 🎯 Key Objectives

- **Technology Trend Forecasting**: Predict future automotive performance metrics
- **Model Validation**: Backtest predictions using historical data
- **Uncertainty Quantification**: Provide confidence intervals for forecasts
- **Performance Analysis**: Evaluate Prophet's effectiveness for nonlinear forecasting

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Facebook Prophet**: Time series forecasting
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Model validation metrics

## 📈 Methodology

### 1. Data Preprocessing
- **Dataset**: Kaggle car specifications (1945-2020)
- **Filtering**: BMW and Mercedes-Benz vehicles only
- **Cleaning**: Removed missing values and outliers
- **Formatting**: Structured data for Prophet input requirements

### 2. Model Development
- **Separate Models**: Individual Prophet models for horsepower and acceleration
- **Growth Constraints**: Applied logistic growth constraints
- **Regressors**: Added time-based regressors for performance trends
- **Forecasting Horizon**: Extended predictions to 2050

### 3. Validation Strategy
- **Backtesting**: Used 2010 as cutoff year
- **Error Metrics**: Calculated normalized error rates
- **Comparison**: Partial vs. full data predictions
- **Accuracy Assessment**: Actual vs. predicted values (1980-2017)

## 📊 Results

### Model Performance
- **Horsepower Forecast Error**: 8.91%
- **Acceleration Forecast Error**: 15.78%
- **Confidence Intervals**: Provided uncertainty quantification
- **Trend Analysis**: Identified performance improvement trajectories

### Key Insights
- **Horsepower Trends**: Consistent upward trajectory with diminishing returns
- **Acceleration Trends**: Decreasing 0-100 km/h times (improving performance)
- **Technology Evolution**: Clear patterns in automotive advancement
- **Manufacturer Competition**: Similar trends between BMW and Mercedes-Benz

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn prophet scikit-learn
```

### Data Requirements
- Kaggle car specifications dataset
- BMW and Mercedes-Benz vehicle data (1945-2020)
- Performance metrics: horsepower, acceleration, fuel consumption

### Model Training
```python
# Example Prophet model setup
from prophet import Prophet

# Configure model with growth constraints
model = Prophet(
    growth='logistic',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# Add custom regressors for performance trends
model.add_regressor('performance_trend')
```

## 📁 Project Structure

```
bmw-mercedes-prophet-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── horsepower_model.pkl
│   └── acceleration_model.pkl
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_validation_analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   └── visualization.py
├── results/
│   ├── forecasts/
│   └── visualizations/
└── documentation/
    └── (PDF available via Google Drive link)
```

## 📄 Documentation

This project includes comprehensive PDF documentation that provides:
- **Detailed Methodology**: Step-by-step implementation guide
- **Code Explanations**: In-depth analysis of key algorithms
- **Results Analysis**: Detailed interpretation of forecasting results
- **Validation Procedures**: Complete backtesting methodology

**📖 View Documentation**: [Code_Documentation_Facebook_Prophet.pdf](https://drive.google.com/file/d/1qgidOPYxjxfbk7WNfMsS9lWws5D-6szC/view?usp=sharing)

## 🔬 Research Applications

This project demonstrates:
- **Technology Roadmapping**: Long-term automotive trend prediction
- **Competitive Intelligence**: Understanding manufacturer strategies
- **Investment Analysis**: Supporting automotive industry decisions
- **Policy Planning**: Informing regulatory frameworks

## 📚 References

- **Original Dataset**: [Kaggle Car Specifications](https://www.kaggle.com/datasets/CooperUnion/car-dataset)
- **Facebook Prophet**: [Official Documentation](https://facebook.github.io/prophet/)
- **Technology Forecasting**: Academic literature on automotive trends

## 👨‍💻 Author

**Arshia Feizmohammady**
- Industrial Engineering Student, University of Toronto
- Research focus: Quantitative analysis and machine learning
- [LinkedIn](https://linkedin.com/in/arshiafeiz)
- [Personal Website](https://arshiafeizmohammady.com)

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic or commercial applications.

---

*This project was completed as part of research work at ISAE-SUPAERO, focusing on technology forecasting and competitive analysis in the automotive industry.*
