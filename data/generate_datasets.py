"""
Synthetic and Real-World Canonical Datasets Generator for Econometric Modeling.
Generates:
1. macroeconomic_ts.csv: Time-series exhibiting serial correlation (AR(1)) and time-varying ARCH volatility.
2. housing_prices.csv: Cross-sectional dataset with fan-shaped multiplicative heteroscedasticity.
3. corporate_wages.csv: Cross-sectional wage/education dataset with variance tied to experience.
"""

import numpy as np
import pandas as pd
import os

def generate_datasets(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Macroeconomic Time Series (T = 200 quarters, ~50 years)
    T = 200
    dates = pd.date_range(start="1975-01-01", periods=T, freq="QS")
    
    # Exogenous variables
    interest_rate = np.zeros(T)
    money_growth = np.zeros(T)
    unemployment = np.zeros(T)
    
    ir_val, mg_val, un_val = 5.0, 4.0, 6.0
    for t in range(T):
        ir_val = 0.85 * ir_val + 0.15 * 4.5 + np.random.normal(0, 0.4)
        mg_val = 0.70 * mg_val + 0.30 * 5.0 + np.random.normal(0, 0.6)
        un_val = 0.80 * un_val + 0.20 * 5.5 + np.random.normal(0, 0.3)
        interest_rate[t] = max(0.5, ir_val)
        money_growth[t] = mg_val
        unemployment[t] = max(3.0, un_val)
        
    # True relationship for GDP Growth:
    # GDP_t = 3.0 - 0.4 * InterestRate_t + 0.35 * MoneyGrowth_t - 0.5 * Unemployment_t + epsilon_t
    # Where epsilon_t has AR(1) serial correlation: epsilon_t = 0.72 * epsilon_{t-1} + u_t
    # and u_t has ARCH-like conditional heteroscedasticity (periods of high volatility)
    epsilon = np.zeros(T)
    sigma2 = np.ones(T) * 1.0
    for t in range(1, T):
        sigma2[t] = 0.6 + 0.45 * (epsilon[t-1] ** 2)
        u_t = np.random.normal(0, np.sqrt(sigma2[t]))
        epsilon[t] = 0.72 * epsilon[t-1] + u_t

    gdp_growth = 3.0 - 0.38 * interest_rate + 0.32 * money_growth - 0.45 * unemployment + epsilon

    macro_df = pd.DataFrame({
        "Date": dates.strftime("%Y-Q%q"),
        "GDP_Growth": np.round(gdp_growth, 3),
        "Interest_Rate": np.round(interest_rate, 3),
        "Money_Growth": np.round(money_growth, 3),
        "Unemployment_Rate": np.round(unemployment, 3),
    })
    macro_path = os.path.join(output_dir, "macroeconomic_ts.csv")
    macro_df.to_csv(macro_path, index=False)
    print(f"Generated Macroeconomic Time-Series: {macro_path} (N={len(macro_df)})")

    # 2. Housing Prices Cross-Sectional Dataset (N = 1,000 houses)
    N_houses = 1000
    sqft = np.random.uniform(800, 4800, N_houses)
    lot_size = np.random.uniform(2000, 15000, N_houses)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=N_houses, p=[0.05, 0.20, 0.45, 0.22, 0.08])
    age = np.random.uniform(1, 60, N_houses)
    school_rating = np.random.uniform(2.0, 10.0, N_houses)
    distance_cbd = np.random.uniform(1.0, 30.0, N_houses)
    
    # Classic heteroscedasticity: variance of house price increases dramatically with square footage
    # Price = 50k + 160 * SqFt + 8 * LotSize + 15k * Bedrooms - 1.2k * Age + 18k * SchoolRating - 3.5k * DistCBD + error
    sigma_house = 12000 + 35 * (sqft ** 1.15)  # Variance increases with house size!
    error_house = np.random.normal(0, sigma_house)
    
    price = (50000 
             + 165.0 * sqft 
             + 7.5 * lot_size 
             + 14000.0 * bedrooms 
             - 1100.0 * age 
             + 17500.0 * school_rating 
             - 3200.0 * distance_cbd 
             + error_house)
    # Ensure all prices positive
    price = np.maximum(price, 65000)

    housing_df = pd.DataFrame({
        "Price": np.round(price, 2),
        "Square_Feet": np.round(sqft, 1),
        "Lot_Size": np.round(lot_size, 1),
        "Bedrooms": bedrooms,
        "Age_Years": np.round(age, 1),
        "School_Rating": np.round(school_rating, 2),
        "Distance_CBD_Miles": np.round(distance_cbd, 2),
    })
    housing_path = os.path.join(output_dir, "housing_prices.csv")
    housing_df.to_csv(housing_path, index=False)
    print(f"Generated Housing Cross-Sectional Dataset: {housing_path} (N={len(housing_df)})")

    # 3. Corporate Wage and Human Capital Dataset (N = 800 employees)
    N_wages = 800
    education_years = np.random.choice([10, 12, 14, 16, 18, 20], size=N_wages, p=[0.05, 0.25, 0.20, 0.30, 0.15, 0.05])
    experience_years = np.random.uniform(0, 35, N_wages)
    tenure_months = np.minimum(experience_years * 12, np.random.exponential(48, N_wages))
    firm_size = np.random.choice([25, 100, 500, 2500, 10000], size=N_wages, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    
    # Heteroscedasticity: higher education & experience allows much wider divergence in earnings
    sigma_wage = 3.0 + 0.45 * education_years + 0.35 * experience_years
    error_wage = np.random.normal(0, sigma_wage)
    
    hourly_wage = (8.5 
                   + 2.1 * education_years 
                   + 1.4 * experience_years 
                   - 0.02 * (experience_years ** 2) 
                   + 0.05 * (tenure_months / 12) 
                   + 1.8 * np.log(firm_size) 
                   + error_wage)
    hourly_wage = np.maximum(hourly_wage, 10.0)

    wage_df = pd.DataFrame({
        "Hourly_Wage": np.round(hourly_wage, 2),
        "Education_Years": education_years,
        "Experience_Years": np.round(experience_years, 1),
        "Tenure_Months": np.round(tenure_months, 1),
        "Firm_Size": firm_size,
    })
    wage_path = os.path.join(output_dir, "corporate_wages.csv")
    wage_df.to_csv(wage_path, index=False)
    print(f"Generated Corporate Wage Dataset: {wage_path} (N={len(wage_df)})")

if __name__ == "__main__":
    generate_datasets()
