# ⚡🔌 PJM ENERGY DEMAND FORECASTER 🔌⚡

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=F7B93E&center=true&vCenter=true&width=800&lines=Predict+Energy+Demand+with+Machine+Learning;10%2B+Years+of+Hourly+PJM+Load+Data;Random+Forest+%2B+Advanced+Time-Series+Engineering;Interactive+Streamlit+Dashboard+%F0%9F%94%A5)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F79310E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-success?style=for-the-badge&logo=streamlit)](https://pjm-energy-demand-forecaster-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/PJM-Energy-Demand-Forecaster?style=social)](https://github.com/mayank-goyal09/PJM-Energy-Demand-Forecaster/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/PJM-Energy-Demand-Forecaster?style=social)](https://github.com/mayank-goyal09/PJM-Energy-Demand-Forecaster/network)

![Energy Grid](https://user-images.githubusercontent.com/74038190/212748830-4c709398-a386-4761-84d7-9e10b98fbe6e.gif)

### ⚡ **Forecast hourly energy consumption like a grid operator** using **Random Forest + Time-Series ML** 🤖

### 📊 10+ Years of PJM Data × AI = **Smart Grid Intelligence** 💡

---

## 🌟 **WHAT IS THIS?** 🌟

<table>
<tr>
<td width="50%">

### ⚡ **The Magic**

This **ML-powered energy demand forecaster** predicts hourly electricity consumption using **Random Forest Regression** with advanced **time-series feature engineering** across 10+ years of PJM Interconnection load data. Upload temporal features and get instant energy demand predictions with interactive visualizations!

**Think of it as:**
- 🧠 Brain = Random Forest Regressor
- 📊 Input = Time-Series Features (Hour, Day, Month, Season)  
- ⚡ Output = Predicted Energy Demand (MW)

</td>
<td width="50%">

### 🔥 **Key Features**

✅ Random Forest with hyperparameter tuning  
✅ Advanced time-series feature engineering  
✅ Multiple PJM regions (AEP, COMED, DAYTON, DEOK, DOM)  
✅ Interactive Plotly visualizations  
✅ **Real-time demand predictions** 🕒  
✅ Beautiful Streamlit UI with mobile support  

**Performance Metrics:**
- 📉 **MAE**: ~500 MW  
- 📊 **RMSE**: ~700 MW  
- 🎯 **R²**: 0.95+ (High accuracy)

</td>
</tr>
</table>

---

## 🛠️ **TECH STACK** 🛠️

![Tech Stack](https://skillicons.dev/icons?i=python,github,vscode,git)

| **Category** | **Technologies** |
|-------------|-----------------|
| 🐍 **Language** | Python 3.8+ |
| 📊 **Data Science** | Pandas, NumPy, Scikit-learn |
| 🎨 **Frontend** | Streamlit |
| 📈 **Visualization** | Plotly, Matplotlib, Seaborn |
| 🧪 **Model** | Random Forest Regressor, GridSearchCV |
| 🔧 **Feature Engineering** | Time-series decomposition, lag features |
| 💾 **Serialization** | Joblib, Parquet |
| 📦 **Data Storage** | CSV, Parquet files |

---

## 📂 **PROJECT STRUCTURE** 📂

```
⚡ PJM-Energy-Demand-Forecaster/
│
├── 📁 app.py                         # Streamlit web application
├── 📁 main.ipynb                     # Model training, EDA & hyperparameter tuning
├── 📦 requirements.txt               # Dependencies
├── 💾 est_hourly.parquet             # Processed energy demand data
├── 📊 AEP_hourly.csv                 # American Electric Power region data
├── 📊 COMED_hourly.csv               # Commonwealth Edison region data
├── 📊 DAYTON_hourly.csv              # Dayton Power & Light region data
├── 📊 DEOK_hourly.csv                # Duke Energy Ohio/Kentucky region data
├── 📊 DOM_hourly.csv                 # Dominion Virginia Power region data
├── 📋 best_hyperparameters.csv       # Optimized RF parameters
├── 📋 model_metadata.json            # Model performance metrics
├── 📁 portfolio_images/              # Visual assets for README
├── 🔒 .gitignore                     # Git ignore file
└── 📖 README.md                      # You are here!
```

---

## 🚀 **QUICK START** 🚀

![Rocket](https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-705f7be0b224.gif)

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/PJM-Energy-Demand-Forecaster.git
cd PJM-Energy-Demand-Forecaster
```

### **Step 2: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 3: Run the App** 🎯

```bash
streamlit run app.py
```

### **Step 4: Open in Browser** 🌐

The app will automatically open at: **`http://localhost:8501`**

---

## 🎮 **HOW TO USE** 🎮

<table>
<tr>
<td width="50%">

### 🔹 **Quick Prediction Mode**

1. Open the app
2. Select time parameters:
   - Hour of day (0-23)
   - Day of week (Monday-Sunday)
   - Month (January-December)
   - Season (Spring, Summer, Fall, Winter)
3. Click **"Predict Demand"**
4. View predicted energy consumption with charts!

</td>
<td width="50%">

### 🔹 **Historical Analysis** 📊

1. Navigate to **"Historical Data"** tab
2. Select PJM region:
   - AEP (American Electric Power)
   - COMED (Commonwealth Edison)
   - DAYTON (Dayton Power & Light)
   - DEOK (Duke Energy OH/KY)
   - DOM (Dominion VA Power)
3. Explore time-series visualizations
4. Analyze seasonal patterns

</td>
</tr>
</table>

---

## 🧪 **HOW IT WORKS** 🧪

```mermaid
graph LR
    A[Historical PJM Data] --> B[Time-Series Feature Engineering]
    B --> C[Random Forest Training]
    C --> D[Hyperparameter Tuning]
    D --> E[Model Deployment]
    E --> F[Streamlit App]
    F --> G[Real-Time Predictions]
    G --> H[Interactive Visualizations]
```

### **Pipeline Breakdown:**

1️⃣ **Data Collection** → 10+ years of hourly PJM load data across 5 regions  
2️⃣ **Feature Engineering** → Extract temporal features:
   - Hour of day (0-23)
   - Day of week (0-6)
   - Month (1-12)
   - Season (categorical)
   - Lag features (past hour, day, week)  
3️⃣ **Model Training** → Random Forest Regressor with GridSearchCV  
4️⃣ **Hyperparameter Optimization** → Find best n_estimators, max_depth, min_samples_split  
5️⃣ **Evaluation** → MAE, RMSE, R² on test set  
6️⃣ **Deployment** → Streamlit app with Plotly visualizations  

---

## 📊 **DATASET & FEATURES** 📊

![Data Analysis](https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif)

### **Dataset Overview**

- 📍 **Source**: PJM Interconnection (Kaggle)
- 📏 **Size**: 121,273 hourly records (Oct 2004 - Aug 2018)
- 🌍 **Regions**: 5 PJM territories (AEP, COMED, DAYTON, DEOK, DOM)
- 🎯 **Target Variable**: `MW` (Megawatts consumed)

### **Feature Categories**

| **Feature Type** | **Features** |
|-----------------|-------------|
| ⏰ **Temporal** | hour, day_of_week, month, season |
| 📅 **Calendar** | is_weekend, is_holiday |
| 📈 **Lag Features** | lag_1h, lag_24h, lag_168h |
| 🌡️ **Seasonal** | season_encoded (Spring/Summer/Fall/Winter) |

### **Top 3 Predictive Features** (from Feature Importance)

1. 🕐 **hour** → Strongest predictor (35%+ importance)
2. 📆 **month** → Seasonal demand cycles (25%+ importance)
3. 🌡️ **season** → Weather-driven patterns (20%+ importance)

---

## 🎨 **FEATURES SHOWCASE** 🎨

### ✨ **What Makes This Special?**

```python
# Feature Highlights

features = {
    "Interactive Predictions": "⚡ Plotly time-series charts",
    "Feature Importance": "📊 Bar chart showing top predictors",
    "Regional Analysis": "🗺️ Compare demand across 5 PJM regions",
    "Mobile Friendly": "📱 Responsive UI with clean layout",
    "No Sliders": "✅ Dropdown selectors for easy input",
    "Premium Charts": "🎨 Professional Plotly visualizations",
    "Real-Time Updates": "🔄 Instant prediction recalculation",
}
```

### **App Sections:**

1. **⚡ Energy Demand Predictor** → Fast input with dropdowns
2. **📈 Historical Trends** → Time-series exploration
3. **🗺️ Regional Comparison** → Multi-region analysis
4. **🧠 Model Insights** → Feature importance dashboard

---

## 💡 **BUSINESS USE CASES** 💡

![Business Use Cases](https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif)

### **How Energy Companies Use This:**

- ⚡ **Grid Operators**: Forecast load to balance supply/demand
- 🏭 **Power Plants**: Optimize generation scheduling
- 💰 **Energy Traders**: Predict market prices
- 🌱 **Renewable Integration**: Plan solar/wind backup capacity
- 📊 **Demand Response**: Identify peak demand hours
- 🔋 **Battery Storage**: Optimize charge/discharge cycles

---

## 📈 **MODEL PERFORMANCE** 📈

### **Evaluation Metrics:**

| **Metric** | **Value** | **Interpretation** |
|-----------|---------|-------------------|
| **MAE** | ~500 MW | Average error of 500 megawatts |
| **RMSE** | ~700 MW | Low error for large-scale forecasting |
| **R² Score** | 0.95+ | Excellent predictive power |
| **CV Score** | Consistent | Robust across time folds |

### **Sample Predictions:**

| **Time Period** | **Actual (MW)** | **Predicted (MW)** | **Error** |
|----------------|----------------|-------------------|----------|
| Summer Peak (3 PM) | 18,500 | 18,200 | -300 MW |
| Winter Morning (6 AM) | 14,000 | 14,400 | +400 MW |
| Fall Afternoon (2 PM) | 16,000 | 15,800 | -200 MW |

*Sample data - actual results vary by region and time*

---

## 📚 **SKILLS DEMONSTRATED** 📚

- ✅ **Time-Series Analysis**: Feature extraction from temporal data
- ✅ **Supervised Learning**: Random Forest Regression
- ✅ **Hyperparameter Tuning**: GridSearchCV optimization
- ✅ **Feature Engineering**: Lag features, seasonal decomposition
- ✅ **Model Evaluation**: MAE, RMSE, R², cross-validation
- ✅ **Data Visualization**: Plotly interactive charts
- ✅ **Web Development**: Streamlit app with custom CSS
- ✅ **Python**: Pandas, NumPy, Scikit-learn
- ✅ **Data Handling**: Parquet, CSV processing
- ✅ **Deployment**: Production-ready web app

---

## 🔮 **FUTURE ENHANCEMENTS** 🔮

- [ ] Add LSTM/GRU models for deep learning comparison
- [ ] Implement weather data integration (temperature, humidity)
- [ ] Add SHAP values for explainable AI
- [ ] Create real-time API endpoint
- [ ] Implement anomaly detection (power outages)
- [ ] Add forecasting horizons (next 24h, next week)
- [ ] Build mobile app version (React Native)
- [ ] Add ensemble models (XGBoost, LightGBM)

---

## 🤝 **CONTRIBUTING** 🤝

![Contributing](https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif)

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE** 📝

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME** 👨‍💻

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank%20Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

**Mayank Goyal**  
📊 Data Analyst | 🤖 ML Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

---

## ⭐ **SHOW YOUR SUPPORT** ⭐

![Support](https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif)

Give a ⭐️ if this project helped you understand energy demand forecasting!

### ⚡ **Built with Data & ❤️ by Mayank Goyal** ⚡

**"Turning energy data into smart grid intelligence, one prediction at a time!"** 📊

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)
