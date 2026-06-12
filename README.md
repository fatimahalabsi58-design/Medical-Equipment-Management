# 🏥 Smart Medical Equipment Management System

> **A data-driven system that monitors hospital equipment usage patterns and predicts when devices are likely to fail — built in two versions: Excel and Python.**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![Made with Excel](https://img.shields.io/badge/Made%20with-Excel-217346?logo=microsoft-excel)](https://www.microsoft.com/en-us/microsoft-365/excel)
[![Data Science](https://img.shields.io/badge/Cisco-Introduction%20to%20Data%20Science-1BA0D7?logo=cisco)](https://www.netacad.com/)
[![Biomedical Engineering](https://img.shields.io/badge/Field-Biomedical%20Engineering-red)]()

-----

## 👩‍💻 About This Project

This project was created as a portfolio piece after completing the **Introduction to Data Science** course by Cisco Networking Academy. It combines data science skills (data collection, analysis, validation, and machine learning) with real-world biomedical engineering problems.

**The problem it solves:** Hospitals often don’t know when equipment will fail until it breaks down — causing delays in patient care and expensive emergency repairs. This system uses data and ML to predict failures *before* they happen, and visualizes equipment usage patterns to improve hospital operations.

-----

## 🗂️ Project Structure

```
Smart-Medical-Equipment-Management/
│
├── 📁 data/
│   └── equipment_data.csv          ← Synthetic dataset (200 medical devices)
│
├── 📁 Version_1_Excel/
│   └── Medical_Equipment_Dashboard.xlsx   ← Full Excel dashboard (5 sheets)
│
├── 📁 Version_2_Python/
│   ├── data_analysis.py            ← EDA, data validation, visualizations
│   ├── ml_prediction.py            ← ML models: Logistic Regression, Decision Tree, Random Forest
│   ├── app.py                      ← Interactive Streamlit web dashboard
│   └── requirements.txt
│
└── README.md
```

-----

## 📊 Version 1 — Excel Dashboard

**Best for:** Healthcare administrators, hospital operations teams, non-coders.

### Sheets Included:

|Sheet                |Description                              |
|---------------------|-----------------------------------------|
|🔵 Raw Data           |Full dataset with colour-coded status    |
|🟢 Summary Stats      |KPI cards + pivot-style breakdowns       |
|🟠 Usage Analysis     |Device usage vs idle hours with bar chart|
|🔴 Maintenance Tracker|Risk-flagged devices sorted by urgency   |
|🟡 Charts Dashboard   |Pie chart + bar chart visuals            |

### How to open:

1. Download `Version_1_Excel/Medical_Equipment_Dashboard.xlsx`
1. Open in Microsoft Excel or Google Sheets

-----

## 🐍 Version 2 — Python + Machine Learning

**Best for:** Data scientists, engineers, technical interviews.

### What it does:

#### `data_analysis.py` — Exploratory Data Analysis

- Loads and validates the dataset (missing values, ranges, duplicates)
- Generates 4 publication-quality figures:
  - Equipment status distribution (pie + bar)
  - Usage vs idle hours by device type
  - Department-level overview
  - Maintenance risk factors scatter plot

#### `ml_prediction.py` — Predictive Maintenance Model

- Trains 3 ML models: Logistic Regression, Decision Tree, Random Forest
- 5-fold cross-validation
- Evaluation: Accuracy, AUC-ROC, Confusion Matrix, Classification Report
- Feature importance visualization
- Saves the best model to `outputs/models/`

#### `app.py` — Interactive Streamlit Dashboard

- **4 pages:** Overview Dashboard, Usage Analysis, Maintenance Tracker, ML Prediction
- Live prediction: enter device parameters and get an instant maintenance risk score
- Filterable data tables and interactive charts

-----

## 🚀 How to Run (Python Version)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/Smart-Medical-Equipment-Management.git
cd Smart-Medical-Equipment-Management/Version_2_Python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data analysis (generates figures)
python data_analysis.py

# 4. Train ML models (saves model to outputs/models/)
python ml_prediction.py

# 5. Launch the web dashboard
streamlit run app.py
```

-----

## 📈 Dataset Overview

The dataset simulates **200 hospital medical devices** across 10 device types and 6 departments.

|Feature                      |Description                                     |
|-----------------------------|------------------------------------------------|
|`Device_Type`                |Ventilator, MRI, Infusion Pump, etc.            |
|`Department`                 |ICU, ER, Radiology, Surgery, etc.               |
|`Age_Years`                  |Device age in years                             |
|`Daily_Usage_Hours`          |Average hours used per day                      |
|`Temperature_C`              |Operating temperature                           |
|`Error_Count_Monthly`        |Number of errors logged per month               |
|`Days_Since_Last_Maintenance`|Days elapsed since last service                 |
|`Needs_Maintenance`          |**Target label** — 0 = OK, 1 = Needs maintenance|

-----

## 🧠 ML Results

|Model              |CV Accuracy|AUC-ROC  |
|-------------------|-----------|---------|
|Logistic Regression|~0.89      |~0.94    |
|Decision Tree      |~0.88      |~0.93    |
|**Random Forest**  |**~0.92**  |**~0.97**|

Random Forest was selected as the best model based on AUC-ROC score.

-----

## 🔑 Key Skills Demonstrated

- ✅ Data Collection & Synthetic Data Generation
- ✅ Data Validation & Quality Checks
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Visualization (Matplotlib)
- ✅ Machine Learning (Scikit-learn)
- ✅ Model Evaluation (AUC-ROC, Confusion Matrix, Cross-Validation)
- ✅ Excel Dashboard Design
- ✅ Interactive Web App (Streamlit)

-----

## 👩‍🔬 Author

**Fatimah Jamaan**  
Biomedical Engineering Student  
📜 Cisco Introduction to Data Science — Certified June 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/YOUR_USERNAME)

-----

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

-----

## 📚 Citations & References

### 🎓 Course

> Cisco Networking Academy. (2026). *Introduction to Data Science*. Cisco Systems, Inc.
> <https://www.netacad.com/courses/data-science>

### 🤖 Machine Learning Library

> Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
> <https://scikit-learn.org>

### 📊 Data Analysis & Visualization

> McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference.
> <https://pandas.pydata.org>
> 
> Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90–95.
> <https://matplotlib.org>

### 🌐 Web Dashboard

> Streamlit Inc. (2024). *Streamlit: The fastest way to build data apps*.
> <https://streamlit.io>

### 📁 Excel Processing

> OpenPyXL Contributors. (2024). *OpenPyXL: A Python library to read/write Excel files*.
> <https://openpyxl.readthedocs.io>

### 🏥 Domain Knowledge & Inspiration

> Mobley, R. K. (2002). *An Introduction to Predictive Maintenance* (2nd ed.). Butterworth-Heinemann.
> 
> World Health Organization. (2011). *Medical Equipment Maintenance Programme Overview*. WHO Press.
> <https://www.who.int/publications/i/item/medical-equipment-maintenance-programme-overview>
> 
> Ramirez-Elizondo, L., & Palensky, P. (2014). *Predictive Maintenance for Medical Devices*. IEEE Transactions on Industrial Electronics.

### 📂 Dataset

> Dataset used in this project is **synthetically generated** for educational purposes, inspired by real-world medical device monitoring parameters including temperature, runtime hours, error counts, and maintenance logs.
