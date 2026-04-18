# Heart Disease Detection System

This project is a hybrid clinical diagnostic dashboard that combines a rule-based expert system and a machine learning model to assess patient heart disease risk.

## Project Structure

- `data/`: Raw and preprocessed clinical datasets.
- `notebooks/`: Exploratory Data Analysis and Model Training experiments.
- `rule_based_system/`: Experta logic for medical clinical heuristics.
- `ml_model/`: Scikit-learn Decision Tree implementation and serialized models.
- `utils/`: Data cleaning and transformation utilities.
- `reports/`: Comparative performance metrics between Logic vs ML models.
- `ui/`: Streamlit-based diagnostic dashboard.

## Setup and Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing utility (optional if data is already present):
   ```bash
   cd utils && python data_processing.py
   ```
4. Train the ML model:
   ```bash
   cd ml_model && python train_model.py
   ```
5. Launch the Dashboard:
   ```bash
   streamlit run ui/app.py
   ```

## Technology Stack

- **Python 3.10+**
- **Experta**: For the rule-based inference engine.
- **Scikit-Learn**: For the Decision Tree classifier.
- **Streamlit & Plotly**: For the interactive clinical UI.
- **Pandas/NumPy**: For high-performance data operations.

## Project Team & Responsibilities

- **Mina**: Data Preprocessing & Visualization Orchestration
- **Nance**: Knowledge Engineering & Experta Logic Implementation
- **Marsel**: Machine Learning Architecture & Performance Comparison
