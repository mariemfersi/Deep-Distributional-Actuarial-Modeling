# Deep Distributional Actuarial Modeling

**Uncertainty-Aware Pricing, Reserving and Fraud Detection**

A production-ready actuarial platform combining classical GLM methods with modern deep learning, featuring guaranteed confidence intervals via conformal prediction and SHAP explainability.

## 🎯 Key Results

| Module | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| **Pricing** (Gini) | GLM Poisson | CANN Interaction | **+7%** (in notebook) |
| **Reserving** (Coverage) | Mack Chain-Ladder | Mack + Conformal | **74.4% → 91.9%** |
| **Fraud** (AUC-ROC) | Isolation Forest | Random Forest | **0.815** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Pricing  │  │Reserving │  │  Fraud   │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
└───────┼─────────────┼─────────────┼────────────────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     Backend (FastAPI)                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Pricing Service                                      │  │
│  │  • GLM Poisson (baseline)                            │  │
│  │  • CANN Interaction Model (VehPower/VehAge/VehGas)   │  │
│  │  • NGBoost Severity (distributional)                 │  │
│  │  • Gaussian Copula (frequency-severity dependence)   │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Reserving Service                                    │  │
│  │  • Mack Chain-Ladder (stochastic baseline)            │  │
│  │  • Conformal Prediction (guaranteed intervals)       │  │
│  │  • Deep Triangle GRU (sequential modeling)           │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Fraud Service                                       │  │
│  │  • Random Forest (supervised, AUC-ROC 0.815)         │  │
│  │  • Default values for 30 features                   │  │
│  │  • Graph construction (for future GNN extension)      │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Explainability Service                              │  │
│  │  • SHAP values for pricing predictions               │  │
│  │  • SHAP values for fraud predictions                 │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Modules

### 1. Pricing (Tarification)
- **GLM Poisson** for frequency with exposure offset (baseline)
- **CANN Interaction Model** targeting VehPower/VehAge/VehGas/VehBrand interactions
  - **-1.61% deviance** improvement over GLM
  - Skip connection from GLM preserves interpretability
- **NGBoost Gamma** for distributional severity with confidence intervals
- **Gaussian Copula** for frequency-severity dependence modeling

### 2. Reserving (Provisionnement)
- **Mack Chain-Ladder** stochastic baseline with standard errors
- **Conformal Prediction** calibration for guaranteed coverage
  - Empirical coverage: **74.4% (Mack) → 91.9% (Conformal)**
  - No distributional assumptions required
- **Deep Triangle GRU** for sequential modeling of payment patterns

### 3. Fraud Detection (Détection de fraude)
- **Random Forest** supervised model (AUC-ROC 0.815)
- **30 features** with categorical encoding and numerical normalization
- **Default values** strategy for simplified frontend input
- **Graph construction** functions for future GNN extension

### 4. Explainability (Explicabilité)
- **SHAP values** for pricing predictions (GLM + CANN)
- **SHAP values** for fraud predictions (Random Forest)
- Feature importance visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL (optional, for production)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/projet_actuariat.git
cd projet_actuariat

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Running the Application

```bash
# Start backend (from backend directory)
python -m uvicorn app.main:app --reload --port 8000

# Start frontend (from frontend directory, in a new terminal)
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
projet_actuariat/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── routers/        # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── schemas/        # Pydantic models
│   │   └── main.py         # FastAPI app
│   └── requirements.txt
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   └── lib/               # API client
├── src/                   # Core ML models
│   ├── pricing/           # Pricing models (GLM, CANN, NGBoost)
│   ├── reserving/         # Reserving models (Mack, Deep Triangle)
│   ├── fraud/             # Fraud models (Random Forest, Graph)
│   └── common/            # Shared utilities
├── data/                  # Raw datasets
├── models/                # Trained model files
├── notebooks/             # Jupyter notebooks for training
└── config/                # Configuration files
```

## 🧪 Model Training

### Pricing Models
```bash
# Train GLM and CANN models
jupyter notebooks/03b_pricing_cann_final.ipynb
```

### Reserving Models
```bash
# Train Deep Triangle model
jupyter notebooks/04_reserving_mack.ipynb
```

### Fraud Models
```bash
# Train Random Forest model
jupyter notebooks/05_fraud.ipynb
```

## 📈 API Endpoints

### Pricing
- `POST /pricing/predict` - Predict pure premium with CANN interaction model
- `POST /explain/pricing` - Get SHAP values for pricing prediction

### Reserving
- `POST /reserving/ibnr` - Predict IBNR with Mack + Conformal intervals
- `POST /reserving/predict` - Predict future increments with Deep Triangle

### Fraud
- `POST /fraud/predict` - Predict fraud probability with Random Forest
- `POST /explain/fraud` - Get SHAP values for fraud prediction

## 🔬 Technical Stack

**Machine Learning:**
- PyTorch (CANN, Deep Triangle)
- NGBoost (distributional regression)
- scikit-learn (Random Forest)
- statsmodels (GLM)
- chainladder (Mack Chain-Ladder)
- SHAP (explainability)

**Backend:**
- FastAPI
- Pydantic
- joblib (model persistence)

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- Recharts (visualization)

**MLOps:**
- MLflow (experiment tracking - setup pending)
- Docker (containerization - setup pending)

## 📚 Datasets

- **Pricing**: freMTPL2 (CASdatasets) - French auto insurance portfolio
- **Reserving**: CAS Loss Reserving Database - Real claims triangles
- **Fraude**: Kaggle Auto Insurance Fraud - Fraud detection dataset

## 🎓 Methodology

This project follows a rigorous comparative approach:

1. **Baseline First**: Always establish a classical actuarial baseline (GLM, Mack Chain-Ladder)
2. **ML Enhancement**: Add deep learning only where it provides measurable improvement
3. **Empirical Validation**: Verify interval coverage empirically, not just theoretically
4. **Explainability**: Every model prediction must be explainable via SHAP
5. **Honest Reporting**: Document failures (e.g., GNN approaches that didn't work)

## 🤝 Contributing

This is a portfolio project demonstrating actuarial ML integration. For questions or suggestions, please open an issue.

## 📝 License

MIT License - feel free to use this project for learning and reference.

## 👤 Author

**Actuarial Data Science Portfolio**

This project demonstrates the integration of classical actuarial methods with modern machine learning, with a focus on uncertainty quantification and explainability.

---

*Built for the intersection of actuarial science and machine learning.*
