 **CardioGuard AI**

**Heart Disease Prediction System** — A modern healthcare dashboard powered by Machine Learning.

## Features

- **Patient Risk Assessment** — Enter health parameters and get instant heart disease risk prediction
- **Doctor Dashboard** — View all patients with color-coded risk indicators
- **Model Analytics** — Accuracy metrics, confusion matrix, and feature importance
- **AI Chatbot** — Heart health guidance with medical disclaimers

## Tech Stack

- **Frontend:** Streamlit with glassmorphism UI
- **ML Model:** RandomForestClassifier (scikit-learn)
- **Data:** UCI Heart Disease dataset (auto-generates synthetic data if unavailable)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

## Screenshots
<img width="1906" height="786" alt="image" src="https://github.com/user-attachments/assets/245f8615-d66d-4fd0-b04d-cd082bc7d149" />


| Patient View | Doctor Dashboard |
|:---:|:---:|
| Risk assessment with probability | Color-coded patient cards |

| Analytics | Chatbot |
|:---:|:---:|
| Confusion matrix & feature importance | Chat bubble interface |

