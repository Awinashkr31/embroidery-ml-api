# Embroidery ML API

FastAPI-based Machine Learning backend for [Embroidery By Sana](https://embroiderybysana.com).

## Features
- 🤖 AI Product Recommendations (TF-IDF + Cosine Similarity)
- 💬 AI Chatbot (Groq LLaMA-3)
- 📊 Sentiment Analysis for Reviews
- 📈 Sales Forecasting (Prophet)

## Environment Variables
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
```

## Run Locally
```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
