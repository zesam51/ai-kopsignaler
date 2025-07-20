import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="AI Köpsignaler Sverige")
st.title("📊 AI‑köpsignaler – Sveriges största aktier")

TICKERS = [
    'AZN.ST','ATCO-A.ST','ATCO-B.ST','INVE-B.ST','VOLV-B.ST','ERIC-B.ST',
    'ABB.ST','SHB-A.ST','SEB-A.ST','ESSITY-B.ST','NDA.ST','STERV.ST',
    'HM-B.ST','EPIROC-B.ST','ASSA-B.ST','SWED-A.ST','SAND.ST','TELIA.ST',
    'NIBE-B.ST','TEL2-B.ST'
]

results = []
for ticker in TICKERS:
    try:
        df = yf.download(ticker, start="2019-01-01", end=pd.Timestamp.today())
        if df.empty or len(df) < 200:
            raise ValueError("Otillräcklig data")
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        df = df.dropna()
        X = df[['Open','High','Low','Close','Volume']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        latest = X.tail(1)
        prediction = model.predict(latest)[0]
        signal = "✅ Köp" if prediction == 1 else "❌ Sälj"
        results.append({
            'Aktie': ticker.replace('.ST',''),
            'Signal': signal,
            'Precision (%)': round(acc*100,2)
        })
    except Exception as e:
        results.append({'Aktie': ticker.replace('.ST',''), 'Signal': '⚠️ Fel', 'Precision (%)': '–'})

st.table(pd.DataFrame(results))
