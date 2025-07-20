import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import streamlit as st

TICKERS = ['VOLV-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'HM-B.ST', 'ATCO-A.ST']

def prepare_data(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2024-12-31")
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = 100 - 100 / (1 + df['Close'].pct_change().rolling(14).apply(
        lambda x: (x[x > 0].mean()) / (-x[x < 0].mean() + 1e-5)))
    df['Future'] = df['Close'].shift(-10)
    df['Target'] = (df['Future'] > df['Close'] * 1.05).astype(int)
    return df.dropna()

def get_ai_signal(df):
    features = ['SMA50', 'SMA200', 'RSI']
    X = df[features]; y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    latest = df[features].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]
    signal = "📈 KÖP" if prob > 0.6 else "⏳ VÄNTA"
    return signal, prob

st.title("📊 AI‑köpsignaler – Svenska aktier")

results = []
for ticker in TICKERS:
    with st.spinner(f"Läser {ticker}..."):
        try:
            df = prepare_data(ticker)
            signal, prob = get_ai_signal(df)
            results.append({'Aktie': ticker.replace('.ST',''),
                            'Signal': signal,
                            'Köpsannolikhet': f"{prob:.2%}"})
        except Exception as e:
            st.warning(f"Fel med {ticker}: {e}")

st.subheader("🔍 AI‑signal idag:")
st.table(pd.DataFrame(results))
