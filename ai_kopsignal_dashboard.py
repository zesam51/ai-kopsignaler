import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="AI Köpsignaler Sverige")
st.title("📊 AI‑köpsignaler – Sveriges största aktier")

# Tickrar med fallback-alternativ för tickrar med bristande data
TICKERS = {
    'AZN': ['AZN.ST'],
    'ATCO-A': ['ATCO-A.ST'],
    'ATCO-B': ['ATCO-B.ST'],
    'INVE-B': ['INVE-B.ST'],
    'VOLV-B': ['VOLV-B.ST'],
    'ERIC-B': ['ERIC-B.ST'],
    'ABB': ['ABB.ST'],
    'SHB-A': ['SHB-A.ST'],
    'SEB-A': ['SEB-A.ST'],
    'ESSITY-B': ['ESSITY-B.ST'],
    'NDA': ['NDA.ST', 'NDA-SE.ST'],
    'STERV': ['STERV.ST', 'STEAV.ST', 'STESB.ST'],
    'HM-B': ['HM-B.ST'],
    'EPIROC-B': ['EPIROC-B.ST'],
    'ASSA-B': ['ASSA-B.ST'],
    'SWED-A': ['SWED-A.ST'],
    'SAND': ['SAND.ST'],
    'TELIA': ['TELIA.ST'],
    'NIBE-B': ['NIBE-B.ST'],
    'TEL2-B': ['TEL2-B.ST']
}

results = []
felaktiga_aktier = []

for namn, ticker_list in TICKERS.items():
    df = None
    valt_ticker = None

    # Testa tickers i ordning för fallback
    for ticker in ticker_list:
        df = yf.download(ticker, start="2019-01-01", end=pd.Timestamp.today(), progress=False)
        if df is not None and not df.empty and len(df) > 200:
            valt_ticker = ticker
            break

    if valt_ticker is None:
        results.append({
            'Aktie': namn,
            'Signal': '⚠️ Ingen data',
            'Accuracy (%)': '–',
            'Precision (%)': '–',
            'Recall (%)': '–',
            'F1-score (%)': '–'
        })
        felaktiga_aktier.append(namn)
        continue

    try:
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        df = df.dropna()

        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = pd.Series(model.predict(X_test), index=y_test.index)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        latest = X.tail(1)
        prediction = model.predict(latest)[0]
        signal = "✅ Köp" if prediction == 1 else "❌ Sälj"

        results.append({
            'Aktie': namn,
            'Signal': signal,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-score (%)': round(f1 * 100, 2),
        })

        # För graf: lägg till rätta och falska köpsignaler
        df_test = X_test.copy()
        df_test['Prediction'] = y_pred
        df_test['True'] = y_test
        df_test['Close'] = df.loc[df_test.index, 'Close']

        df_test['True_Signal'] = np.where((df_test['Prediction'] == 1) & (df_test['True'] == 1), df_test['Close'], np.nan)
        df_test['False_Signal'] = np.where((df_test['Prediction'] == 1) & (df_test['True'] == 0), df_test['Close'], np.nan)

        with st.expander(f"📉 Visa graf för {namn}"):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_test.index, df_test['Close'], label='Close-pris', color='gray')
            ax.scatter(df_test.index, df_test['True_Signal'], color='green', label='✅ Rätt köpsignal', marker='^')
            ax.scatter(df_test.index, df_test['False_Signal'], color='red', label='❌ Fel köpsignal', marker='v')
            ax.set_title(f"Köpsignaler för {namn}")
            ax.set_ylabel("Pris (SEK)")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        results.append({
            'Aktie': namn,
            'Signal': '⚠️ Fel i modell',
            'Accuracy (%)': '–',
            'Precision (%
