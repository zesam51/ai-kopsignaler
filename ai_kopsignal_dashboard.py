import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta

st.set_page_config(page_title="AI Köpsignaler", layout="wide")
st.title("📈 AI Köpsignaler för Svenska Aktier")

stocks = ['VOLV-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SWED-A.ST', 'HM-B.ST', 'TEL2-B.ST']
start_date = date.today() - timedelta(days=365 * 3)
end_date = date.today()

results = []

for ticker in stocks:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty or len(data) < 100:
            raise ValueError("Otillräcklig data")

        data['Return'] = data['Close'].pct_change()
        data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
        data = data.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = data[features]
        y = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=y_test.index)  # Säkerställer rätt index
        mse = mean_squared_error(y_test, y_pred_series)

        # Senaste köpsignal
        last_row = X.tail(1)
        prediction = model.predict(last_row)[0]
        signal = '✅ Köp' if prediction == 1 else '❌ Sälj'

        results.append({
            'Aktie': ticker,
            'Senaste signal': signal,
            'Träffsäkerhet (%)': round((1 - mse) * 100, 2)
        })

    except Exception as e:
        results.append({
            'Aktie': ticker,
            'Senaste signal': '⚠️ Fel',
            'Träffsäkerhet (%)': 'N/A',
        })
        st.warning(f"Fel med {ticker}: {e}")

df = pd.DataFrame(results)
st.table(df)
