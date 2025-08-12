import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# LOADING DATA AND CONFIGURATIONS
st.set_page_config("Sales Forecasting App", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# SIDEBAR
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["📈 Forecast", "📉 Visualize Trends"])

# STORE AND FAMILY OPTIONS
store_options = sorted(df['store_nbr'].unique())
family_options = sorted(df['family'].dropna().unique().astype(str))

# FORECASTING PAGE
if page == "📈 Forecast":
    st.title("🔮 Sales Forecasting")

    store = st.selectbox("🏪 Store Number", store_options)
    family = st.selectbox("📂 Product Family", family_options)
    forecast_type = st.radio("📅 Forecast", ["Single Day", "Multiple Periods"])

    if forecast_type == "Single Day":
        date = st.date_input("Date", datetime.today())
        promo = st.number_input("🏷️ Promotions", 0, step=1)

        input_df = pd.DataFrame([{
            'store_nbr': store,
            'family': family,
            'onpromotion': promo,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'Weekday': date.weekday()
        }])

        try:
            input_df['family'] = label_encoder.transform(input_df['family'])
            processed = preprocessor.transform(input_df)
            pred = max(0, model.predict(processed)[0])
            if st.button("📊 Forecast"):
                st.success(f"📈 Predicted Sales: **{round(pred, 2)}**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    else:
        period = st.selectbox("🔁 Forecast by", ["Days", "Weeks", "Months"])
        length = st.number_input(f"How many {period.lower()}?", 1, 90, 7)
        start = st.date_input("📆 Start Date", datetime.today())
        promo = st.number_input("🏷️ Avg. Promotions/day", 0, step=1)

        # Date Generation
        dates = []
        for i in range(int(length)):
            if period == "Days":
                d = start + timedelta(days=i)
            elif period == "Weeks":
                d = start + timedelta(weeks=i)
            else:
                d = start + pd.DateOffset(months=i)
            dates.append(d)

        future_df = pd.DataFrame([{
            'store_nbr': store,
            'family': family,
            'onpromotion': promo,
            'Year': d.year,
            'Month': d.month,
            'Day': d.day,
            'Weekday': d.weekday()
        } for d in dates])

        try:
            future_df['family'] = label_encoder.transform(future_df['family'])
            processed = preprocessor.transform(future_df)
            preds = model.predict(processed)
            future_df['Predicted_Sales'] = [max(0, p) for p in preds]
            future_df['Date'] = pd.to_datetime(future_df[['Year', 'Month', 'Day']])

            st.dataframe(future_df[['Date', 'Predicted_Sales']])
            st.line_chart(future_df.set_index("Date")['Predicted_Sales'])

            csv = future_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# VISUALIZING TRENDS 
else:
    st.title("📉 Historical Trends")
    store = st.selectbox("🏪 Store", store_options, key="v_store")
    family = st.selectbox("📂 Family", family_options, key="v_family")

    df_filtered = df[(df['store_nbr'] == store) & (df['family'] == family)].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])

    if df_filtered.empty:
        st.warning("⚠️ No data found.")
    else:
        df_filtered.sort_values("date", inplace=True)
        st.line_chart(df_filtered.set_index("date")["sales"])

        # OPTION TO DOWNLOAD FORECASTED DATA
        csv = df_filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download History", csv, "historical_data.csv", "text/csv")
