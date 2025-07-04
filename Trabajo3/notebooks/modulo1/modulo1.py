# modulo1.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")


def run_module1(horizon_days: int) -> dict:
    """
    Forecast LSTM por destino. 
    Input: horizon_days (int).
    Lee Final_Updated_Expanded_UserHistory.csv y Expanded_Destinations.csv
    en el mismo directorio de este archivo.
    Output: dict con clave=ruta y valor={
        metrics: {RMSE, MAE, R2},
        forecast_df: DataFrame completo,
        fig_demand: Figura demanda sintética,
        fig_decomp: Figura descomposición
    }
    """
    base = os.path.dirname(__file__)
    history_csv = os.path.join(base, "Final_Updated_Expanded_UserHistory.csv")
    dest_csv    = os.path.join(base, "Expanded_Destinations.csv")

    # 1) Leer datos
    user_hist = pd.read_csv(history_csv, parse_dates=["VisitDate"])
    dests     = pd.read_csv(dest_csv)
    name_map  = dests.set_index("DestinationID")["Name"].to_dict()

    # 2) Pivot diario de trips
    trips = (
        user_hist
        .groupby(["VisitDate", "DestinationID"])
        .size()
        .reset_index(name="Trips")
    )
    idx = pd.date_range(trips["VisitDate"].min(),
                        trips["VisitDate"].max(),
                        freq="D")
    pivot = (
        trips
        .pivot(index="VisitDate", columns="DestinationID", values="Trips")
        .reindex(idx, fill_value=0)
    )
    pivot.index.name = "ds"
    pivot.columns = [
        f"{name_map.get(i,i)}_{i}" for i in pivot.columns
    ]

    results = {}
    seq_len = 7  # ventana de 7 días

    # 3) Por cada destino
    for ruta in pivot.columns:
        df_ts = pivot[[ruta]].reset_index().rename(columns={ruta: "y"})
        if len(df_ts) <= seq_len:
            continue  # no hay datos suficientes

        # 4) Escalar y generar secuencias X, Y
        scaler = MinMaxScaler()
        y_s = scaler.fit_transform(df_ts[["y"]])
        X, Y = [], []
        for i in range(len(y_s) - seq_len):
            X.append(y_s[i : i + seq_len])
            Y.append(y_s[i + seq_len])
        X, Y = np.array(X), np.array(Y)
        if X.shape[0] == 0:
            continue

        # 5) Definir y entrenar LSTM
        model = Sequential([
            LSTM(50, input_shape=(seq_len,1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, Y, epochs=50, batch_size=8, verbose=0)

        # 6) Forecast recursivo
        last = y_s[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        for _ in range(horizon_days):
            p = model.predict(last, verbose=0)[0]
            preds.append(p)
            last = np.roll(last, -1, axis=1)
            last[0,-1,0] = p

        # 7) Construir DataFrame histórico + pronóstico
        fut_idx = pd.date_range(
            df_ts["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D"
        )
        df_fc = pd.DataFrame({
            "ds": np.concatenate([df_ts["ds"].values, fut_idx]),
            "y": np.concatenate([
                df_ts["y"].values,
                scaler.inverse_transform(preds).flatten()
            ])
        })

        # 8) Métricas back-test (solo si longitudes coinciden)
        n = len(df_ts)
        if n >= horizon_days:
            y_true = df_ts["y"].iloc[-horizon_days:].values
            y_pred = df_fc["y"].iloc[-horizon_days : n].values
        else:
            y_true = df_ts["y"].values
            y_pred = scaler.inverse_transform(preds[:n]).flatten()

        if len(y_true)==len(y_pred) and len(y_true)>0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            ssr  = ((y_true-y_pred)**2).sum()
            sst  = ((y_true-y_true.mean())**2).sum()
            r2   = 1 - ssr/sst
            metrics = {"RMSE":rmse, "MAE":mae, "R2":r2}
        else:
            metrics = {"RMSE":np.nan, "MAE":np.nan, "R2":np.nan}

        # 9) Figura de demanda sintética
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(df_fc["ds"], df_fc["y"], label=ruta)
        ax1.set_title(f"Demanda Sintética ({ruta})")
        ax1.set_xlabel("Fecha"); ax1.set_ylabel("Viajes")
        ax1.legend(); fig1.tight_layout()

        # 10) Descomposición (interpolar faltantes)
        ser = df_fc.set_index("ds")["y"].asfreq("D")
        if ser.isna().any():
            ser = ser.interpolate().fillna(method="bfill").fillna(method="ffill")
        decomp = seasonal_decompose(ser, model="additive", period=7)
        fig2 = decomp.plot(); fig2.set_size_inches(8,6)

        # 11) Guardar en resultados
        results[ruta] = {
            "metrics":     metrics,
            "forecast_df": df_fc,
            "fig_demand":  fig1,
            "fig_decomp":  fig2
        }

    return results
