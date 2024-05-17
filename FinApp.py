'''
This is a simple stock market prediction application that uses the yfinance library
to download stock data and the scikit-learn library to perform linear regression on the data.
The application displays the stock prices of a few pre-defined symbols and predicts the future
stock prices using linear regression model (machine learning).
'''

import tkinter as tk
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class FinApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Basic Stock Market Predictions")

        self.create_widgets()
        self.update_chart()

    def create_widgets(self):
        # Area per il grafico
        self.figure = plt.Figure(figsize=(12, 10), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

    def perform_prediction(self, data, ax):
        data = data[['Close']].dropna()
        data['Date'] = np.arange(len(data))
        X = data[['Date']]
        y = data['Close']

        # Dividi i dati in training e test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modello di regressione lineare
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predizione sui dati di test
        future_days = 30
        future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        # Aggiungi le predizioni al grafico
        future_dates = pd.date_range(start=data.index[-1], periods=future_days)
        ax.plot(future_dates, future_predictions, label='Previsione', linestyle='dashed')

    def update_chart(self):
        # Simboli predefiniti dei mercati
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'INTC', 'AMD']

        self.ax.clear()
        for symbol in symbols:
            data = yf.download(symbol, period='1y')
            if not data.empty:
                self.ax.plot(data['Close'], label=symbol.upper())
                self.perform_prediction(data, self.ax)

        self.ax.set_title("Stock Market")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Close Price (USD)")
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FinApp(root)
    root.mainloop()
