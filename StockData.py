import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib for Linux
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint #NOTE: Statsmodels 0.14.1, Scipy 1.12.0 for compatibility
from pr_scraper import getpe
import asyncio






def GetData(ticker1, ticker2, period):
    data = yf.download([ticker1, ticker2], period=f"{period}y", auto_adjust = False).dropna()["Adj Close"]
    return data


class Features:
    def __init__(self, ticker1, ticker2, window, df, period):
        self.period = period
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.window = window
        self.df = df
        self.alpha = None
        self.beta = None
        self.price_a = self.df[self.ticker1]
        self.price_b = self.df[self.ticker2]
        self.predicted = None
        self.residual = None
        self.z_score = None
        self.rolling_mse = None
        self.sma_residual = None
        self.upper_band = None
        self.lower_band = None
        self.volatility_a = None
        self.volatility_b = None
        self.corr_coeffecient = None
        self.rolling_corr = None
        self.coint_p = None
        self.adf_p = None
        self.t1pe = None
        self.t2pe = None

    def run_regression(self):
        regression_input = sm.add_constant(self.price_b)
        model = sm.OLS(self.price_a, regression_input)
        results = model.fit()
        self.alpha = results.params['const']
        self.beta = results.params[self.ticker2]
        #print(f"Hedge Ratio (Beta): {beta:.4f}")
        # print(f"Hedge Ratio (Alpha): {alpha:.4f}")
        self.predicted = self.alpha + self.beta * self.price_b
        self.residual = self.price_a - self.predicted

    def calc_rolling_mse(self):
        self.rolling_mse = (self.residual ** 2).rolling(self.window).mean()
    def calc_z_score(self):
        std = self.residual.std()
        #print(f"Standard Deviation of Residual: {std:.4f}")
        self.z_score = (self.residual - self.residual.mean()) / self.residual.std()
    def calc_bollinger_bands(self):
        self.sma_residual = self.residual.rolling(self.window).mean()
        rolling_residual_std = self.residual.rolling(self.window).std()
        self.upper_band = self.sma_residual + (2 * rolling_residual_std)
        self.lower_band = self.sma_residual - (2 * rolling_residual_std)
    def calc_volatility(self):
        returns_a = self.df[self.ticker1].pct_change()
        returns_b = self.df[self.ticker2].pct_change()
        self.volatility_a = returns_a.rolling(self.window).std() #* np.sqrt(window)  NOTE: Scaled volatility
        self.volatility_b = returns_b.rolling(self.window).std() # np.sqrt(window)
    def calc_correlation(self):
        correlation_matrix = self.df.corr()
        self.corr_coeffecient = correlation_matrix.iloc[0, 1]
        self.rolling_corr = self.df[self.ticker1].rolling(self.window).corr(self.df[self.ticker2])
    def p_test(self):
        self.coint_p = coint(self.price_a, self.price_b)[1]
        self.adf_p = adfuller(self.residual)[1]
        #print(f"Cointegration p-value: {self.coint_p:.4f}")
        #print(f"ADF Test p-value: {self.adf_p:.4f}")
        return self.coint_p, self.adf_p
    def calc_pe(self):
        self.t1pe, self.t2pe = asyncio.run(getpe(self.ticker1, self.ticker2, self.period))
        # Get PE
    
#####################################################################################
    def plot_adj_close(self):
        plt.figure(figsize = (16,8))
        plt.plot(self.price_a, label=f'{self.ticker1}', color = 'red')
        plt.plot(self.price_b, label=f'{self.ticker2}', color = 'blue')
        plt.autoscale(enable = True, axis = 'both')
        plt.title(f'Closing Daily Price')
        plt.legend()
        plt.show()


        

    def plot_mse_predicted_price(self):
        plt.figure(figsize=(16,8))
        plt.plot(self.rolling_mse, label='Rolling MSE', color='orange')
        plt.plot(self.price_a, label=self.ticker1, color='blue', alpha=0.5)
        plt.plot(self.predicted, label=f'Predicted {self.ticker1}', color='red', alpha=0.5)
        plt.legend()
        plt.title(f'Rolling MSE of {self.ticker1} vs {self.ticker2}')
        plt.xlabel('Date')
        plt.ylabel('Mean Squared Residual')
        plt.show()
    def plot_z_score(self):
        plt.figure(figsize=(16,8))
        plt.plot(self.z_score, label='Z-Score', color='purple')
        plt.axhline(2, color='red', linestyle='--', label='Upper Threshold (2)')
        plt.axhline(-2, color='green', linestyle='--', label='Lower Threshold (-2)')
        plt.title(f'Z-Score of {self.ticker1} vs {self.ticker2}')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.legend()
        plt.show()
    def plot_distribution(self):    
        z_array = np.array(self.z_score)
        weights = np.ones_like(z_array) / len(z_array) * 100
        num_bins = int(np.sqrt(len(z_array)))
        plt.hist(self.z_score, bins=num_bins, weights=weights, color='lightblue', edgecolor='black')
        plt.title(f'Z-Score Histogram of {self.ticker1} vs {self.ticker2}')
        plt.xlabel('Z-Score')
        plt.ylabel('Percentage of Data')
        plt.xlim(-3,3)
        plt.show()
    def plot_residual_bollinger_bands(self):
        plt.figure(figsize=(16,8))
        plt.fill_between(self.sma_residual.index, self.upper_band, self.lower_band, color='lightgray', alpha=0.4, label='Bollinger Band')
        plt.plot(self.residual, label='Residual', color='blue')
        plt.plot(self.sma_residual, label='SMA Residual', color='orange')
        #plt.plot(self.upper_band, label='Upper Band', color='red', linestyle='--')
        #plt.plot(self.lower_band, label='Lower Band', color='green', linestyle='--')
        plt.title(f'Residual Bollinger Bands of {self.ticker1} vs {self.ticker2}')
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_volatility(self):
        plt.figure(figsize=(16,8))
        plt.plot(self.volatility_a, label=f'{self.ticker1} Volatility', color='blue')
        plt.plot(self.volatility_b, label=f'{self.ticker2} Volatility', color='orange')
        plt.title(f'Rolling {self.window}-Day Volatility of {self.ticker1} and {self.ticker2}')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_correlation(self):
        plt.figure(figsize=(16,8))
        plt.axhline(self.corr_coeffecient, color='red', linestyle='--', label='Correlation Coefficient')
        plt.plot(self.rolling_corr, label = f'Rolling {self.window}-Day Correlation')
        plt.title(f'{self.window}-Day Rolling Correlation between {self.ticker1} and {self.ticker2}')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.ylim(-1, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_pe(self):
        plt.figure(figsize = (16,8))
        plt.plot(self.t1pe, color = 'red', label = f'{self.ticker1} PE Ratio')
        plt.plot(self.t2pe, color = 'blue', label = f'{self.ticker2} PE Ratio')
        plt.title("Daily PE Ratio")
        plt.xlabel('Date')
        plt.ylabel('PE Ratio')
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.grid(True)
        plt.legend()
        plt.tight_layout
        plt.show()



####################################################################################
    def calc_all(self, pe = True):
        if pe:
            self.calc_pe()
        self.calc_correlation()
        self.calc_rolling_mse()
        self.calc_z_score()
        self.calc_bollinger_bands()
        self.calc_volatility()
    def plot_all(self, pe=True):
        self.plot_adj_close()
        self.plot_correlation()
        self.plot_mse_predicted_price()
        self.plot_z_score()
        self.plot_distribution()
        self.plot_residual_bollinger_bands()
        self.plot_volatility()
        if pe:
            if self.t1pe is not None and self.t2pe is not None and len(self.t1pe) > 0 and len(self.t2pe) > 0:
                self.plot_pe()
            else:
                print("PE data not available or empty.")
            return
    def run_all(self, pe=True):
        if pe:
            self.calc_all()
            self.plot_all()
        else:
            self.calc_all(pe=False)
            self.plot_all(pe=False)



def main():
    ticker1 = input("Enter First Ticker: ").strip().upper()
    ticker2 = input("Enter Second Ticker: ").strip().upper()
    period = input("Enter Period of years (1 or 5): ").strip()
    pe = input("Do you want to run PE analysis? (y/n): ").strip().lower()

    window = 14
    df = GetData(ticker1, ticker2, period)
    features = Features(ticker1, ticker2, window, df, period)

    features.run_regression()
    p_values = features.p_test()

    print(f"p-values: (coint) {p_values[0]:.4f}, (ADF) {p_values[1]:.4f})")
    if p_values[0] > 0.05 or p_values[1] > 0.05:
        print(f"Warning: {ticker1} and {ticker2} may not be cointegrated (p-values:  (coint) {p_values[0]:.4f}, (ADF) {p_values[1]:.4f})")
        if input("Would you like to run the full analysis? (y/n)").strip().lower() != 'y':
            print("Exiting without further analysis.")
            return
    run_pe = (pe == 'y')
    features.run_all(pe=run_pe)


if __name__ == "__main__":
    main()

