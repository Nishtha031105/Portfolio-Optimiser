#Author: Nishtha Shah
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data['Close']  # Use 'Close' price for optimization

def calculate_daily_returns(stock_data):
    returns = stock_data.pct_change()  # Calculate daily returns (percentage change)
    return returns.dropna()  # Remove NaN values from the first row

def calculate_statistics(daily_returns):
    mean_returns = daily_returns.mean()  # Mean of daily returns
    cov_matrix = daily_returns.cov()  # Covariance matrix of daily returns
    return mean_returns, cov_matrix

def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)  # Weighted sum of returns

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Standard deviation of returns

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets  # Equal initial weights

    # Objective function to minimize: portfolio volatility
    def objective(weights):
        return portfolio_volatility(weights, cov_matrix)

    # Constraints: sum of weights must equal 1 (fully invested portfolio)
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]  # Weights between 0 and 1

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)

    return result.x  # Optimal weights


# Calculate Cumulative Returns for Stocks and Portfolio
def calculate_cumulative_returns(stock_data, daily_returns, optimal_weights):
    # Cumulative returns for each stock
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # Cumulative returns for the portfolio
    portfolio_returns = (daily_returns * optimal_weights).sum(axis=1)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return cumulative_returns, portfolio_cumulative_returns

# Visualize Cumulative Returns
def visualize_performance(stock_data, cumulative_returns, portfolio_cumulative_returns):
    plt.figure(figsize=(10, 6))

    # Plot cumulative returns for each stock
    for ticker in stock_data.columns:
        plt.plot(cumulative_returns[ticker], label=f"{ticker} Cumulative Return")
    
    # Plot cumulative returns for the optimized portfolio
    plt.plot(portfolio_cumulative_returns, label="Optimized Portfolio", color='black', linestyle='--')

    # Adding title and labels
    plt.title("Cumulative Returns of Stocks vs. Optimized Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_sharpe_ratio(optimal_weights, mean_returns, cov_matrix):
    portfolio_returns = portfolio_return(optimal_weights, mean_returns)
    portfolio_volatilitys = portfolio_volatility(optimal_weights, cov_matrix)
    sharpe_ratio = portfolio_returns/ portfolio_volatilitys
    return sharpe_ratio

def efficient_frontier(mean_returns, cov_matrix, num_portfolios=1000):
    results = np.zeros((3, num_portfolios))  # Store [returns, volatility, Sharpe Ratio]
    
    for i in range(num_portfolios):
        # Random portfolio weights
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Ensure the sum of weights is 1
        
        # Calculate portfolio return and volatility
        portfolio_ret = portfolio_return(weights, mean_returns)
        portfolio_vol = portfolio_volatility(weights, cov_matrix)
        
        # Store the results
        results[0, i] = portfolio_ret
        results[1, i] = portfolio_vol
        results[2, i] = portfolio_ret / portfolio_vol  # Sharpe Ratio
    
    return results


def plot_efficient_frontier(results):
    plt.figure(figsize=(10, 6))
    
    # Plot the efficient frontier
    plt.scatter(results[1], results[0], c=results[2], cmap='YlGnBu', marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Risk)')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.grid(True)
    plt.show()


# Main Execution
if __name__ == "__main__":

    # tickers = ['AAPL', 'GOOG', 'MSFT']  # Example tickers
    tickers = input("Enter stock tickers separated by space: ").split()
    #start_date = '2023-01-01'
    start_date=input("enter Start date in 'YYYY-MM-DD' format: ")
    #end_date = '2023-12-31'
    end_date=input("enter Start date in 'YYYY-MM-DD' format: ")

    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    daily_returns = calculate_daily_returns(stock_data)
    
    mean_returns, cov_matrix = calculate_statistics(daily_returns)
    
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)

    opt_portfolio_return = portfolio_return(optimal_weights, mean_returns)
    opt_portfolio_volatility = portfolio_volatility(optimal_weights, cov_matrix)

    cumulative_returns, portfolio_cumulative_returns = calculate_cumulative_returns(stock_data, daily_returns, optimal_weights)
    visualize_performance(stock_data, cumulative_returns, portfolio_cumulative_returns)

    sharpe_ratio = calculate_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)
    print(f"Optimized Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")

    results = efficient_frontier(mean_returns, cov_matrix)

    # Normalize weights to 100%
    final_weights = {tickers[i]: round(optimal_weights[i] * 100, 2) for i in range(len(tickers))}

    print("\nPortfolio Allocation (out of 100%):")
    for stock, weight in final_weights.items():
        print(f"{stock}: {weight}%")

    plot_efficient_frontier(results)

    plt.figure(figsize=(8, 8))
    plt.pie(final_weights.values(), labels=final_weights.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Portfolio Allocation')
    plt.show()