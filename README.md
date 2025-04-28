# Project Title: Portfolio Optimizer
## Description:
This project helps in optimizing a stock portfolio by selecting the best weights (percentage allocation) for different stocks based on historical data. It uses modern portfolio theory to calculate the expected returns, volatility, and Sharpe ratio for different portfolio combinations, eventually optimizing for the best risk-return ratio.

How to use this project:
**Install Required Libraries**
Before running the code, ensure that all the required Python libraries are installed:

yfinance: For fetching stock data.

numpy: For numerical operations.

scipy: For optimization.

pandas: For handling data.

matplotlib: For visualizing portfolio allocations.

You can install these using:

pip install yfinance numpy scipy pandas matplotlib
Download and Run the Code

Clone or download the project directory.

Place all Python files inside a folder, and run portfolio_optimizer.py.

**Usage:**

The program will prompt for stock tickers (e.g., 'AAPL', 'GOOG', 'MSFT'), start and end dates for data collection, and proceed to calculate the optimal portfolio weights.

After optimization, the program will display the portfolio allocation, expected returns, volatility, and Sharpe ratio.

Sample Input: The following tickers are supported:

For US stocks: 'AAPL', 'GOOG', 'MSFT', etc.

For Indian stocks (using .NS suffix): 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', etc.

**Example Command:**

tickers = ['AAPL', 'GOOG', 'MSFT']  # Enter stock symbols you want to optimize
start_date = '2022-01-01'           # Start date for historical data
end_date = '2023-01-01'             # End date for historical data
Expected Output:

Optimal Portfolio Allocation: A pie chart displaying the percentage weight of each stock in the portfolio.

Portfolio Performance: The expected return, volatility, and Sharpe ratio will be printed.

## Test Cases:
**Test Case 1: US Stocks Portfolio**

Input:

tickers = ['AAPL', 'GOOG', 'MSFT']
start_date = '2022-01-01'
end_date = '2023-01-01'
Expected Output:
The program should calculate the optimal allocation for these three stocks and print a pie chart showing the weight of each stock.
It will also print the expected return, volatility, and Sharpe ratio for the portfolio.

**Test Case 2: Indian Stocks Portfolio**

Input:

python
Copy
Edit
tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
start_date = '2022-01-01'
end_date = '2023-01-01'
Expected Output:
A pie chart will be generated, displaying the allocation percentage for each stock in the portfolio, along with the portfolio return, volatility, and Sharpe ratio.

**Test Case 3: Portfolio with Different Dates**

Input:

python
Copy
Edit
tickers = ['AAPL', 'GOOG', 'MSFT']
start_date = '2020-01-01'
end_date = '2021-01-01'
Expected Output:
Similar to previous tests, but with data for the different period. The portfolio may have different optimal weights based on the different data range.

**Author: Nishtha Shah**