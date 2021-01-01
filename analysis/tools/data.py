import pandas as pd
import yfinance as yf
import iexfinance.stocks as stocks
import os


TOKEN = 'pk_0ba3095274d146efaaa32c3536b75320'
PROJECT_PATH = 'C:\\Users\\shiya\\PycharmProjects\\h1b-filing-stock-price-analysis'
DATA_PATH = 'C:\\Users\\shiya\\PycharmProjects\\h1b-filing-stock-price-analysis\\data'


def get_h1b(search_terms):
    try:
        filings = pd.read_pickle(os.path.join(DATA_PATH, "h1b_filings.p"))
    except FileNotFoundError as e:
        tables = []
        for term in search_terms[:]:
            print(term, end="")
            for year in range(2012, 2021):
                url = "https://h1bdata.info/index.php?em=&job={}&city=&year={}"
                table = pd.read_html(url.format(term, year))[0]
                tables.append(table)
                print(".", end="")

        filings = pd.concat(tables)
        filings.to_pickle(os.path.join(DATA_PATH, "h1b_filings.p"))
        filings = filings.dropna()
        filings = filings[filings["CASE STATUS"] == "CERTIFIED"]
    return filings


def _get_info(tickers):
    return stocks.Stock(tickers, token=TOKEN).get_company()


def _process_info(info):
    info["industry"] = info["industry"].str[:80] + "..."
    info["industry"] = info["industry"].replace("...", "UNKNOWN")
    return info


def get_sp500_info():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = table["Symbol"].tolist()
    try:
        info = pd.read_pickle(os.path.join(DATA_PATH, "sp500_info.p"))
    except FileNotFoundError as e:
        info = pd.concat(
            [_get_info(tickers[i : i + 100]) for i in range(0, len(tickers), 100)]
        )
        info = _process_info(info)
        info.to_pickle(os.path.join(DATA_PATH, "sp500_info.p"))
    return info


def get_prices(tickers, start, end):
    try:
        prices = pd.read_pickle(os.path.join(DATA_PATH, "sp500_prices.p"))
    except FileNotFoundError as e:
        data = {}
        for raw_ticker in tickers:
            ticker = raw_ticker.split(".")[0]
            ticker_object = yf.Ticker(ticker)
            data[ticker] = ticker_object.history(start=start, end=end)["Close"]
        prices = pd.DataFrame(data)
        prices.to_pickle(os.path.join(DATA_PATH, "sp500_prices.p"))
    return prices
