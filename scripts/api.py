def get_prices(asset,period):
    y_ob = yf.Ticker(asset)
    return y_ob.history(period=period)
     
def dune_api_results(query_num, save_csv=False, csv_path=None):
    results = dune.get_latest_result(query_num)
    df = pd.DataFrame(results.result.rows)

    if save_csv and csv_path:
        df.to_csv(csv_path, index=False)
    return df