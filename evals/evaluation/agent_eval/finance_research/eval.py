import os
import argparse
import json
import pandas as pd
import time

API_KEY=os.getenv('API_KEY', "token-abc123")
DOW30_ticker =[
    "AMZN",
    "AXP",
    "AAPL",
    "AMGN",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "SHW",
    "TRV",
    "UNH",
    "CRM",
    "NVDA",
    "V",
    "VZ",
    "WMT",
    "DIS",
]

def get_stock_price(ticker, date):
    from langchain_community.tools.polygon.aggregates import PolygonAggregates
    from langchain_community.utilities.polygon import PolygonAPIWrapper

    api_wrapper = PolygonAPIWrapper()
    aggregate_tool = PolygonAggregates(api_wrapper=api_wrapper)

    # We can invoke directly with input
    res = aggregate_tool.invoke(
        {
            "ticker": ticker,
            "timespan": "day",
            "timespan_multiplier": 1,
            "from_date": date,
            "to_date": date,
        }
    )
    print(res)
    try:
        res = json.loads(res)
        closing_price = res[0]["c"]
    except:
        closing_price = "N/A"
    print(f"Closing price of {ticker} on {date} is {closing_price}")
    time.sleep(15)
    return closing_price
    

def generate_answer(args, prompt):
    """
    Use vllm endpoint to generate the answer
    """
    from openai import OpenAI
    # send request to vllm endpoint
    client = OpenAI(
        base_url=f"{args.llm_endpoint_url}/v1",
        api_key=API_KEY,
    )

    params = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        **params
        )

    # get response
    response = completion.choices[0].message.content

    return response

INVESTOR_PROMPT ="""\
You are a professional trader. 
Your financial analyst has made a research report on {company} using the data up to today, March 6, 2025. 

==== Research Report ====
{report}
==== End of Report ====

Read the report carefully and make projections for:
1. Whether the stock price of {company} will be up or down tomorrow, March 7, 2025.
2. The closing stock price of {company} today, March 6, 2025 was ${price}. What will be the closing stock price of {company} tomorrow, March 7, 2025?

You should read the report carefully and make your projections based on only the information in the report. Think carefully before making your projections. 
Give your final projections in the following json format at the end:
```json
{{
  "stock_price_tomorrow": "up",
  "closing_stock_price_tomorrow": 100.5
}}
```
"""

def parse_answer(answer):
    if "```json" in answer:
        answer = answer.split("```json")[-1].split("```")[0]
    elif "```" in answer:
        answer = answer.split("```")[-1].split("```")[0]
    try:
        answer = json.loads(answer)
    except:
        answer = "parsing error"
    return answer

def get_predictions(answer):
    stock_price_tomorrow = answer.get("stock_price_tomorrow", None)
    closing_stock_price_tomorrow = answer.get("closing_stock_price_tomorrow", None)
    return stock_price_tomorrow, closing_stock_price_tomorrow

def calculate_accuracy(pred, true):
    if pred == true:
        return 1
    else:
        return 0
    
def calculate_mae(pred, true):
    return abs(pred - true)

def get_report(company):
    # TODO
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8888")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--get_stock_price", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    args = parser.parse_args()
    
    if args.get_stock_price:
        previous_day_price = []
        today_price = []
        for ticker in DOW30_ticker:
            print("ticker:", ticker)
            pp = get_stock_price(ticker, "2025-03-06")
            tp = get_stock_price(ticker, "2025-03-07")
            previous_day_price.append(pp)
            today_price.append(tp)
            print("="*50)
        
        df = pd.DataFrame({
            "ticker": DOW30_ticker,
            "2025-03-06-close-price": previous_day_price,
            "2025-03-07-close-price": today_price
        })
        df.to_csv("stock_price.csv", index=False)
        print(df.head(5))
    else:
        # load stock price data
        df = pd.read_csv("stock_price.csv")
    
    if args.run_eval:
        # use LLM to predict stock price
        # and calculate prediction accuracy and MAE
        pred_acc = []
        mae = []
        for i, row in df.iterrows():
            company = row["ticker"]
            price = row["2025-03-06-close-price"]
            report = get_report(company)
            prompt = INVESTOR_PROMPT.format(company=company, price=price, report=report)
            answer = generate_answer(args, prompt)
            answer = parse_answer(answer)
            print(f"Prediction for {company}: {answer}")

            pred_stock_price_tomorrow, pred_closing_stock_price_tomorrow = get_predictions(answer)
            true_stock_price_tomorrow = "up" if row["2025-03-07-close-price"] > row["2025-03-06-close-price"] else "down"
            true_closing_stock_price_tomorrow = row["2025-03-07-close-price"]
            pred_acc.append(calculate_accuracy(pred_stock_price_tomorrow, true_stock_price_tomorrow))
            mae.append(calculate_mae(pred_closing_stock_price_tomorrow, true_closing_stock_price_tomorrow))
            print("="*50)

        print(f"Prediction accuracy: {sum(pred_acc)/len(pred_acc)}")
        print(f"Mean Absolute Error: {sum(mae)/len(mae)}")
