import asyncio
from playwright.async_api import async_playwright
import time
from pathlib import Path
import json
import pandas as pd

def parse(data):
    if data is None:
        print("No data received from the request.")
        return None
    ts = data[0]["series"]
    df = pd.DataFrame(ts)
    keep = ["date", "PE"]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    print("Data parsed successfully.")
    return df

def save_to_pkl(df, path, ticker):
    df.to_pickle(path)
    print(f"Data for {ticker} saved to {path}")

async def WebNavigator(ticker, period):
    data = None

    async def handle_request(response):
        url = response.url
        if "timeseries" in url and "PE" in url and "frequency=d" in url:
            try:
                nonlocal data
                data = await response.json()
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from response: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False,
                                          args=["--start-maximized",
                                                "--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        await page.goto("https://www.morningstar.com", timeout=60000)
        search_input = await page.wait_for_selector("input.mds-search-field__input__mdc")
        time.sleep(1.5)
        await search_input.click()
        await search_input.fill(ticker)
        time.sleep(3)
        await page.keyboard.press("Enter")
        await page.wait_for_function(f"() => document.title.includes('{ticker}')", timeout=15000)

        current_url = page.url
        if "quote" in current_url:
            print(f"[{ticker}] Navigating to chart page...")
            chart_url = current_url.replace("quote", "chart")
            await page.goto(chart_url)

        try:
            await page.get_by_role("radio", name=f"{period}Y").click()
            print(f"[{ticker}]  Clicked {period}Y button.")
        except:
            print(f"[{ticker}]  {period}Y button not found.")

        try:
            label = await page.query_selector('label[aria-label="frequency"]')
            dropdown_id = await label.get_attribute("for")
            dropdown = await page.query_selector(f"#{dropdown_id}")
            await dropdown.select_option(label="Daily")
            print(f"[{ticker}]  Frequency set to Daily.")
        except:
            print(f"[{ticker}]  Could not set frequency to Daily.")



        page.on("response", handle_request)

        try:
            await page.get_by_role("button", name="Fundamentals").click()
            await page.evaluate("""
                () => {
                    const input = document.querySelector('input[value="priceEarning"]');
                    if (input) {
                        const event = new MouseEvent('click', {
                            view: window,
                            bubbles: true,
                            cancelable: true
                        });
                        input.dispatchEvent(event);
                    }
                }
            """)
            print(f"[{ticker}] PE chart enabled.")
        except:
            print(f"[{ticker}]  Could not enable PE chart.")

        await page.wait_for_timeout(6000)
        await browser.close()

        if data:
            return data
        else:
            print(f"[{ticker}]  No PE data found.")
            return 1

async def scrapedata(ticker, period):
    data_path = Path(f"DataCache/{ticker}_data_{period}y.pkl")
    if data_path.exists():
        print(f"[{ticker}] Loading from Cache...")
        return pd.read_pickle(data_path)
    
    print(f"[{ticker}] Not found in cache, retrieving data...")
    try:
        data = await WebNavigator(ticker, period)
        if data is None:
            return None
    except Exception as e:
        print(f"[{ticker}]  Error retrieving data: {e}")
        return None

    try:
        df = parse(data)
        save_to_pkl(df, data_path, ticker)
        return df
    except Exception as e:
        print(f"[{ticker}] Error parsing data: {e}")
        return None

async def getpe(ticker1, ticker2, period):
    df1, df2 = await asyncio.gather(scrapedata(ticker1, period), scrapedata(ticker2, period))
    return df1, df2

