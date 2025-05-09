# forex_orderflow_bot/data_handler.py

import requests
import pandas as pd
from config import TWELVEDATA_API_KEY
from datetime import datetime, timedelta

BASE_URL = "https://api.twelvedata.com/"

def get_historical_data(symbol, interval, outputsize=300, end_date=None):
    """
    Получает исторические данные свечей с TwelveData.
    end_date: в формате 'YYYY-MM-DD HH:MM:SS' или None для текущего момента
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": outputsize,
        "format": "JSON"
    }
    if end_date:
        params["end_date"] = end_date

    try:
        response = requests.get(BASE_URL + "time_series", params=params)
        response.raise_for_status()  # Проверка на ошибки HTTP
        data = response.json()

        if data.get("status") == "error":
            print(f"Ошибка API TwelveData: {data.get('message')}")
            return pd.DataFrame()

        if "values" not in data or not data["values"]:
            print(f"Нет данных для {symbol} с интервалом {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.astype(float)
        df = df.iloc[::-1] # API возвращает в обратном порядке, переворачиваем для хронологического
        return df
    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети или HTTP при запросе к TwelveData: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Неожиданная ошибка при получении данных: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Тестирование функции
    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY:
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY в .env файле или в config.py")
    else:
        print("Тестирование data_handler.py...")
        symbol = "EUR/USD"
        interval_1h = "1h"
        interval_3m = "3min"

        # Получаем данные за последние N свечей (end_date не указан)
        df_1h = get_historical_data(symbol, interval_1h, outputsize=100)
        if not df_1h.empty:
            print(f"\n1H данные для {symbol} (последние 5):")
            print(df_1h.tail())
        else:
            print(f"Не удалось получить 1H данные для {symbol}")

        df_3m = get_historical_data(symbol, interval_3m, outputsize=100)
        if not df_3m.empty:
            print(f"\n3M данные для {symbol} (последние 5):")
            print(df_3m.tail())
        else:
            print(f"Не удалось получить 3M данные для {symbol}")

        # Пример получения данных до определенной даты
        # end_date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        # df_1h_specific_end = get_historical_data(symbol, interval_1h, outputsize=50, end_date=end_date_str)
        # if not df_1h_specific_end.empty:
        #     print(f"\n1H данные для {symbol} до {end_date_str} (последние 5):")
        #     print(df_1h_specific_end.tail())
        # else:
        #     print(f"Не удалось получить 1H данные до {end_date_str} для {symbol}")