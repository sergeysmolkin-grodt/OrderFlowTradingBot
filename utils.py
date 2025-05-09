# forex_orderflow_bot/utils.py

import pandas as pd
from datetime import datetime, timedelta
import os
from config import CHART_OUTPUT_DIR

def ensure_dir(directory_path):
    """Убеждается, что директория существует, если нет - создает ее."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def calculate_pip_value(symbol):
    """Возвращает стоимость пипса для символа."""
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001

def get_current_datetime_str():
    """Возвращает текущую дату и время в виде строки для имен файлов."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def is_bullish_candle(candle):
    """Проверяет, является ли свеча бычьей."""
    return candle['close'] > candle['open']

def is_bearish_candle(candle):
    """Проверяет, является ли свеча медвежьей."""
    return candle['close'] < candle['open']

def print_trade_info(symbol, direction, entry_price, stop_loss, take_profit, timeframe_1h_signal_time, timeframe_3m_entry_time):
    """Выводит информацию о найденной сделке."""
    print("\n--- Найдена Торговая Возможность ---")
    print(f"Символ: {symbol}")
    print(f"Направление: {direction}")
    print(f"Сигнал на 1H: {timeframe_1h_signal_time}")
    print(f"Точка входа (3M): {entry_price:.5f} ({timeframe_3m_entry_time})")
    print(f"Стоп Лосс: {stop_loss:.5f}")
    print(f"Тейк Профит: {take_profit:.5f}")
    print("-----------------------------------\n")