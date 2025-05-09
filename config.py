# forex_orderflow_bot/config.py

import os
from dotenv import load_dotenv

load_dotenv() # Загружает переменные из .env файла

# Ключ API TwelveData (храните его в .env файле для безопасности)
# Создайте файл .env в корне проекта и добавьте строку:
# TWELVEDATA_API_KEY="ваш_ключ_здесь"
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "YOUR_TWELVEDATA_API_KEY_FALLBACK")


DEFAULT_SYMBOL = "EUR/USD"
TIMEFRAME_1H = "1h"
# TIMEFRAME_3M = "3min" # Старое значение
TIMEFRAME_ENTRY = "5min" # <<< ИЗМЕНЕНО: Новый таймфрейм для входа
# Убедитесь, что TwelveData поддерживает этот формат (5min поддерживается)

# Параметры анализа
FVG_LOOKBACK = 30
LIQUIDITY_LOOKBACK = 30
REACTION_CANDLES = 2
MIN_FVG_SIZE_PIPS = 0.5
PIP_SIZE = 0.0001

# forex_orderflow_bot/config.py
# ... (существующие параметры) ...

# Параметры для реакции на снятие фрактальной ликвидности
FRACTAL_REACTION_MIN_CANDLES = 3   # Минимальное количество свечей для подтверждения реакции
FRACTAL_REACTION_MAX_CANDLES = 10  # Максимальное количество свечей для проверки реакции
FRACTAL_REACTION_PIP_PROGRESSION = 1.5 # "Пара пипсов" - на сколько пипсов цена должна продвинуться 
                                     # на каждой свече реакции (High[i] < High[i-1] - X)

# Параметры для слома на таймфрейме входа
# CHoCH_LOOKBACK_3M = 15 # Старое имя
CHoCH_LOOKBACK_ENTRY_TF = 15 # <<< ИЗМЕНЕНО: Новое имя константы (значение можно оставить или настроить)

# Параметры графиков
CHART_OUTPUT_DIR = "charts"
CHART_WIDTH = 15
CHART_HEIGHT = 7