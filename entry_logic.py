# forex_orderflow_bot/entry_logic.py

import pandas as pd
# from config import CHoCH_LOOKBACK_3M # Старая константа
from config import CHoCH_LOOKBACK_ENTRY_TF # <<< ИЗМЕНЕНО: Новая константа

def identify_break_of_structure(df_entry_tf, direction, order_flow_signal_time): # <<< ИЗМЕНЕНО: df_3m на df_entry_tf
    """
    Идентифицирует слом структуры (CHoCH/BoS) на таймфрейме входа (например, 5M) после сигнала с 1H.
    direction: 'bullish_of' или 'bearish_of' от 1H сигнала.
    order_flow_signal_time: время 1H сигнала, чтобы искать слом ПОСЛЕ этого времени.

    Возвращает (entry_price, entry_time, last_swing_for_sl_ref) или (None, None, None)
    """
    # <<< ИЗМЕНЕНО: df_3m на df_entry_tf и CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF
    if df_entry_tf.empty or len(df_entry_tf) < CHoCH_LOOKBACK_ENTRY_TF + 2:
        return None, None, None

    # Ищем слом только на данных ПОСЛЕ времени 1H сигнала
    relevant_entry_data = df_entry_tf[df_entry_tf.index > order_flow_signal_time] # <<< ИЗМЕНЕНО: relevant_3m_data
    # <<< ИЗМЕНЕНО: CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF
    if len(relevant_entry_data) < CHoCH_LOOKBACK_ENTRY_TF + 2 :
        return None, None, None

    # <<< ИЗМЕНЕНО: CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF и relevant_3m_data на relevant_entry_data
    for i in range(CHoCH_LOOKBACK_ENTRY_TF, len(relevant_entry_data)):
        current_candle_entry_tf = relevant_entry_data.iloc[i] # <<< ИЗМЕНЕНО: current_candle_3m
        # <<< ИЗМЕНЕНО: history_3m, CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF
        history_entry_tf = relevant_entry_data.iloc[i - CHoCH_LOOKBACK_ENTRY_TF : i]

        if history_entry_tf.empty:
            continue

        if direction == 'bullish_of':
            # <<< ИЗМЕНЕНО: history_3m на history_entry_tf
            last_swing_high = history_entry_tf['high'].max()

            # <<< ИЗМЕНЕНО: current_candle_3m на current_candle_entry_tf
            if current_candle_entry_tf['close'] > last_swing_high:
                entry_price = current_candle_entry_tf['close']
                entry_time = current_candle_entry_tf.name
                # <<< ИЗМЕНЕНО: history_3m на history_entry_tf
                last_local_low_before_break = history_entry_tf['low'].min()
                return entry_price, entry_time, last_local_low_before_break

        elif direction == 'bearish_of':
            # <<< ИЗМЕНЕНО: history_3m на history_entry_tf
            last_swing_low = history_entry_tf['low'].min()

            # <<< ИЗМЕНЕНО: current_candle_3m на current_candle_entry_tf
            if current_candle_entry_tf['close'] < last_swing_low:
                entry_price = current_candle_entry_tf['close']
                entry_time = current_candle_entry_tf.name
                # <<< ИЗМЕНЕНО: history_3m на history_entry_tf
                last_local_high_before_break = history_entry_tf['high'].max()
                return entry_price, entry_time, last_local_high_before_break

    return None, None, None


if __name__ == '__main__':
    from data_handler import get_historical_data
    # from config import DEFAULT_SYMBOL, TIMEFRAME_3M # Старое
    from config import DEFAULT_SYMBOL, TIMEFRAME_ENTRY, TWELVEDATA_API_KEY # <<< ИЗМЕНЕНО
    from datetime import datetime, timedelta

    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY:
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY в .env файле или в config.py")
    else:
        print("Тестирование entry_logic.py...")
        symbol = DEFAULT_SYMBOL
        # <<< ИЗМЕНЕНО: TIMEFRAME_3M на TIMEFRAME_ENTRY
        df_entry_tf_data = get_historical_data(symbol, TIMEFRAME_ENTRY, outputsize=100)

        # <<< ИЗМЕНЕНО: df_3m на df_entry_tf_data
        if not df_entry_tf_data.empty:
            # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
            print(f"Получено {len(df_entry_tf_data)} {TIMEFRAME_ENTRY} свечей для {symbol}.")

            # <<< ИЗМЕНЕНО: df_3m на df_entry_tf_data и CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF
            if len(df_entry_tf_data) > CHoCH_LOOKBACK_ENTRY_TF + 21:
                # <<< ИЗМЕНЕНО: df_3m на df_entry_tf_data
                order_flow_signal_time_bullish = df_entry_tf_data.index[-(CHoCH_LOOKBACK_ENTRY_TF + 21)]
                order_flow_signal_time_bearish = df_entry_tf_data.index[-(CHoCH_LOOKBACK_ENTRY_TF + 21)]

                print(f"Имитация 1H сигнала (бычий) в: {order_flow_signal_time_bullish}")
                # <<< ИЗМЕНЕНО: df_3m на df_entry_tf_data
                entry_price_b, entry_time_b, _ = identify_break_of_structure(df_entry_tf_data, 'bullish_of', order_flow_signal_time_bullish)
                if entry_price_b:
                    # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
                    print(f"  Найден бычий слом на {TIMEFRAME_ENTRY}: Цена входа={entry_price_b:.5f} в {entry_time_b}")
                else:
                     # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
                    print(f"  Бычий слом на {TIMEFRAME_ENTRY} не найден.")

                print(f"Имитация 1H сигнала (медвежий) в: {order_flow_signal_time_bearish}")
                # <<< ИЗМЕНЕНО: df_3m на df_entry_tf_data
                entry_price_s, entry_time_s, _ = identify_break_of_structure(df_entry_tf_data, 'bearish_of', order_flow_signal_time_bearish)
                if entry_price_s:
                    # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
                    print(f"  Найден медвежий слом на {TIMEFRAME_ENTRY}: Цена входа={entry_price_s:.5f} в {entry_time_s}")
                else:
                    # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
                    print(f"  Медвежий слом на {TIMEFRAME_ENTRY} не найден.")
            else:
                 # <<< ИЗМЕНЕНО: "3М" на TIMEFRAME_ENTRY
                print(f"Недостаточно данных {TIMEFRAME_ENTRY} для теста с имитацией сигнала.")

        else:
            # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
            print(f"Не удалось получить {TIMEFRAME_ENTRY} данные для {symbol} для теста entry_logic.")