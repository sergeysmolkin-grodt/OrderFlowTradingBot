# forex_orderflow_bot/orderflow_analyzer.py

import pandas as pd
import numpy as np 
from config import (
    FVG_LOOKBACK, LIQUIDITY_LOOKBACK, REACTION_CANDLES, MIN_FVG_SIZE_PIPS,
    FRACTAL_REACTION_MIN_CANDLES, FRACTAL_REACTION_MAX_CANDLES, FRACTAL_REACTION_PIP_PROGRESSION,
    TWELVEDATA_API_KEY, DEFAULT_SYMBOL, TIMEFRAME_1H # Добавлены для if __name__ == '__main__':
)
from utils import calculate_pip_value, is_bullish_candle, is_bearish_candle

def identify_fractal_levels(df):
    """
    Идентифицирует фрактальные максимумы и минимумы на 1H графике.
    Фрактал: 3 свечи, где средняя свеча имеет самый высокий максимум (для BSL)
             или самый низкий минимум (для SSL) среди трех.
    Возвращает два списка словарей: [{'time': timestamp, 'price': float}, ...]
    """
    fractal_highs = []
    fractal_lows = []

    if len(df) < 3:
        return fractal_highs, fractal_lows

    highs = df['high'].to_numpy() 
    lows = df['low'].to_numpy()
    times = df.index

    for i in range(1, len(df) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            fractal_highs.append({'time': times[i], 'price': highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            fractal_lows.append({'time': times[i], 'price': lows[i]})
            
    return fractal_highs, fractal_lows

def check_fractal_sweep_reaction(df, sweep_candle_idx, 
                                 context_is_bearish=True, 
                                 pip_size_for_symbol=0.0001):
    """
    Проверяет реакцию цены после снятия фрактальной ликвидности.
    """
    progression_threshold_abs = FRACTAL_REACTION_PIP_PROGRESSION * pip_size_for_symbol
    start_check_idx = sweep_candle_idx + 1
    
    for num_reaction_candles in range(FRACTAL_REACTION_MIN_CANDLES, FRACTAL_REACTION_MAX_CANDLES + 1):
        current_check_end_idx = start_check_idx + num_reaction_candles - 1
        if current_check_end_idx >= len(df):
            break 

        reaction_candles_df = df.iloc[start_check_idx : current_check_end_idx + 1]
        if len(reaction_candles_df) < num_reaction_candles:
            continue

        is_progressive_structure = True
        if num_reaction_candles < 2:
             if num_reaction_candles == 1:
                 first_reaction_candle = reaction_candles_df.iloc[0]
                 if context_is_bearish:
                     if not is_bearish_candle(first_reaction_candle): is_progressive_structure = False
                 else: 
                     if not is_bullish_candle(first_reaction_candle): is_progressive_structure = False
             else: 
                 is_progressive_structure = False
        else: 
            for j in range(1, len(reaction_candles_df)):
                prev_high = reaction_candles_df['high'].iloc[j-1]
                curr_high = reaction_candles_df['high'].iloc[j]
                prev_low = reaction_candles_df['low'].iloc[j-1]
                curr_low = reaction_candles_df['low'].iloc[j]

                if context_is_bearish:
                    if not (curr_high < prev_high - progression_threshold_abs and \
                            curr_low < prev_low - progression_threshold_abs):
                        is_progressive_structure = False
                        break
                else: 
                    if not (curr_high > prev_high + progression_threshold_abs and \
                            curr_low > prev_low + progression_threshold_abs):
                        is_progressive_structure = False
                        break
        if is_progressive_structure:
            return True 
    return False

def check_reaction(df, index, direction, num_candles=REACTION_CANDLES):
    if index + num_candles >= len(df):
        return False 
    relevant_candles = df.iloc[index + 1 : index + 1 + num_candles]
    if relevant_candles.empty:
        return False
    if direction == 'bullish':
        return any(relevant_candles['close'] > relevant_candles['open']) and \
               relevant_candles.iloc[-1]['close'] > relevant_candles.iloc[-1]['open']
    elif direction == 'bearish':
        return any(relevant_candles['close'] < relevant_candles['open']) and \
               relevant_candles.iloc[-1]['close'] < relevant_candles.iloc[-1]['open']
    return False

def identify_fair_value_gaps(df_1h, symbol):
    fvgs = []
    pip_value = calculate_pip_value(symbol)
    min_fvg_size = MIN_FVG_SIZE_PIPS * pip_value

    if len(df_1h) < 3:
        return pd.DataFrame(columns=['fvg_time', 'fvg_top', 'fvg_bottom', 'type', 'mid_price', 'trigger_candle_time'])

    required_cols = ['high', 'low']
    if not all(col in df_1h.columns for col in required_cols):
        print(f"Ошибка в identify_fair_value_gaps: Отсутствуют необходимые столбцы: {required_cols}")
        return pd.DataFrame(columns=['fvg_time', 'fvg_top', 'fvg_bottom', 'type', 'mid_price', 'trigger_candle_time'])

    for i in range(len(df_1h) - 2):
        candle1 = df_1h.iloc[i]
        candle2 = df_1h.iloc[i+1]
        candle3 = df_1h.iloc[i+2]

        if candle3['low'] > candle1['high']:
            fvg_top = candle3['low']
            fvg_bottom = candle1['high']
            fvg_size = fvg_top - fvg_bottom
            if fvg_size >= min_fvg_size:
                fvgs.append({
                    'fvg_time': candle2.name, 
                    'fvg_top': fvg_top,
                    'fvg_bottom': fvg_bottom,
                    'type': 'bullish',
                    'mid_price': (fvg_top + fvg_bottom) / 2,
                    'trigger_candle_time': candle3.name
                })
        if candle3['high'] < candle1['low']:
            fvg_top = candle1['low']
            fvg_bottom = candle3['high']
            fvg_size = fvg_top - fvg_bottom 
            if fvg_size >= min_fvg_size:
                fvgs.append({
                    'fvg_time': candle2.name,
                    'fvg_top': fvg_top,
                    'fvg_bottom': fvg_bottom,
                    'type': 'bearish',
                    'mid_price': (fvg_top + fvg_bottom) / 2,
                    'trigger_candle_time': candle3.name
                })
    if not fvgs:
        return pd.DataFrame(columns=['fvg_time', 'fvg_top', 'fvg_bottom', 'type', 'mid_price', 'trigger_candle_time'])
    return pd.DataFrame(fvgs).sort_values(by='fvg_time')


def analyze_order_flow(df_1h, symbol):
    if len(df_1h) < LIQUIDITY_LOOKBACK + FVG_LOOKBACK + FRACTAL_REACTION_MAX_CANDLES + REACTION_CANDLES + 15: # Немного увеличен запас
        print("Недостаточно данных на 1H для анализа Order Flow с учетом фракталов.")
        return []

    order_flow_setups = []
    pip_value_for_symbol = calculate_pip_value(symbol)
    all_fvgs_df = identify_fair_value_gaps(df_1h, symbol)
    
    all_fractal_highs, all_fractal_lows = identify_fractal_levels(df_1h)
    if not all_fractal_highs and not all_fractal_lows:
        # print("Фрактальные уровни не найдены.") # Можно раскомментировать для отладки
        pass # Продолжаем, даже если нет фракталов, т.к. старая логика может что-то найти

    df_fractal_highs = pd.DataFrame(all_fractal_highs).set_index('time') if all_fractal_highs else pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))
    df_fractal_lows = pd.DataFrame(all_fractal_lows).set_index('time') if all_fractal_lows else pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))

    min_total_candles_for_pattern = (FRACTAL_REACTION_MIN_CANDLES + 3 + 1 + REACTION_CANDLES + 5) # + запас

    for i in range(min_total_candles_for_pattern + LIQUIDITY_LOOKBACK, len(df_1h)):
        current_signal_completion_time = df_1h.index[i]
        
        # --- Поиск Bearish Order Flow ---
        required_gap_for_events = FRACTAL_REACTION_MIN_CANDLES + 3 + 1 + REACTION_CANDLES + 2 

        for sweep_candle_idx in range(1, i - required_gap_for_events):
            sweep_candle = df_1h.iloc[sweep_candle_idx]
            sweep_candle_time = df_1h.index[sweep_candle_idx]
            earliest_fractal_time_limit = sweep_candle_time - pd.Timedelta(hours=LIQUIDITY_LOOKBACK + 5) 

            if not df_fractal_highs.empty:
                candidate_bsls = df_fractal_highs[
                    (df_fractal_highs.index < sweep_candle_time) &
                    (df_fractal_highs.index >= earliest_fractal_time_limit)
                ]
                for fractal_bsl_time, fractal_bsl_data in candidate_bsls.iterrows():
                    fractal_bsl_level = fractal_bsl_data['price']
                    if sweep_candle['high'] > fractal_bsl_level:
                        if check_fractal_sweep_reaction(df_1h, sweep_candle_idx,
                                                        context_is_bearish=True,
                                                        pip_size_for_symbol=pip_value_for_symbol):
                            fvg_earliest_start_time = df_1h.index[sweep_candle_idx + FRACTAL_REACTION_MIN_CANDLES]
                            
                            # ИСПРАВЛЕНИЕ ЗДЕСЬ: Используем all_fvgs_df['fvg_time'] вместо all_fvgs_df.index
                            relevant_fvgs = all_fvgs_df[
                                (all_fvgs_df['fvg_time'] > fvg_earliest_start_time) & 
                                (all_fvgs_df['trigger_candle_time'] < current_signal_completion_time) &
                                (all_fvgs_df['type'] == 'bearish') &
                                (all_fvgs_df['fvg_top'] < sweep_candle['high'])
                            ]

                            if not relevant_fvgs.empty:
                                for _, fvg_data in relevant_fvgs.iterrows():
                                    fvg_formation_trigger_time = fvg_data['trigger_candle_time']
                                    idx_fvg_trigger = df_1h.index.get_loc(fvg_formation_trigger_time)
                                    
                                    for test_candle_idx in range(idx_fvg_trigger + 1, i - REACTION_CANDLES + 1):
                                        test_candle = df_1h.iloc[test_candle_idx]
                                        if test_candle['high'] >= fvg_data['fvg_bottom'] and test_candle['low'] <= fvg_data['fvg_top']:
                                            if check_reaction(df_1h, test_candle_idx, 'bearish', num_candles=REACTION_CANDLES):
                                                target_ssl_level = None
                                                if not df_fractal_lows.empty:
                                                    potential_ssl_targets = df_fractal_lows[df_fractal_lows.index < fractal_bsl_time]
                                                    if not potential_ssl_targets.empty:
                                                        target_ssl_level = potential_ssl_targets['price'].min()
                                                if target_ssl_level is None:
                                                    try: # Добавим try-except для безопасности
                                                        target_ssl_level = df_1h.iloc[:df_1h.index.get_loc(fractal_bsl_time)]['low'].min()
                                                    except (IndexError, KeyError): # Если срез пуст или индекс не найден
                                                        target_ssl_level = df_1h['low'].min() # Очень грубый fallback
                                                
                                                setup = {
                                                    'type': 'bearish_of',
                                                    'liquidity_sweep_price': fractal_bsl_level,
                                                    'liquidity_sweep_time': fractal_bsl_time,
                                                    'actual_sweep_high_price': sweep_candle['high'],
                                                    'actual_sweep_time': sweep_candle_time,
                                                    'liquidity_type': 'BSL_Fractal',
                                                    'fvg_tested_data': fvg_data.to_dict(),
                                                    'fvg_test_time': df_1h.index[test_candle_idx],
                                                    'signal_candle_time': current_signal_completion_time,
                                                    'target_price': target_ssl_level,
                                                    'stop_loss_level': sweep_candle['high'] + pip_value_for_symbol
                                                }
                                                order_flow_setups.append(setup)
                                                break 
                                    if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time and \
                                       order_flow_setups[-1]['liquidity_sweep_time'] == fractal_bsl_time and \
                                       order_flow_setups[-1]['fvg_tested_data']['fvg_time'] == fvg_data['fvg_time']: # Используем fvg_data['fvg_time']
                                        break 
                            if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time and \
                               order_flow_setups[-1]['liquidity_sweep_time'] == fractal_bsl_time:
                                break 
            if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time:
                break

        # --- Поиск Bullish Order Flow (TODO: Аналогично Bearish OF) ---
        # В этой секции также нужно будет заменить all_fvgs_df.index на all_fvgs_df['fvg_time'] при фильтрации
        # Пример для Bullish (нужно будет доработать полностью):
        for sweep_candle_idx in range(1, i - required_gap_for_events):
            sweep_candle = df_1h.iloc[sweep_candle_idx]
            sweep_candle_time = df_1h.index[sweep_candle_idx]
            earliest_fractal_time_limit = sweep_candle_time - pd.Timedelta(hours=LIQUIDITY_LOOKBACK + 5)

            if not df_fractal_lows.empty:
                candidate_ssls = df_fractal_lows[
                    (df_fractal_lows.index < sweep_candle_time) &
                    (df_fractal_lows.index >= earliest_fractal_time_limit)
                ]
                for fractal_ssl_time, fractal_ssl_data in candidate_ssls.iterrows():
                    fractal_ssl_level = fractal_ssl_data['price']
                    if sweep_candle['low'] < fractal_ssl_level:
                        if check_fractal_sweep_reaction(df_1h, sweep_candle_idx,
                                                        context_is_bearish=False, # Bullish context
                                                        pip_size_for_symbol=pip_value_for_symbol):
                            fvg_earliest_start_time = df_1h.index[sweep_candle_idx + FRACTAL_REACTION_MIN_CANDLES]
                            
                            relevant_fvgs = all_fvgs_df[
                                (all_fvgs_df['fvg_time'] > fvg_earliest_start_time) & # <<< ИСПРАВЛЕНИЕ
                                (all_fvgs_df['trigger_candle_time'] < current_signal_completion_time) &
                                (all_fvgs_df['type'] == 'bullish') & # Bullish FVG
                                (all_fvgs_df['fvg_bottom'] > sweep_candle['low']) # FVG выше точки снятия SSL
                            ]
                            # ... (дальнейшая логика для Bullish OF по аналогии с Bearish) ...
                            if not relevant_fvgs.empty:
                                # ... (циклы по FVG, тестам и т.д.)
                                pass # Заглушка

    if order_flow_setups:
        final_setups = []
        seen_signals_keys = set()
        sorted_setups = sorted(order_flow_setups, key=lambda x: x['signal_candle_time'], reverse=True)
        for setup in sorted_setups:
            signal_key = (
                setup['actual_sweep_time'], 
                setup['liquidity_sweep_time'], 
                setup['fvg_test_time'], 
                setup['type']
            )
            if signal_key not in seen_signals_keys:
                final_setups.append(setup)
                seen_signals_keys.add(signal_key)
        return sorted(final_setups, key=lambda x: x['signal_candle_time'])
    return []

if __name__ == '__main__':
    from data_handler import get_historical_data # Импорт для теста

    if TWELVEDATA_API_KEY == "355094c06e9c4fe7b4e8d45fb2c1236c" or not TWELVEDATA_API_KEY:
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY")
    else:
        print("Тестирование orderflow_analyzer.py с фракталами...")
        symbol_test_main = DEFAULT_SYMBOL 
        df_1h_test_main = get_historical_data(symbol_test_main, TIMEFRAME_1H, outputsize=500)

        if not df_1h_test_main.empty:
            print(f"Получено {len(df_1h_test_main)} 1H свечей для {symbol_test_main}.")
            order_flow_signals_test_main = analyze_order_flow(df_1h_test_main, symbol_test_main)
            if order_flow_signals_test_main:
                print(f"\nНайдено {len(order_flow_signals_test_main)} потенциальных Order Flow сигналов (с фракталами):")
                for signal in order_flow_signals_test_main[-3:]:
                    print(f"  Тип: {signal['type']}, Время сигнала 1H: {signal['signal_candle_time']}, "
                          f"Цель: {signal.get('target_price', 'N/A')}, SL: {signal.get('stop_loss_level', 'N/A')}") # Добавил .get для безопасности
                    if isinstance(signal.get('fvg_tested_data'), dict):
                        print(f"    Ликвидность: {signal.get('liquidity_type','N/A')} @ {signal.get('liquidity_sweep_price','N/A')} ({signal.get('liquidity_sweep_time','N/A')})")
                        print(f"    FVG протестирован: {signal['fvg_tested_data'].get('type','N/A')} FVG @ {signal['fvg_tested_data'].get('mid_price','N/A')} (сформ. {signal['fvg_tested_data'].get('fvg_time','N/A')})")
                        print(f"    Тест FVG: {signal.get('fvg_test_time','N/A')}")
                    else:
                        print(f"    Проблема с fvg_tested_data: {signal.get('fvg_tested_data')}")
            else:
                print("Order Flow сигналы (с фракталами) не найдены.")
        else:
            print(f"Не удалось получить данные для {symbol_test_main} для теста orderflow_analyzer.")
