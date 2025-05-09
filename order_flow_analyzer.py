# forex_orderflow_bot/orderflow_analyzer.py

import pandas as pd
import numpy as np # Для возможных будущих оптимизаций, пока не критично
from config import (
    FVG_LOOKBACK, LIQUIDITY_LOOKBACK, REACTION_CANDLES, MIN_FVG_SIZE_PIPS,
    FRACTAL_REACTION_MIN_CANDLES, FRACTAL_REACTION_MAX_CANDLES, FRACTAL_REACTION_PIP_PROGRESSION
) # <<< ДОБАВЛЕНЫ НОВЫЕ КОНФИГИ
from utils import calculate_pip_value, is_bullish_candle, is_bearish_candle # Убедитесь, что они есть

# --- НОВАЯ ФУНКЦИЯ: Идентификация фрактальных уровней ---
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

    # df.index должен быть DatetimeIndex
    # df['high'] и df['low'] должны быть float
    highs = df['high'].to_numpy() # Для скорости доступа
    lows = df['low'].to_numpy()
    times = df.index

    for i in range(1, len(df) - 1):
        # Фрактальный максимум (потенциальный BSL)
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            fractal_highs.append({'time': times[i], 'price': highs[i]})

        # Фрактальный минимум (потенциальный SSL)
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            fractal_lows.append({'time': times[i], 'price': lows[i]})
            
    return fractal_highs, fractal_lows

# --- НОВАЯ ФУНКЦИЯ: Проверка реакции на снятие фрактальной ликвидности ---
def check_fractal_sweep_reaction(df, sweep_candle_idx, 
                                 context_is_bearish=True, 
                                 pip_size_for_symbol=0.0001):
    """
    Проверяет реакцию цены после снятия фрактальной ликвидности.
    sweep_candle_idx: индекс свечи, которая совершила снятие.
    context_is_bearish: True, если ожидаем медвежью реакцию (после снятия BSL).
    pip_size_for_symbol: размер пипса для текущего инструмента.
    """
    
    progression_threshold_abs = FRACTAL_REACTION_PIP_PROGRESSION * pip_size_for_symbol
    
    start_check_idx = sweep_candle_idx + 1 # Начинаем проверку со свечи ПОСЛЕ снятия
    
    # Проверяем последовательности длиной от MIN до MAX свечей
    for num_reaction_candles in range(FRACTAL_REACTION_MIN_CANDLES, FRACTAL_REACTION_MAX_CANDLES + 1):
        current_check_end_idx = start_check_idx + num_reaction_candles - 1
        
        if current_check_end_idx >= len(df):
            break # Недостаточно данных для проверки последовательности такой длины

        reaction_candles_df = df.iloc[start_check_idx : current_check_end_idx + 1]
        
        if len(reaction_candles_df) < num_reaction_candles: # Дополнительная проверка
            continue

        is_progressive_structure = True
        # Для прогрессивной структуры нужно как минимум 2 свечи в reaction_candles_df,
        # чтобы было что сравнивать (candle[j] vs candle[j-1]).
        # num_reaction_candles начинается с FRACTAL_REACTION_MIN_CANDLES (например, 3).
        # Значит, в reaction_candles_df будет минимум 3 свечи.
        # Цикл по j будет от 1 до num_reaction_candles - 1.
        
        if num_reaction_candles < 2: # Если MIN_CANDLES = 1, эта логика не сработает корректно
             if num_reaction_candles == 1: # Проверка одной свечи (простое направление)
                 first_reaction_candle = reaction_candles_df.iloc[0]
                 if context_is_bearish:
                     if not is_bearish_candle(first_reaction_candle): is_progressive_structure = False
                 else: # Bullish
                     if not is_bullish_candle(first_reaction_candle): is_progressive_structure = False
             else: # 0 свечей
                 is_progressive_structure = False

        else: # num_reaction_candles >= 2
            for j in range(1, len(reaction_candles_df)): # Сравниваем j-ую свечу с (j-1)-ой
                prev_high = reaction_candles_df['high'].iloc[j-1]
                curr_high = reaction_candles_df['high'].iloc[j]
                prev_low = reaction_candles_df['low'].iloc[j-1]
                curr_low = reaction_candles_df['low'].iloc[j]

                if context_is_bearish:
                    # High[j] должен быть ниже High[j-1] на X пипсов, И Low[j] ниже Low[j-1] на X пипсов
                    if not (curr_high < prev_high - progression_threshold_abs and \
                            curr_low < prev_low - progression_threshold_abs):
                        is_progressive_structure = False
                        break
                else: # Bullish context
                    if not (curr_high > prev_high + progression_threshold_abs and \
                            curr_low > prev_low + progression_threshold_abs):
                        is_progressive_structure = False
                        break
        
        if is_progressive_structure:
            return True # Найдена подтверждающая реакция нужной длины
            
    return False # Подтверждающая реакция не найдена

# --- Существующая функция check_reaction (для теста FVG) ---
# Оставляем ее как есть или также можно улучшить, но это отдельная задача
def check_reaction(df, index, direction, num_candles=REACTION_CANDLES):
    # ... (код этой функции остается прежним) ...
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

# --- Существующая функция identify_fair_value_gaps ---
# Оставляем ее как есть
def identify_fair_value_gaps(df_1h, symbol):
    # ... (код этой функции остается прежним) ...
    fvgs = []
    pip_value = calculate_pip_value(symbol)
    min_fvg_size = MIN_FVG_SIZE_PIPS * pip_value
    if len(df_1h) < 3:
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
                fvgs.append({'fvg_time': candle2.name, 'fvg_top': fvg_top, 'fvg_bottom': fvg_bottom, 'type': 'bullish', 'mid_price': (fvg_top + fvg_bottom) / 2, 'trigger_candle_time': candle3.name})
        if candle3['high'] < candle1['low']:
            fvg_top = candle1['low']
            fvg_bottom = candle3['high']
            fvg_size = fvg_top - fvg_bottom
            if fvg_size >= min_fvg_size:
                fvgs.append({'fvg_time': candle2.name, 'fvg_top': fvg_top, 'fvg_bottom': fvg_bottom, 'type': 'bearish', 'mid_price': (fvg_top + fvg_bottom) / 2, 'trigger_candle_time': candle3.name})
    if not fvgs:
        return pd.DataFrame(columns=['fvg_time', 'fvg_top', 'fvg_bottom', 'type', 'mid_price', 'trigger_candle_time'])
    return pd.DataFrame(fvgs).set_index('fvg_time').sort_index()


# --- ОБНОВЛЕННАЯ ФУНКЦИЯ analyze_order_flow ---
def analyze_order_flow(df_1h, symbol):
    if len(df_1h) < LIQUIDITY_LOOKBACK + FVG_LOOKBACK + FRACTAL_REACTION_MAX_CANDLES + REACTION_CANDLES + 10:
        print("Недостаточно данных на 1H для анализа Order Flow с учетом фракталов.")
        return []

    order_flow_setups = []
    pip_value_for_symbol = calculate_pip_value(symbol) # Используем для SL и реакции
    all_fvgs_df = identify_fair_value_gaps(df_1h, symbol)
    
    # 1. Идентифицируем все фрактальные уровни один раз
    all_fractal_highs, all_fractal_lows = identify_fractal_levels(df_1h)
    if not all_fractal_highs and not all_fractal_lows:
        print("Фрактальные уровни не найдены.")
        return []

    # Преобразуем в DataFrame для удобства фильтрации (если много фракталов)
    # Индексируем по времени для быстрого поиска
    df_fractal_highs = pd.DataFrame(all_fractal_highs).set_index('time') if all_fractal_highs else pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))
    df_fractal_lows = pd.DataFrame(all_fractal_lows).set_index('time') if all_fractal_lows else pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))


    # `i` - это индекс свечи, на которой ЗАВЕРШАЕТСЯ реакция на тест FVG
    # Все события OF (снятие, реакция на снятие, FVG, тест FVG) должны произойти ДО этой свечи `i`.
    min_total_candles_for_pattern = (FRACTAL_REACTION_MIN_CANDLES + # реакция на снятие
                                     3 +                         # формирование FVG (3 свечи)
                                     1 +                         # тест FVG
                                     REACTION_CANDLES)           # реакция на тест FVG
                                     # + еще неск свечей запаса

    for i in range(min_total_candles_for_pattern + LIQUIDITY_LOOKBACK, len(df_1h)):
        current_signal_completion_time = df_1h.index[i]

        # --- Поиск Bearish Order Flow (Снятие фрактального BSL -> Медвежья реакция -> Bearish FVG -> Тест FVG -> Медвежья реакция) ---
        
        # Перебираем все свечи, которые могли быть sweep_candle (свеча, снимающая ликвидность)
        # sweep_candle_idx должен быть значительно раньше, чем i
        # Минимальное расстояние: sweep + fractal_reaction + FVG_formation + FVG_test + FVG_test_reaction
        required_gap_for_events = FRACTAL_REACTION_MIN_CANDLES + 3 + 1 + REACTION_CANDLES + 2 # + запас

        for sweep_candle_idx in range(1, i - required_gap_for_events): # Начинаем с 1, так как фракталу нужна свеча до
            sweep_candle = df_1h.iloc[sweep_candle_idx]
            sweep_candle_time = df_1h.index[sweep_candle_idx]

            # Ищем фрактальные максимумы (BSL), которые были ДО sweep_candle_time
            # и могли быть сняты этой свечой.
            # Ограничим поиск фракталов не слишком старыми (в пределах LIQUIDITY_LOOKBACK от sweep_candle)
            earliest_fractal_time_limit = sweep_candle_time - pd.Timedelta(hours=LIQUIDITY_LOOKBACK + 5) # +5 для запаса

            if not df_fractal_highs.empty:
                candidate_bsls = df_fractal_highs[
                    (df_fractal_highs.index < sweep_candle_time) &  # Фрактал сформировался до свечи снятия
                    (df_fractal_highs.index >= earliest_fractal_time_limit)
                ]

                for fractal_bsl_time, fractal_bsl_data in candidate_bsls.iterrows():
                    fractal_bsl_level = fractal_bsl_data['price']

                    if sweep_candle['high'] > fractal_bsl_level:  # Фрактальный BSL снят
                        # 2. Проверяем медвежью реакцию на это снятие фрактала
                        if check_fractal_sweep_reaction(df_1h, sweep_candle_idx,
                                                        context_is_bearish=True,
                                                        pip_size_for_symbol=pip_value_for_symbol):
                            
                            # Реакция на снятие BSL подтверждена.
                            # FVG должен формироваться ПОСЛЕ этой реакции.
                            # Время окончания реакции на снятие: sweep_candle_time + X свечей
                            # Для простоты, FVG должен начаться хотя бы через FRACTAL_REACTION_MIN_CANDLES после sweep_candle
                            fvg_earliest_start_time = df_1h.index[sweep_candle_idx + FRACTAL_REACTION_MIN_CANDLES]
                            
                            # 3. Ищем Bearish FVG, сформированный ПОСЛЕ реакции на снятие
                            relevant_fvgs = all_fvgs_df[
                                (all_fvgs_df.index > fvg_earliest_start_time) & # FVG НАЧИНАЕТСЯ после реакции на снятие
                                (all_fvgs_df['trigger_candle_time'] < current_signal_completion_time) & # FVG ЗАКОНЧИЛСЯ до текущей свечи i
                                (all_fvgs_df['type'] == 'bearish') &
                                (all_fvgs_df['fvg_top'] < sweep_candle['high']) # FVG ниже точки снятия BSL
                            ]

                            if not relevant_fvgs.empty:
                                for _, fvg_data in relevant_fvgs.iterrows(): # Перебираем подходящие FVG
                                    fvg_formation_trigger_time = fvg_data['trigger_candle_time'] # Время 3й свечи FVG

                                    # 4. Ищем тест FVG и последующую медвежью реакцию (стандартную)
                                    # Тест должен произойти после формирования FVG и до (i - REACTION_CANDLES)
                                    idx_fvg_trigger = df_1h.index.get_loc(fvg_formation_trigger_time)
                                    
                                    for test_candle_idx in range(idx_fvg_trigger + 1, i - REACTION_CANDLES +1):
                                        test_candle = df_1h.iloc[test_candle_idx]
                                        
                                        # Условие касания/теста Bearish FVG
                                        if test_candle['high'] >= fvg_data['fvg_bottom'] and test_candle['low'] <= fvg_data['fvg_top']:
                                            # 5. Проверяем стандартную реакцию на тест FVG
                                            if check_reaction(df_1h, test_candle_idx, 'bearish', num_candles=REACTION_CANDLES):
                                                # --- Bearish Order Flow Сформирован ---
                                                # Определяем цель: фрактальный SSL перед снятым BSL
                                                target_ssl_level = None
                                                if not df_fractal_lows.empty:
                                                    # Ищем фрактальные минимумы ДО времени формирования BSL, который был снят
                                                    potential_ssl_targets = df_fractal_lows[df_fractal_lows.index < fractal_bsl_time]
                                                    if not potential_ssl_targets.empty:
                                                        target_ssl_level = potential_ssl_targets['price'].min() # Берем самый низкий из них
                                                
                                                if target_ssl_level is None: # Fallback, если не нашли фрактальный SSL
                                                    target_ssl_level = df_1h.iloc[:df_1h.index.get_loc(fractal_bsl_time)]['low'].min()


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
                                                # Можно добавить break для выхода из циклов, если один сетап достаточен
                                                break # Выход из цикла test_candle_idx
                                    
                                    # Если сетап добавлен, выходим из цикла по FVG
                                    if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time and \
                                       order_flow_setups[-1]['liquidity_sweep_time'] == fractal_bsl_time and \
                                       order_flow_setups[-1]['fvg_tested_data']['fvg_time'] == fvg_data.name: # fvg_data.name это индекс FVG (fvg_time)
                                        break 
                            
                            # Если сетап добавлен, выходим из цикла по фракталам BSL
                            if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time and \
                               order_flow_setups[-1]['liquidity_sweep_time'] == fractal_bsl_time:
                                break 
            
            # Если сетап добавлен, выходим из цикла по sweep_candle_idx
            if order_flow_setups and order_flow_setups[-1]['actual_sweep_time'] == sweep_candle_time:
                break # Нашли один сетап для текущего current_signal_completion_time, можно перейти к следующему `i`


        # --- Поиск Bullish Order Flow (Снятие фрактального SSL -> ...) ---
        # TODO: Реализовать аналогичную логику для Bullish Order Flow, используя:
        # - df_fractal_lows для поиска SSL
        # - check_fractal_sweep_reaction с context_is_bearish=False
        # - Bullish FVG
        # - check_reaction на тест FVG с direction='bullish'
        # - Цель: фрактальный BSL
        pass # Заглушка для Bullish OF

    # Фильтрация дубликатов (можно улучшить ключ уникальности)
    if order_flow_setups:
        final_setups = []
        seen_signals_keys = set()
        sorted_setups = sorted(order_flow_setups, key=lambda x: x['signal_candle_time'], reverse=True)
        for setup in sorted_setups:
            signal_key = (
                setup['actual_sweep_time'], 
                setup['liquidity_sweep_time'], # Добавим время исходного фрактала
                setup['fvg_test_time'], 
                setup['type']
            )
            if signal_key not in seen_signals_keys:
                final_setups.append(setup)
                seen_signals_keys.add(signal_key)
        return sorted(final_setups, key=lambda x: x['signal_candle_time'])

    return []


if __name__ == '__main__':
    from data_handler import get_historical_data
    # from config import DEFAULT_SYMBOL, TIMEFRAME_1H # config уже импортирован выше

    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY: # Добавьте TWELVEDATA_API_KEY в импорт config или определите
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY")
    else:
        print("Тестирование orderflow_analyzer.py с фракталами...")
        symbol = DEFAULT_SYMBOL # Убедитесь, что DEFAULT_SYMBOL доступен (из config)
        df_1h_test = get_historical_data(symbol, TIMEFRAME_1H, outputsize=500)

        if not df_1h_test.empty:
            print(f"Получено {len(df_1h_test)} 1H свечей для {symbol}.")
            
            # Тест identify_fractal_levels
            # highs, lows = identify_fractal_levels(df_1h_test)
            # print(f"Найдено фрактальных максимумов: {len(highs)}")
            # if highs: print(f"Последний фрактальный максимум: {highs[-1]}")
            # print(f"Найдено фрактальных минимумов: {len(lows)}")
            # if lows: print(f"Последний фрактальный минимум: {lows[-1]}")

            order_flow_signals_test = analyze_order_flow(df_1h_test, symbol)
            if order_flow_signals_test:
                print(f"\nНайдено {len(order_flow_signals_test)} потенциальных Order Flow сигналов (с фракталами):")
                for signal in order_flow_signals_test[-3:]: # Показать последние 3
                    print(f"  Тип: {signal['type']}, Время сигнала 1H: {signal['signal_candle_time']}, "
                          f"Цель: {signal['target_price']:.5f}, SL: {signal['stop_loss_level']:.5f}")
                    print(f"    Ликвидность: {signal['liquidity_type']} @ {signal['liquidity_sweep_price']:.5f} ({signal['liquidity_sweep_time']})")
                    print(f"    FVG протестирован: {signal['fvg_tested_data']['type']} FVG @ {signal['fvg_tested_data']['mid_price']:.5f} (сформирован в {signal['fvg_tested_data']['fvg_time']})")
                    print(f"    Тест FVG: {signal['fvg_test_time']}")
            else:
                print("Order Flow сигналы (с фракталами) не найдены.")
        else:
            print(f"Не удалось получить данные для {symbol} для теста orderflow_analyzer.")