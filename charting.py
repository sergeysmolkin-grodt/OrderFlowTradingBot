# forex_orderflow_bot/charting.py

import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import os

from config import CHART_OUTPUT_DIR, CHART_WIDTH, CHART_HEIGHT
from utils import ensure_dir, get_current_datetime_str, calculate_pip_value

def plot_order_flow_and_trade(df_1h_full, order_flow_signal, entry_details, symbol):
    ensure_dir(CHART_OUTPUT_DIR)

    if df_1h_full is None or df_1h_full.empty:
        print(f"Ошибка (charting): df_1h_full пуст или None для {symbol}. График не будет построен.")
        return None
    if order_flow_signal is None:
        print(f"Ошибка (charting): order_flow_signal is None для {symbol}. График не будет построен.")
        return None

    # Убедимся, что индекс df_1h_full это DatetimeIndex и он наивный (без TZ)
    if not isinstance(df_1h_full.index, pd.DatetimeIndex):
        try:
            df_1h_full.index = pd.to_datetime(df_1h_full.index)
            print(f"Предупреждение (charting): Индекс df_1h_full преобразован в DatetimeIndex для {symbol}.")
        except Exception as e:
            print(f"Ошибка (charting): Не удалось преобразовать индекс df_1h_full в DatetimeIndex для {symbol}: {e}. График не будет построен.")
            return None
    
    df_1h_full.index = df_1h_full.index.tz_localize(None)


    of_type = order_flow_signal.get('type', 'unknown_of') # Безопасное извлечение
    try:
        signal_time_1h = pd.to_datetime(order_flow_signal['signal_candle_time']).tz_localize(None)
    except Exception as e:
        print(f"Ошибка (charting): Некорректное время сигнала 1H: {order_flow_signal.get('signal_candle_time')}, {e}")
        return None

    focus_time = signal_time_1h 
    if entry_details and entry_details.get('entry_time'):
        try:
            entry_time_dt = pd.to_datetime(entry_details['entry_time']).tz_localize(None)
            if entry_time_dt > signal_time_1h: # focus_time должен быть Timestamp
                focus_time = entry_time_dt
        except Exception as e:
            print(f"Предупреждение (charting): Некорректное время входа: {entry_details.get('entry_time')}, {e}. Используется signal_time_1h как focus_time.")
    
    # --- Логика определения focus_idx и df_to_plot ---
    df_to_plot = None
    _focus_idx_set = False
    focus_idx = -1 

    try:
        if not df_1h_full.index.is_monotonic_increasing: # Убедимся, что индекс отсортирован
            df_1h_full = df_1h_full.sort_index()

        if df_1h_full.index.empty:
            print(f"Предупреждение (charting): Индекс в df_1h_full пуст после начальных проверок для {symbol}.")
        elif focus_time in df_1h_full.index: 
            focus_idx = df_1h_full.index.get_loc(focus_time)
            _focus_idx_set = True
        else:
            # Альтернатива для method='nearest'
            time_diffs = (df_1h_full.index - focus_time).to_series().abs() 
            if time_diffs.empty:
                print(f"Ошибка (charting): time_diffs пуст (индекс df_1h_full не пуст, но разница не вычислена). Focus time: {focus_time} для {symbol}.")
            else:
                closest_time = time_diffs.idxmin()
                focus_idx = df_1h_full.index.get_loc(closest_time)
                _focus_idx_set = True
        
        if not _focus_idx_set:
            print(f"Предупреждение (charting): Не удалось определить focus_idx для focus_time {focus_time} (символ {symbol}). Используется fallback (последние 100 свечей).")
            df_to_plot = df_1h_full.tail(100)
        
    except Exception as e:
        print(f"Критическая ошибка (charting) при поиске focus_idx для {focus_time} (символ {symbol}): {e}. Используется fallback.")
        _focus_idx_set = False 
        df_to_plot = df_1h_full.tail(100)

    if _focus_idx_set and focus_idx != -1 : 
        plot_start_idx = max(0, focus_idx - 70) 
        plot_end_idx = min(len(df_1h_full) - 1, focus_idx + 50) 
        df_to_plot = df_1h_full.iloc[plot_start_idx : plot_end_idx + 1]
    
    if df_to_plot is None or df_to_plot.empty:
        if df_1h_full.empty: 
             print(f"Ошибка (charting): df_1h_full изначально пуст, df_to_plot не может быть создан для {symbol}.")
        else: 
             print(f"Предупреждение (charting): df_to_plot пуст после всех попыток для {symbol}. Используем последние 100 свечей из df_1h_full если возможно.")
             df_to_plot = df_1h_full.tail(100)
             if df_to_plot.empty:
                  print(f"Критическая ошибка (charting): df_to_plot все еще пуст после fallback для {symbol}. График не будет построен.")
                  return None
    # --- Конец логики ---

    fvg_tested_data = order_flow_signal.get('fvg_tested_data')
    # Эта проверка КЛЮЧЕВАЯ для предотвращения KeyError: 'fvg_time'
    if not isinstance(fvg_tested_data, dict) or 'fvg_time' not in fvg_tested_data:
        print(f"Ошибка (charting): fvg_tested_data некорректен или отсутствует ключ 'fvg_time' для {symbol}.")
        print(f"Содержимое fvg_tested_data: {fvg_tested_data}")
        print(f"Полный order_flow_signal: {order_flow_signal}") # Дополнительный вывод для отладки
        return None 

    try:
        fvg_time = pd.to_datetime(fvg_tested_data['fvg_time']).tz_localize(None)
        fvg_top = fvg_tested_data['fvg_top']
        fvg_bottom = fvg_tested_data['fvg_bottom']
        fvg_type = fvg_tested_data['type']
    except Exception as e:
        print(f"Ошибка (charting) при извлечении данных FVG из fvg_tested_data: {e}")
        return None

    addplot = []
    hlines_config = {'linewidths': 0.9, 'alpha': 0.8}
    fvg_fill_alpha = 0.2

    if entry_details:
        sl_price = entry_details['stop_loss']
        tp_price = entry_details['take_profit']
        entry_price_val = entry_details['entry_price']
        entry_time_val = None
        try:
            entry_time_val = pd.to_datetime(entry_details['entry_time']).tz_localize(None)
        except Exception as e:
            print(f"Предупреждение (charting): Не удалось обработать entry_time для маркера: {entry_details.get('entry_time')}, {e}")

        addplot.append(mpf.make_addplot([sl_price] * len(df_to_plot.index), panel=0, color='red', linestyle='--', **hlines_config, secondary_y=False))
        addplot.append(mpf.make_addplot([tp_price] * len(df_to_plot.index), panel=0, color='green', linestyle='--', **hlines_config, secondary_y=False))
        addplot.append(mpf.make_addplot([entry_price_val] * len(df_to_plot.index), panel=0, color='blue', linestyle=':', **hlines_config, secondary_y=False))
        
        if entry_time_val and entry_time_val in df_to_plot.index:
            entry_marker_data = [float('nan')] * len(df_to_plot.index)
            try:
                entry_idx_plot = df_to_plot.index.get_loc(entry_time_val)
                marker_shape = '^' if of_type == 'bullish_of' else 'v'
                y_pos_factor = 0.998 if of_type == 'bullish_of' else 1.002
                price_level_for_marker = 'low' if of_type == 'bullish_of' else 'high'
                
                if price_level_for_marker in df_to_plot.columns:
                    entry_marker_data[entry_idx_plot] = df_to_plot[price_level_for_marker].iloc[entry_idx_plot] * y_pos_factor
                    addplot.append(mpf.make_addplot(entry_marker_data, type='scatter', marker=marker_shape, color='blue', markersize=120, panel=0, secondary_y=False))
                else:
                    print(f"Предупреждение (charting): Столбец '{price_level_for_marker}' отсутствует в df_to_plot для отметки входа.")
            except KeyError:
                print(f"Предупреждение (charting): Время входа {entry_time_val} не найдено в df_to_plot.index для отметки на графике.")

    fvg_top_line = [float('nan')] * len(df_to_plot.index)
    fvg_bottom_line = [float('nan')] * len(df_to_plot.index)
    fvg_plot_color = 'lightgreen' if fvg_type == 'bullish' else 'lightcoral'

    for i_plot_fvg, time_idx_fvg in enumerate(df_to_plot.index):
        if time_idx_fvg >= fvg_time: 
            fvg_top_line[i_plot_fvg] = fvg_top
            fvg_bottom_line[i_plot_fvg] = fvg_bottom
    
    addplot.append(mpf.make_addplot(fvg_top_line, panel=0, color=fvg_plot_color, linestyle='-.', width=0.7, alpha=0.5, secondary_y=False))
    addplot.append(mpf.make_addplot(fvg_bottom_line, panel=0, color=fvg_plot_color, linestyle='-.', width=0.7, alpha=0.5, secondary_y=False))
    
    liq_sweep_time_dt = pd.NaT
    try:
        liq_sweep_time_dt = pd.to_datetime(order_flow_signal.get('liquidity_sweep_time')).tz_localize(None)
    except Exception as e:
        print(f"Предупреждение (charting): Не удалось обработать liquidity_sweep_time: {order_flow_signal.get('liquidity_sweep_time')}, {e}")

    # Безопасное извлечение значений для заголовка
    liquidity_type_str = order_flow_signal.get('liquidity_type','N/A_type')
    liquidity_sweep_price_val = order_flow_signal.get('liquidity_sweep_price')
    liquidity_sweep_price_str = f"{liquidity_sweep_price_val:.5f}" if isinstance(liquidity_sweep_price_val, (int, float)) else 'N/A_price'
    
    fig_title_parts = [
        f"{symbol} 1H - {of_type.replace('_of', '').capitalize()} OF",
        f"1H Sig: {signal_time_1h.strftime('%y%m%d-%H%M')}",
        f"{liquidity_type_str} Sw: {liquidity_sweep_price_str} @ {liq_sweep_time_dt.strftime('%H%M') if pd.notna(liq_sweep_time_dt) else 'N/A_time'}",
        f"FVG ({fvg_type}): {fvg_bottom:.5f}-{fvg_top:.5f} @ {fvg_time.strftime('%H%M')}"
    ]
    if entry_details and entry_time_val:
        entry_price_str = f"{entry_details['entry_price']:.5f}" if isinstance(entry_details.get('entry_price'), (int,float)) else 'N/A'
        fig_title_parts.append(f"Entry: {entry_price_str} @ {entry_time_val.strftime('%y%m%d-%H%M')}")
    
    fig_title = "\n".join(fig_title_parts)
    filename_suffix = f"{symbol.replace('/', '')}_{of_type}_{get_current_datetime_str()}"
    filename_suffix += "_TRADE" if entry_details else "_SIGNAL"
    filepath = os.path.join(CHART_OUTPUT_DIR, f"{filename_suffix}.png")

    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    fig = None 
    try:
        fig, axlist = mpf.plot(
            df_to_plot, type='candle', style=s, title=fig_title,
            addplot=addplot, volume=False, figsize=(CHART_WIDTH, CHART_HEIGHT),
            returnfig=True, panel_ratios=(1,)
        )
        main_ax = axlist[0]
        
        try:
            if pd.notna(liq_sweep_time_dt) and liq_sweep_time_dt in df_to_plot.index and isinstance(order_flow_signal.get('liquidity_sweep_price'), (int,float)):
                liq_idx_plot = df_to_plot.index.get_loc(liq_sweep_time_dt)
                main_ax.plot(liq_idx_plot, order_flow_signal['liquidity_sweep_price'], 'o', color='orange', markersize=7, label=f"{liquidity_type_str} Lvl")
                main_ax.text(liq_idx_plot, order_flow_signal['liquidity_sweep_price'], f" {liquidity_type_str}", color='orange', va='bottom', ha='left', fontsize=9)
        except Exception as e: print(f"Предупреждение (charting): Ошибка аннотации уровня ликвидности: {e}")

        try:
            actual_sweep_time = pd.to_datetime(order_flow_signal.get('actual_sweep_time')).tz_localize(None)
            liq_type_for_key = order_flow_signal.get('liquidity_type', '') 
            actual_sweep_price_key = 'actual_sweep_low_price' if 'SSL' in liq_type_for_key else 'actual_sweep_high_price'
            actual_sweep_price = order_flow_signal.get(actual_sweep_price_key)
            if isinstance(actual_sweep_price, (int,float)) and pd.notna(actual_sweep_time) and actual_sweep_time in df_to_plot.index:
                actual_sweep_idx_plot = df_to_plot.index.get_loc(actual_sweep_time)
                main_ax.plot(actual_sweep_idx_plot, actual_sweep_price, 'x', color='purple', markersize=7, label=f'Sweep Pt')
        except Exception as e: print(f"Предупреждение (charting): Ошибка аннотации точки свипа: {e}")
        
        # Заливка FVG
        if not df_to_plot.empty: # Дополнительная проверка перед searchsorted
            fvg_start_plot_idx = df_to_plot.index.searchsorted(fvg_time)
            if fvg_start_plot_idx < len(df_to_plot.index) and df_to_plot.index[fvg_start_plot_idx] >= fvg_time:
                 main_ax.fill_between(df_to_plot.index[fvg_start_plot_idx:], fvg_bottom, fvg_top,
                                     color=fvg_plot_color, alpha=fvg_fill_alpha, step='post',
                                     label=f'{fvg_type} FVG')
            elif fvg_time <= df_to_plot.index[-1] : # Проверяем, что FVG не полностью за пределами df_to_plot
                 main_ax.fill_between(df_to_plot.index, fvg_bottom, fvg_top, where=df_to_plot.index >= fvg_time,
                                     color=fvg_plot_color, alpha=fvg_fill_alpha, step='post',
                                     label=f'{fvg_type} FVG (part)')

        main_ax.legend(loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(filepath)
        print(f"График сохранен: {filepath}")
        return filepath
    except Exception as e:
        print(f"Ошибка (charting) при построении или сохранении графика для {symbol}: {e}")
        return None
    finally:
        if fig: 
            plt.close(fig)

if __name__ == '__main__':
    from data_handler import get_historical_data
    from orderflow_analyzer import analyze_order_flow 
    from config import DEFAULT_SYMBOL, TIMEFRAME_1H, TWELVEDATA_API_KEY # Убедитесь, что все импортировано
    from datetime import datetime, timedelta

    ensure_dir(CHART_OUTPUT_DIR) # Убедимся, что директория существует

    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY:
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY в .env файле или в config.py")
    else:
        print("Тестирование charting.py...")
        symbol_test = DEFAULT_SYMBOL
        df_1h_test = get_historical_data(symbol_test, TIMEFRAME_1H, outputsize=350)

        if not df_1h_test.empty:
            print(f"Получено {len(df_1h_test)} 1H свечей для {symbol_test}.")
            order_flow_signals_test = analyze_order_flow(df_1h_test, symbol_test)

            if order_flow_signals_test:
                test_signal_chart = order_flow_signals_test[-1] 
                print(f"Тестовый сигнал 1H: {test_signal_chart.get('type','N/A')} в {test_signal_chart.get('signal_candle_time','N/A')}")
                plot_order_flow_and_trade(df_1h_test, test_signal_chart, None, symbol_test)
                
                # Имитация входа для теста
                if len(df_1h_test) > 20 and isinstance(test_signal_chart.get('fvg_tested_data'), dict) and \
                   isinstance(test_signal_chart.get('stop_loss_level'), (int,float)) and \
                   isinstance(test_signal_chart.get('target_price'), (int,float)):
                    
                    mock_entry_time_chart = pd.to_datetime(test_signal_chart['signal_candle_time']).tz_localize(None) + timedelta(hours=1, minutes=30)
                    
                    if not df_1h_test.index.empty: # Проверка, что индекс не пуст
                        if mock_entry_time_chart > df_1h_test.index[-1]: mock_entry_time_chart = df_1h_test.index[-1]
                        elif mock_entry_time_chart < df_1h_test.index[0]: mock_entry_time_chart = df_1h_test.index[0]
                        else:
                            temp_diffs = (df_1h_test.index - mock_entry_time_chart).to_series().abs()
                            if not temp_diffs.empty: mock_entry_time_chart = temp_diffs.idxmin()
                            else: mock_entry_time_chart = df_1h_test.index[-1] 
                    else:
                        print("Предупреждение (тест charting): df_1h_test.index пуст, не могу установить mock_entry_time_chart.")
                        # mock_entry_time_chart останется как есть или можно установить в None

                    pip_val_test = calculate_pip_value(symbol_test)
                    # Безопасное получение цены закрытия
                    mock_entry_price_chart = None
                    if pd.notna(mock_entry_time_chart) and mock_entry_time_chart in df_1h_test.index:
                         mock_entry_price_chart = df_1h_test.loc[mock_entry_time_chart]['close']
                    elif not df_1h_test.empty:
                         mock_entry_price_chart = df_1h_test.iloc[-1]['close']
                    
                    if mock_entry_price_chart is not None:
                        mock_sl_chart = test_signal_chart['stop_loss_level']
                        mock_tp_chart = test_signal_chart['target_price']

                        if test_signal_chart['type'] == 'bullish_of':
                            if mock_entry_price_chart <= mock_sl_chart: mock_sl_chart = mock_entry_price_chart - 15 * pip_val_test
                            if mock_tp_chart <= mock_entry_price_chart: mock_tp_chart = mock_entry_price_chart + 30 * pip_val_test
                        else: 
                            if mock_entry_price_chart >= mock_sl_chart: mock_sl_chart = mock_entry_price_chart + 15 * pip_val_test
                            if mock_tp_chart >= mock_entry_price_chart: mock_tp_chart = mock_entry_price_chart - 30 * pip_val_test

                        mock_entry_details_chart = {
                            'entry_price': mock_entry_price_chart,
                            'entry_time': mock_entry_time_chart,
                            'stop_loss': mock_sl_chart,
                            'take_profit': mock_tp_chart
                        }
                        print("\nГрафик с 1H сигналом и имитацией входа:")
                        plot_order_flow_and_trade(df_1h_test, test_signal_chart, mock_entry_details_chart, symbol_test)
                    else:
                        print("Предупреждение (тест charting): Не удалось определить mock_entry_price_chart.")
            else:
                print("Order Flow сигналы не найдены, тестовый график для charting.py не будет построен.")
        else:
            print(f"Не удалось получить данные для {symbol_test} для теста charting.")

