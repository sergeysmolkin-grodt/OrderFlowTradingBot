# forex_orderflow_bot/charting.py

import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import os # Убедитесь, что os импортирован

from config import CHART_OUTPUT_DIR, CHART_WIDTH, CHART_HEIGHT
from utils import ensure_dir, get_current_datetime_str, calculate_pip_value

def plot_order_flow_and_trade(df_1h_full, order_flow_signal, entry_details, symbol):
    """
    Рисует график 1H с Order Flow и деталями входа/SL/TP.
    df_1h_full: DataFrame с данными 1H, достаточными для отображения контекста.
    order_flow_signal: Словарь с информацией о 1H сигнале.
    entry_details: Словарь {'entry_price': float, 'entry_time': datetime, 'stop_loss': float, 'take_profit': float}
                   или None, если входа не было.
    """
    ensure_dir(CHART_OUTPUT_DIR)

    if df_1h_full.empty:
        print("Ошибка: df_1h_full пуст. График не будет построен.")
        return None
    if order_flow_signal is None:
        print("Ошибка: order_flow_signal is None. График не будет построен.")
        return None


    of_type = order_flow_signal['type']
    signal_time_1h = pd.to_datetime(order_flow_signal['signal_candle_time'])

    # Определяем фокусное время для графика
    focus_time = signal_time_1h
    if entry_details and entry_details.get('entry_time'):
        entry_time_dt = pd.to_datetime(entry_details['entry_time'])
        if entry_time_dt > signal_time_1h:
            focus_time = entry_time_dt

    # --- Логика определения focus_idx и df_to_plot ---
    df_to_plot = None
    _focus_idx_set = False

    try:
        # Убедимся, что индекс отсортирован, если он существует и не пуст
        if not df_1h_full.index.is_monotonic_increasing and not df_1h_full.empty:
            df_1h_full = df_1h_full.sort_index()

        if not isinstance(df_1h_full.index, pd.DatetimeIndex) or df_1h_full.index.empty:
            if df_1h_full.index.empty:
                print(f"Предупреждение: Индекс в df_1h_full пуст для символа {symbol}.")
            else:
                print(f"Предупреждение: Индекс в df_1h_full ({type(df_1h_full.index)}) не является DatetimeIndex для {symbol}.")
            try:
                # Попытка получить loc, если это возможно, иначе fallback
                # Эта ветка менее вероятна, если данные приходят от data_handler
                focus_idx = df_1h_full.index.get_loc(focus_time)
                _focus_idx_set = True
            except Exception:
                pass # _focus_idx_set останется False
        else: # Индекс является DatetimeIndex и не пуст
            focus_time_ts = pd.to_datetime(focus_time)

            if focus_time_ts in df_1h_full.index:
                focus_idx = df_1h_full.index.get_loc(focus_time_ts)
                _focus_idx_set = True
            else:
                # Альтернатива для method='nearest' для старых версий Pandas
                time_diffs = (df_1h_full.index - focus_time_ts).to_series().abs()
                if time_diffs.empty:
                    print(f"Ошибка: time_diffs пуст (индекс df_1h_full не пуст, но разница не вычислена). Focus time: {focus_time_ts} для {symbol}.")
                    # _focus_idx_set останется False
                else:
                    closest_time = time_diffs.idxmin()
                    focus_idx = df_1h_full.index.get_loc(closest_time)
                    _focus_idx_set = True
        
        if not _focus_idx_set:
            print(f"Предупреждение: Не удалось определить focus_idx для focus_time {focus_time} (символ {symbol}). Используется fallback (последние 100 свечей).")
            df_to_plot = df_1h_full.tail(100)
            # focus_idx для fallback (если нужен для среза, но df_to_plot уже определен)
            # focus_idx = len(df_1h_full) - 50 if len(df_1h_full) >= 50 else 0
        
    except Exception as e:
        print(f"Критическая ошибка при поиске focus_idx для {focus_time} (символ {symbol}): {e}. Используется fallback.")
        _focus_idx_set = False
        df_to_plot = df_1h_full.tail(100)

    if _focus_idx_set:
        plot_start_idx = max(0, focus_idx - 70) # Немного больше контекста слева
        plot_end_idx = min(len(df_1h_full) - 1, focus_idx + 50) # Немного больше контекста справа
        df_to_plot = df_1h_full.iloc[plot_start_idx : plot_end_idx + 1]
    
    if df_to_plot is None or df_to_plot.empty:
        print(f"Ошибка: df_to_plot пуст после всех попыток определения диапазона для {symbol}. График не будет построен.")
        return None
    # --- Конец логики определения focus_idx и df_to_plot ---


    fvg_data = order_flow_signal['fvg_tested_data']
    fvg_time = pd.to_datetime(fvg_data['fvg_time'])
    fvg_top = fvg_data['fvg_top']
    fvg_bottom = fvg_data['fvg_bottom']
    fvg_type = fvg_data['type']

    addplot = []
    hlines_config = {'linewidths': 0.9, 'alpha': 0.8}
    fvg_fill_alpha = 0.2

    if entry_details:
        sl_price = entry_details['stop_loss']
        tp_price = entry_details['take_profit']
        entry_price_val = entry_details['entry_price']
        entry_time_val = pd.to_datetime(entry_details['entry_time'])

        # Линии для SL, TP, Entry
        sl_line_data = [sl_price] * len(df_to_plot.index)
        tp_line_data = [tp_price] * len(df_to_plot.index)
        entry_line_data = [entry_price_val] * len(df_to_plot.index)

        addplot.append(mpf.make_addplot(sl_line_data, panel=0, color='red', linestyle='--', **hlines_config, secondary_y=False))
        addplot.append(mpf.make_addplot(tp_line_data, panel=0, color='green', linestyle='--', **hlines_config, secondary_y=False))
        addplot.append(mpf.make_addplot(entry_line_data, panel=0, color='blue', linestyle=':', **hlines_config, secondary_y=False))
        
        if entry_time_val in df_to_plot.index:
            entry_marker_data = [float('nan')] * len(df_to_plot.index)
            try:
                entry_idx_plot = df_to_plot.index.get_loc(entry_time_val)
                marker_shape = '^' if of_type == 'bullish_of' else 'v'
                marker_color = 'blue'
                y_pos_factor = 0.998 if of_type == 'bullish_of' else 1.002
                price_level_for_marker = 'low' if of_type == 'bullish_of' else 'high'
                
                # Проверка на наличие столбца low/high
                if price_level_for_marker in df_to_plot.columns:
                    entry_marker_data[entry_idx_plot] = df_to_plot[price_level_for_marker].iloc[entry_idx_plot] * y_pos_factor
                    addplot.append(mpf.make_addplot(entry_marker_data, type='scatter', marker=marker_shape, color=marker_color, markersize=120, panel=0, secondary_y=False))
                else:
                    print(f"Предупреждение: Столбец '{price_level_for_marker}' отсутствует в df_to_plot для отметки входа.")

            except KeyError:
                print(f"Предупреждение: Время входа {entry_time_val} не найдено в df_to_plot.index для отметки на графике.")


    fvg_top_line = [float('nan')] * len(df_to_plot.index)
    fvg_bottom_line = [float('nan')] * len(df_to_plot.index)
    fvg_plot_color = 'lightgreen' if fvg_type == 'bullish' else 'lightcoral'

    for i, time_idx in enumerate(df_to_plot.index):
        if time_idx >= fvg_time:
            fvg_top_line[i] = fvg_top
            fvg_bottom_line[i] = fvg_bottom
    
    addplot.append(mpf.make_addplot(fvg_top_line, panel=0, color=fvg_plot_color, linestyle='-.', width=0.7, alpha=0.5, secondary_y=False))
    addplot.append(mpf.make_addplot(fvg_bottom_line, panel=0, color=fvg_plot_color, linestyle='-.', width=0.7, alpha=0.5, secondary_y=False))
    
    fig_title_parts = [
        f"{symbol} 1H - {of_type.replace('_of', '').capitalize()} OrderFlow",
        f"1H Signal: {signal_time_1h.strftime('%Y-%m-%d %H:%M')}",
        f"{order_flow_signal['liquidity_type']} Sweep: {order_flow_signal['liquidity_sweep_price']:.5f} @ {pd.to_datetime(order_flow_signal['liquidity_sweep_time']).strftime('%H:%M')}",
        f"FVG ({fvg_type}): {fvg_bottom:.5f}-{fvg_top:.5f} @ {fvg_time.strftime('%H:%M')}"
    ]
    if entry_details:
        fig_title_parts.append(f"Entry: {entry_details['entry_price']:.5f} @ {pd.to_datetime(entry_details['entry_time']).strftime('%Y-%m-%d %H:%M')}")
    
    fig_title = "\n".join(fig_title_parts)

    filename_suffix = f"{symbol.replace('/', '')}_{of_type}_{get_current_datetime_str()}"
    if entry_details:
        filename_suffix += "_TRADE"
    else:
        filename_suffix += "_SIGNAL"
    filepath = os.path.join(CHART_OUTPUT_DIR, f"{filename_suffix}.png")

    mc = mpf.make_marketcolors(up='green', down='red', inherit=True, edge='inherit', wick='inherit', volume='inherit')
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

    fig = None # Инициализируем fig
    try:
        fig, axlist = mpf.plot(
            df_to_plot,
            type='candle',
            style=s,
            title=fig_title,
            addplot=addplot,
            volume=False,
            figsize=(CHART_WIDTH, CHART_HEIGHT),
            returnfig=True,
            panel_ratios=(1,0.1) if 'volume' in df_to_plot.columns and False else (1,) # Пример для объема, если он есть
        )

        main_ax = axlist[0]
        
        liq_time = pd.to_datetime(order_flow_signal['liquidity_sweep_time'])
        liq_price = order_flow_signal['liquidity_sweep_price']
        liq_type = order_flow_signal['liquidity_type']
        
        # Аннотация для уровня снятия ликвидности (целевой уровень ликвидности)
        if liq_time in df_to_plot.index:
            try:
                liq_idx_plot = df_to_plot.index.get_loc(liq_time)
                main_ax.plot(liq_idx_plot, liq_price, 'o', color='orange', markersize=7, label=f'{liq_type} Level')
                main_ax.text(liq_idx_plot, liq_price, f' {liq_type}', color='orange', va='bottom', ha='left', fontsize=9)
            except KeyError:
                 print(f"Предупреждение: Время ликвидности {liq_time} не найдено в df_to_plot.index для аннотации.")


        # Аннотация для фактической точки снятия (свеча-свип)
        actual_sweep_time = pd.to_datetime(order_flow_signal['actual_sweep_time'])
        actual_sweep_price_key = 'actual_sweep_low_price' if liq_type == 'SSL' else 'actual_sweep_high_price'
        actual_sweep_price = order_flow_signal.get(actual_sweep_price_key)

        if actual_sweep_price and actual_sweep_time in df_to_plot.index:
            try:
                actual_sweep_idx_plot = df_to_plot.index.get_loc(actual_sweep_time)
                main_ax.plot(actual_sweep_idx_plot, actual_sweep_price, 'x', color='purple', markersize=7, label=f'Sweep Point')
            except KeyError:
                print(f"Предупреждение: Время фактического свипа {actual_sweep_time} не найдено в df_to_plot.index для аннотации.")


        # Заливка FVG
        fvg_start_plot_idx = -1
        for i_plot, time_val_plot in enumerate(df_to_plot.index):
            if time_val_plot >= fvg_time:
                if fvg_start_plot_idx == -1: fvg_start_plot_idx = i_plot
        
        if fvg_start_plot_idx != -1:
            main_ax.fill_between(df_to_plot.index[fvg_start_plot_idx:], fvg_bottom, fvg_top,
                                 color=fvg_plot_color, alpha=fvg_fill_alpha, step='post',
                                 label=f'{fvg_type} FVG')
        
        main_ax.legend(loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место для заголовка и нижних элементов

        fig.savefig(filepath)
        print(f"График сохранен: {filepath}")
        return filepath
    except Exception as e:
        print(f"Ошибка при построении или сохранении графика для {symbol}: {e}")
        return None
    finally:
        if fig: # Закрываем фигуру, чтобы освободить память
            plt.close(fig)


if __name__ == '__main__':
    from data_handler import get_historical_data
    from orderflow_analyzer import analyze_order_flow 
    from config import DEFAULT_SYMBOL, TIMEFRAME_1H, TWELVEDATA_API_KEY
    from datetime import datetime, timedelta

    # Создаем директорию для чартов, если ее нет
    ensure_dir(CHART_OUTPUT_DIR)

    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY:
        print("Ошибка: Пожалуйста, установите ваш TWELVEDATA_API_KEY в .env файле или в config.py")
    else:
        print("Тестирование charting.py...")
        symbol_test = DEFAULT_SYMBOL
        df_1h_test = get_historical_data(symbol_test, TIMEFRAME_1H, outputsize=350) # Больше данных для контекста

        if not df_1h_test.empty:
            print(f"Получено {len(df_1h_test)} 1H свечей для {symbol_test}.")
            order_flow_signals_test = analyze_order_flow(df_1h_test, symbol_test)

            if order_flow_signals_test:
                test_signal_chart = order_flow_signals_test[-1] 
                print(f"Тестовый сигнал 1H: {test_signal_chart['type']} в {test_signal_chart['signal_candle_time']}")

                print("\nГрафик только с 1H сигналом:")
                plot_order_flow_and_trade(df_1h_test, test_signal_chart, None, symbol_test)

                if len(df_1h_test) > 20:
                    mock_entry_time_chart = pd.to_datetime(test_signal_chart['signal_candle_time']) + timedelta(hours=1, minutes=30)
                    
                    # Убедимся, что mock_entry_time_chart находится в пределах df_1h_test.index
                    # Если нет, берем ближайшее доступное время из df_1h_test для имитации
                    if mock_entry_time_chart > df_1h_test.index[-1]:
                        mock_entry_time_chart = df_1h_test.index[-1]
                    elif mock_entry_time_chart < df_1h_test.index[0]:
                         mock_entry_time_chart = df_1h_test.index[0]
                    else:
                        # Ищем ближайший индекс, если точного совпадения нет (хотя для теста это может быть излишним)
                        temp_diffs = (df_1h_test.index - mock_entry_time_chart).to_series().abs()
                        mock_entry_time_chart = temp_diffs.idxmin()


                    pip_val_test = calculate_pip_value(symbol_test)
                    mock_entry_price_chart = df_1h_test.loc[mock_entry_time_chart]['close'] if mock_entry_time_chart in df_1h_test.index else df_1h_test.iloc[-1]['close']
                    
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
                    print("Недостаточно данных для имитации входа на графике в тесте.")
            else:
                print("Order Flow сигналы не найдены, тестовый график для charting.py не будет построен.")
        else:
            print(f"Не удалось получить данные для {symbol_test} для теста charting.")