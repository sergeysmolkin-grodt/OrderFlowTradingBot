# forex_orderflow_bot/main.py
# forex_orderflow_bot/main.py

import time
from datetime import datetime, timedelta
import pandas as pd
import os # <<< ДОБАВЛЕНО
import shutil # <<< ДОБАВЛЕНО
import atexit # <<< ДОБАВЛЕНО
import sys # <<< ДОБАВЛЕНО для Варианта 2, если выберете его

sys.dont_write_bytecode = True 

from config import (
    DEFAULT_SYMBOL,
    TIMEFRAME_1H,
    TIMEFRAME_ENTRY,
    TWELVEDATA_API_KEY,
    CHART_OUTPUT_DIR,
    CHoCH_LOOKBACK_ENTRY_TF
)
from data_handler import get_historical_data
from order_flow_analyzer import analyze_order_flow
from entry_logic import identify_break_of_structure
from trade_manager import validate_trade
from charting import plot_order_flow_and_trade
from utils import ensure_dir, print_trade_info, get_current_datetime_str


def cleanup_pycache():
    """
    Находит и удаляет папки __pycache__ в директории запущенного скрипта
    и в текущей рабочей директории.
    """
    print("Запущена очистка папок __pycache__...")
    deleted_count = 0
    
    script_dir_to_check = None
    try:
        # Используем sys.argv[0] для получения пути к запущенному скрипту
        # os.path.abspath нужен, если sys.argv[0] - относительный путь
        main_script_path = os.path.abspath(sys.argv[0])
        script_dir_to_check = os.path.dirname(main_script_path)
    except Exception as e:
        # В редких случаях sys.argv[0] может быть недоступен или некорректен
        print(f"Предупреждение: Не удалось определить директорию скрипта через sys.argv[0]: {e}.")
        # script_dir_to_check останется None

    cwd = os.getcwd() # Текущая рабочая директория
    
    dirs_to_check = {cwd} # Начинаем с текущей рабочей директории
    if script_dir_to_check and script_dir_to_check != cwd:
        dirs_to_check.add(script_dir_to_check) # Добавляем директорию скрипта, если она отличается

    for check_dir in dirs_to_check:
        if not check_dir: # Пропускаем, если путь не определен
            continue
            
        pycache_dir = os.path.join(check_dir, "__pycache__")
        if os.path.exists(pycache_dir) and os.path.isdir(pycache_dir):
            try:
                shutil.rmtree(pycache_dir)
                print(f"Папка {pycache_dir} успешно удалена.")
                deleted_count += 1
            except OSError as e:
                print(f"Ошибка при удалении папки {pycache_dir}: {e}")
        # Раскомментируйте для отладки, если папки не находятся:
        # else:
        #     print(f"Папка __pycache__ не найдена в директории: {check_dir}")


    if deleted_count == 0:
        # Проверим также поддиректории текущей рабочей директории (один уровень)
        # Это полезно, если __pycache__ создаются для модулей в подпапках,
        # хотя в текущей структуре проекта все файлы на одном уровне.
        # Для более глубокого поиска можно использовать os.walk.
        found_in_subdir = False
        for item in os.listdir(cwd):
            item_path = os.path.join(cwd, item)
            if os.path.isdir(item_path):
                pycache_subdir = os.path.join(item_path, "__pycache__")
                if os.path.exists(pycache_subdir) and os.path.isdir(pycache_subdir):
                    try:
                        shutil.rmtree(pycache_subdir)
                        print(f"Папка {pycache_subdir} (в поддиректории) успешно удалена.")
                        deleted_count +=1
                        found_in_subdir = True
                    except OSError as e:
                        print(f"Ошибка при удалении папки {pycache_subdir}: {e}")
        if not found_in_subdir and deleted_count == 0:
             print("Папки __pycache__ не найдены для удаления в проверенных директориях.")


# Регистрируем функцию очистки, чтобы она вызывалась при выходе
atexit.register(cleanup_pycache)
# --- Конец функции для очистки ---


def run_bot(symbol=DEFAULT_SYMBOL):
    """Основная функция для запуска бота."""
    print(f"Запуск бота для символа: {symbol} в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ensure_dir(CHART_OUTPUT_DIR)

    if TWELVEDATA_API_KEY == "YOUR_TWELVEDATA_API_KEY_FALLBACK" or not TWELVEDATA_API_KEY:
        print("КРИТИЧЕСКАЯ ОШИБКА: API ключ для TwelveData не установлен. Проверьте .env файл или config.py.")
        print("Бот не может продолжить работу без API ключа.")
        return

    print(f"Получение {TIMEFRAME_1H} данных для {symbol}...")
    df_1h = get_historical_data(symbol, TIMEFRAME_1H, outputsize=500)
    if df_1h.empty:
        print(f"Не удалось получить {TIMEFRAME_1H} данные. Бот завершает текущий цикл.")
        return

    print("Анализ Order Flow на 1H...")
    order_flow_signals = analyze_order_flow(df_1h, symbol)

    if not order_flow_signals:
        print("На данный момент нет активных Order Flow сигналов на 1H.")
    else:
        latest_1h_signal = order_flow_signals[-1]
        print(f"\nНайден потенциальный 1H Order Flow сигнал:")
        print(f"  Тип: {latest_1h_signal['type']}")
        print(f"  Время сигнала 1H: {latest_1h_signal['signal_candle_time']}")
        print(f"  Предполагаемый SL: {latest_1h_signal['stop_loss_level']:.5f}")
        print(f"  Предполагаемый TP: {latest_1h_signal['target_price']:.5f}")
        print(f"  Детали FVG: {latest_1h_signal['fvg_tested_data']['type']} с {latest_1h_signal['fvg_tested_data']['fvg_bottom']:.5f} по {latest_1h_signal['fvg_tested_data']['fvg_top']:.5f}")

        # <<< ИЗМЕНЕНО: TIMEFRAME_3M на TIMEFRAME_ENTRY
        print(f"\nПолучение {TIMEFRAME_ENTRY} данных для {symbol} для поиска точки входа...")

        # Определяем, с какой даты начать загрузку данных для входа, чтобы покрыть время после 1H сигнала
        # Добавим буфер перед сигналом для истории структуры на таймфрейме входа
        # Рассчитываем множитель для timedelta на основе интервала (например, 5 для '5min')
        interval_minutes = int(TIMEFRAME_ENTRY.replace('min', ''))
        # <<< ИЗМЕНЕНО: CHoCH_LOOKBACK_3M на CHoCH_LOOKBACK_ENTRY_TF
        start_date_entry_tf_needed = latest_1h_signal['signal_candle_time'] - timedelta(minutes=interval_minutes * CHoCH_LOOKBACK_ENTRY_TF)

        # Загружаем данные для таймфрейма входа
        # outputsize 250 свечей 5min = 1250 минут = ~20 часов. Этого должно быть достаточно.
        # <<< ИЗМЕНЕНО: TIMEFRAME_3M на TIMEFRAME_ENTRY и df_3m на df_entry_tf
        df_entry_tf = get_historical_data(symbol, TIMEFRAME_ENTRY, outputsize=250)

        if df_entry_tf.empty:
            # <<< ИЗМЕНЕНО: TIMEFRAME_3M на TIMEFRAME_ENTRY
            print(f"Не удалось получить {TIMEFRAME_ENTRY} данные. Невозможно проверить вход.")
        else:
            # <<< ИЗМЕНЕНО: "3M" на TIMEFRAME_ENTRY
            print(f"Поиск слома структуры на {TIMEFRAME_ENTRY} для подтверждения входа...")
            # <<< ИЗМЕНЕНО: df_3m на df_entry_tf
            entry_price, entry_time, _ = identify_break_of_structure(
                df_entry_tf,
                latest_1h_signal['type'],
                latest_1h_signal['signal_candle_time']
            )

            if entry_price and entry_time:
                stop_loss = latest_1h_signal['stop_loss_level']
                take_profit = latest_1h_signal['target_price']
                direction = latest_1h_signal['type']

                if validate_trade(symbol, direction, entry_price, stop_loss, take_profit):
                    print_trade_info(
                        symbol, direction, entry_price, stop_loss, take_profit,
                        latest_1h_signal['signal_candle_time'], entry_time
                    )
                    entry_details_for_plot = {
                        'entry_price': entry_price,
                        'entry_time': entry_time,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    chart_filepath = plot_order_flow_and_trade(df_1h, latest_1h_signal, entry_details_for_plot, symbol)
                    if chart_filepath:
                        print(f"График сделки сохранен: {chart_filepath}")
                    else:
                        print("Не удалось сохранить график сделки.")
                    print(">>> Потенциальная сделка найдена и соответствует критериям. <<<\n")
                else:
                    print(f"Сделка на {direction} для {symbol} в {entry_time} по цене {entry_price:.5f} не прошла валидацию (SL/TP/RR).")
                    entry_details_for_plot = {
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'stop_loss': stop_loss, 'take_profit': take_profit
                    }
                    plot_order_flow_and_trade(df_1h, latest_1h_signal, entry_details_for_plot, symbol)
            else:
                # <<< ИЗМЕНЕНО: TIMEFRAME_3M на TIMEFRAME_ENTRY
                print(f"На {TIMEFRAME_ENTRY} не найден подходящий слом структуры после 1H сигнала ({latest_1h_signal['signal_candle_time']}).")

    print(f"Бот завершил цикл для {symbol} в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")



if __name__ == "__main__":
    # Пример запуска для одной пары
    run_bot(symbol="EUR/USD")

    # Если вы хотите запускать для нескольких пар или по расписанию:
    # trading_symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
    # run_interval_seconds = 300 # Каждые 5 минут

    # while True:
    #     print(f"--- Новый цикл проверки ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    #     for sym in trading_symbols:
    #         try:
    #             run_bot(symbol=sym)
    #         except Exception as e:
    #             print(f"Критическая ошибка при обработке символа {sym}: {e}")
    #             # Здесь можно добавить логирование ошибок в файл
    #     print(f"--- Цикл завершен. Следующий запуск через {run_interval_seconds} секунд ---")
    #     time.sleep(run_interval_seconds)