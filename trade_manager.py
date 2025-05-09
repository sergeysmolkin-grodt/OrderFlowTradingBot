# forex_orderflow_bot/trade_manager.py

from utils import calculate_pip_value

def validate_trade(symbol, direction, entry_price, stop_loss, take_profit):
    """
    Проверяет жизнеспособность сделки (например, TP/SL > минимального расстояния, положительное R:R).
    Возвращает True, если сделка валидна, иначе False.
    """
    if entry_price is None or stop_loss is None or take_profit is None:
        return False

    pip_val = calculate_pip_value(symbol)
    min_distance_pips = 5 # Минимальное расстояние SL/TP от входа в пипсах
    min_reward_risk_ratio = 1.0 # Минимальное соотношение прибыль/риск

    if direction == 'bullish_of':
        if entry_price <= stop_loss: # Вход ниже или на уровне стопа
            print(f"Ошибка валидации (бычья): Цена входа {entry_price} <= Стоп Лосс {stop_loss}")
            return False
        if take_profit <= entry_price: # Тейк ниже или на уровне входа
            print(f"Ошибка валидации (бычья): Тейк Профит {take_profit} <= Цена входа {entry_price}")
            return False
        
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit - entry_price

        if risk_amount < min_distance_pips * pip_val:
            print(f"Ошибка валидации (бычья): Слишком близкий стоп-лосс ({risk_amount/pip_val:.1f} пипсов)")
            return False
        if reward_amount < min_distance_pips * pip_val:
            print(f"Ошибка валидации (бычья): Слишком близкий тейк-профит ({reward_amount/pip_val:.1f} пипсов)")
            return False
        
        if risk_amount <= 0: return False # Избегаем деления на ноль
        reward_risk_ratio = reward_amount / risk_amount
        if reward_risk_ratio < min_reward_risk_ratio:
            print(f"Ошибка валидации (бычья): Низкое R:R = {reward_risk_ratio:.2f}")
            return False

    elif direction == 'bearish_of':
        if entry_price >= stop_loss:
            print(f"Ошибка валидации (медвежья): Цена входа {entry_price} >= Стоп Лосс {stop_loss}")
            return False
        if take_profit >= entry_price:
            print(f"Ошибка валидации (медвежья): Тейк Профит {take_profit} >= Цена входа {entry_price}")
            return False

        risk_amount = stop_loss - entry_price
        reward_amount = entry_price - take_profit

        if risk_amount < min_distance_pips * pip_val:
            print(f"Ошибка валидации (медвежья): Слишком близкий стоп-лосс ({risk_amount/pip_val:.1f} пипсов)")
            return False
        if reward_amount < min_distance_pips * pip_val:
            print(f"Ошибка валидации (медвежья): Слишком близкий тейк-профит ({reward_amount/pip_val:.1f} пипсов)")
            return False

        if risk_amount <= 0: return False
        reward_risk_ratio = reward_amount / risk_amount
        if reward_risk_ratio < min_reward_risk_ratio:
            print(f"Ошибка валидации (медвежья): Низкое R:R = {reward_risk_ratio:.2f}")
            return False
    else:
        return False # Неизвестное направление

    return True

if __name__ == '__main__':
    print("Тестирование trade_manager.py...")
    symbol = "EUR/USD"

    # Тест бычьей сделки
    print("\nТест валидной бычьей сделки:")
    valid_bull = validate_trade(symbol, 'bullish_of', 1.08500, 1.08300, 1.09000) # R:R = 2.5
    print(f"Результат: {valid_bull}")

    print("\nТест невалидной бычьей сделки (плохой R:R):")
    invalid_bull_rr = validate_trade(symbol, 'bullish_of', 1.08500, 1.08000, 1.08700) # R:R = 0.4
    print(f"Результат: {invalid_bull_rr}")
    
    print("\nТест невалидной бычьей сделки (TP ниже входа):")
    invalid_bull_tp = validate_trade(symbol, 'bullish_of', 1.08500, 1.08300, 1.08400)
    print(f"Результат: {invalid_bull_tp}")

    # Тест медвежьей сделки
    print("\nТест валидной медвежьей сделки:")
    valid_bear = validate_trade(symbol, 'bearish_of', 1.08500, 1.08700, 1.08000) # R:R = 2.5
    print(f"Результат: {valid_bear}")

    print("\nТест невалидной медвежьей сделки (SL слишком близко):")
    pip_val = calculate_pip_value(symbol)
    invalid_bear_sl = validate_trade(symbol, 'bearish_of', 1.08500, 1.08520, 1.08000) # SL 2 пипса
    print(f"Результат: {invalid_bear_sl}")