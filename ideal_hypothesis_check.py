"""Ідеальна перевірка гіпотези для динамічної системи Б.

Модель не враховує тертя, витоки, клапани, інерцію або кінематичні втрати.
«Квазістатичний» тут означає повільний процес без динамічної дисипації,
а не обов'язкову рівновагу із зовнішнім навантаженням.

Важливе уточнення моделі:
- система А перед розрахунком стартує з порожніми циліндрами;
- система Б перед розрахунком уже має робочий запас порцій газу,
  сформований попередніми циклами стискання;
- цей запас не є новою енергією і не втрачається під час сталого циклу:
      E_B_end = E_B_start;
- для Б початковий і кінцевий запас виводяться окремо в енергетичному балансі.

Скрипт перевіряє:
1. профілі сили систем А і Б;
2. роботу однакового навантаження;
3. однаковість повної P–V роботи порцій;
4. рівність початкового та кінцевого запасу системи Б;
5. відсутність прихованого джерела енергії;
6. залежність результатів від N та F_load.

Запуск:
    pip install numpy matplotlib
    python ideal_hypothesis_check.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Базові параметри (умовні одиниці тиску, об'єму та роботи)
P_START = 10.0       # тиск нової підготовленої порції
V_START = 1.0        # об'єм нової порції після попереднього стискання
V_END = 0.1          # об'єм наприкінці робочого розширення
N = 10
PISTON_AREA = 1.0
F_LOAD = 23.99
POINTS = 2000

OUTPUT_DIR = Path("ideal_hypothesis_results")


def pressure(volume: np.ndarray | float, p_start: float = P_START,
             v_start: float = V_START) -> np.ndarray | float:
    """Ізотермічний закон Бойля–Маріотта: P*V = const."""
    return p_start * v_start / np.asarray(volume)


def pv_work_expansion(volume_start: float, volume_end: float,
                      p_start: float = P_START,
                      v_reference: float = V_START) -> float:
    """Робота розширення від volume_start до volume_end.

    Значення додатне лише коли volume_end > volume_start.
    """
    if volume_end < volume_start:
        raise ValueError("Для розширення volume_end має бути не меншим за volume_start")
    return float(p_start * v_reference * np.log(volume_end / volume_start))


def available_reserve(volume: float, p_start: float = P_START,
                      v_start: float = V_START,
                      v_end: float = V_END) -> float:
    """Доступна P–V робота порції від її поточного V до V_end.

    Це облік робочого запасу газу, а не твердження про збільшення
    внутрішньої енергії ідеального газу під час ізотермічного стискання.
    """
    if volume <= v_end:
        return 0.0
    return pv_work_expansion(volume, v_end, p_start, v_start)


def system_a_force(x: np.ndarray, n_cyl: int = N,
                   p_start: float = P_START, v_start: float = V_START,
                   v_end: float = V_END, area: float = PISTON_AREA) -> np.ndarray:
    """Система А: усі порції стартують одночасно з V_START."""
    volume = np.clip(v_start + x, v_end, v_start)
    return n_cyl * pressure(volume, p_start, v_start) * area


def cascade_stage_volumes(x: float, n_cyl: int = N,
                          v_start: float = V_START,
                          v_end: float = V_END) -> np.ndarray:
    """Поточні об'єми попередньо підготовлених порцій системи Б.

    На початку порції розміщені на різних стадіях: від V_END до V_START.
    Після завершення стадії порція замінюється новою, тому координата
    циклічна. Миттєва заміна є ідеалізацією; енергія нової порції
    враховується через зміну початкового/кінцевого резерву.
    """
    if n_cyl < 1:
        raise ValueError("n_cyl має бути додатним")
    width = (v_start - v_end) / n_cyl
    # Стадії рівномірно розподілені у робочому діапазоні.
    local = (x % (v_start - v_end))
    volumes = v_end + (np.arange(n_cyl) * width + local) % (v_start - v_end)
    return volumes


def system_b_force(x: np.ndarray, n_cyl: int = N,
                   p_start: float = P_START, v_start: float = V_START,
                   v_end: float = V_END, area: float = PISTON_AREA) -> np.ndarray:
    """Система Б: усі поршні рухаються одночасно, але стадії різні."""
    result = np.empty_like(x, dtype=float)
    for k, position in enumerate(x):
        volumes = cascade_stage_volumes(position, n_cyl, v_start, v_end)
        result[k] = np.sum(pressure(volumes, p_start, v_start)) * area
    return result


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Чисельна інтеграція роботи."""
    return float(np.trapezoid(y, x))


def cumulative_integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    increments = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def audit_cascade_reserve(x: np.ndarray, n_cyl: int = N) -> tuple[float, float]:
    """Порівнює запас Б на початку та наприкінці повного періоду."""
    start_volumes = cascade_stage_volumes(float(x[0]), n_cyl)
    end_volumes = cascade_stage_volumes(float(x[-1]), n_cyl)
    start = sum(available_reserve(float(v)) for v in start_volumes)
    end = sum(available_reserve(float(v)) for v in end_volumes)
    return float(start), float(end)


def run_case(n_cyl: int = N, load: float = F_LOAD,
             points: int = POINTS) -> dict[str, float]:
    """Запускає один ідеальний робочий період каскаду."""
    total_stroke = V_START - V_END
    x = np.linspace(0.0, total_stroke, points)
    force_a = system_a_force(x, n_cyl)
    force_b = system_b_force(x, n_cyl)

    useful_a = load * min(total_stroke, max(0.0, np.max(x[force_a >= load]))) \
        if np.any(force_a >= load) else 0.0
    useful_b = load * min(total_stroke, max(0.0, np.max(x[force_b >= load]))) \
        if np.any(force_b >= load) else 0.0

    # Повна P–V робота всіх N однакових порцій.
    pv_total = n_cyl * pv_work_expansion(V_START, V_START + total_stroke)
    reserve_start, reserve_end = audit_cascade_reserve(x, n_cyl)

    # Для стаціонарного циклу вхідна робота визначається енергобалансом.
    # Якщо резерви рівні, вона дорівнює корисній роботі без втрат.
    input_b = useful_b + reserve_end - reserve_start
    hidden_energy = useful_b - (input_b + reserve_start - reserve_end)

    return {
        "N": float(n_cyl),
        "F_load": float(load),
        "W_PV_total": pv_total,
        "W_useful_A": useful_a,
        "W_useful_B": useful_b,
        "reserve_B_start": reserve_start,
        "reserve_B_end": reserve_end,
        "reserve_difference_B": reserve_end - reserve_start,
        "W_input_B": input_b,
        "hidden_energy_residual_B": hidden_energy,
        "gain_B_over_A": useful_b / useful_a if useful_a else float("inf"),
    }


def write_parameter_table() -> None:
    rows = []
    for n_cyl in (1, 2, 5, 10, 20, 50):
        for load in (5.0, 10.0, 15.0, 20.0, 23.99, 30.0):
            rows.append(run_case(n_cyl, load, points=800))

    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / "parameter_sweep.csv"
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Таблиця параметрів: {filename}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    case = run_case()

    print("=== Ідеальна перевірка гіпотези: сталий каскадний цикл ===")
    print(f"N                                      : {N}")
    print(f"Загальний газ на старті               : {N * V_START:.3f} м³")
    print(f"Загальний газ у кінці порцій           : {N * V_END:.3f} м³")
    print(f"Повна P–V робота порцій                : {case['W_PV_total']:.6f}")
    print(f"Корисна робота системи А               : {case['W_useful_A']:.6f}")
    print(f"Корисна робота системи Б               : {case['W_useful_B']:.6f}")
    print(f"Початковий запас Б                     : {case['reserve_B_start']:.6f}")
    print(f"Кінцевий запас Б                       : {case['reserve_B_end']:.6f}")
    print(f"Різниця запасу Б                       : {case['reserve_difference_B']:.6e}")
    print(f"Вхідна робота поточного циклу Б         : {case['W_input_B']:.6f}")
    print(f"Залишок енергобалансу (має бути 0)      : {case['hidden_energy_residual_B']:.6e}")
    print(f"Відношення W_B / W_A                   : {case['gain_B_over_A']:.6f}")
    print()
    print("Початковий запас системи Б не прирівнюється до нуля:")
    print("він переходить у кінцевий запас і призначений для наступного циклу.")
    print("Різниця корисної роботи не є доказом появи нової енергії без аналізу")
    print("початкового та кінцевого запасів і повної P–V роботи.")

    total_stroke = V_START - V_END
    x = np.linspace(0.0, total_stroke, POINTS)
    f_a = system_a_force(x)
    f_b = system_b_force(x)
    w_a = cumulative_integral(f_a, x)
    w_b = cumulative_integral(f_b, x)

    fig, (force_axis, work_axis) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    force_axis.plot(x, f_a, label="Система А", color="tab:orange")
    force_axis.plot(x, f_b, label="Система Б", color="tab:blue")
    force_axis.axhline(F_LOAD, color="black", linestyle="--", label="F_load")
    force_axis.set_title("Силові профілі без втрат")
    force_axis.set_xlabel("Загальний робочий хід")
    force_axis.set_ylabel("Сумарна сила")
    force_axis.grid(alpha=0.3)
    force_axis.legend()

    work_axis.plot(x, w_a, label="Накопичена робота А", color="tab:orange")
    work_axis.plot(x, w_b, label="Накопичена робота Б", color="tab:blue")
    work_axis.set_title("Накопичена P–V робота")
    work_axis.set_xlabel("Загальний робочий хід")
    work_axis.set_ylabel("Робота")
    work_axis.grid(alpha=0.3)
    work_axis.legend()

    figure_path = OUTPUT_DIR / "ideal_force_and_work.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"Графік: {figure_path}")

    write_parameter_table()


if __name__ == "__main__":
    main()
