"""Ідеальна математична перевірка гіпотези каскадного стискання газу.

Модель призначена для перевірки одного питання:
    Чи створює каскадна організація процесу додаткову енергію?

Постановка:
- ідеальний газ;
- ізотермічний процес;
- квазістатичний (повільний, без динамічних втрат) процес;
- відсутні сили тертя, витоки, інерція, робота клапанів;
- однакові початковий і кінцевий стани в обох системах;
- однаковий об'єм газу, однакова маса, однаковий час і однакове навантаження;
- каскадна система відрізняється тільки організацією розподілу сил по стадіях.

Мета: перевірити, чи в межах цієї моделі каскадне рознесення стадій створює
додаткову роботу, чи лише змінює форму силових профілів.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------
# Параметри базового прикладу
# ------------------------------
N = 10
P0 = 10.0       # початковий тиск, атм
V0 = 1.0        # об'єм однієї порції на старті, м^3
Vf = 0.1        # об'єм однієї порції в кінці, м^3
S = 1.0         # площа поршня

# Загальний об'єм газу системи = N * V0
# Кінцевий загальний об'єм = N * Vf
# Тобто весь газ стискається з 10 м^3 до 1 м^3

# ------------------------------
# Базова термодинаміка
# ------------------------------
def pv_work_isothermal(P_start: float, V_start: float, V_end: float) -> float:
    """Робота ізотермічного стискання/розширення: W = P*V*ln(V_start/V_end)"""
    return P_start * V_start * np.log(V_start / V_end)


def system_a_force(displacement: np.ndarray, N_cyl: int = N, P_start: float = P0,
                   V_start: float = V0, S_area: float = S) -> np.ndarray:
    """Система А: всі циліндри синхронно знаходяться в однаковому стані.

    Для кожної порції:
        V = V_start - displacement
        P = P_start * V_start / V
    Загальна сила:
        F_A = N * P * S
    """
    V_current = V_start - displacement
    V_current = np.clip(V_current, a_min=Vf, a_max=V_start)
    P_current = P_start * V_start / V_current
    return N_cyl * P_current * S_area


def system_b_force(displacement: np.ndarray, N_cyl: int = N, P_start: float = P0,
                   V_start: float = V0, V_end: float = Vf, S_area: float = S) -> np.ndarray:
    """Система Б: всі циліндри рухаються одночасно, але знаходяться на різних стадіях.

    Формула відповідає базовій моделі каскаду:
        V_i = i * V_stage + dV
        P_i = P_start * V_start / V_i
    де dV — загальний крок, що представляє відносне переміщення по��шнів.
    """
    stage_width = (V_start - V_end) / N_cyl
    force = np.zeros_like(displacement, dtype=float)

    for i in range(1, N_cyl + 1):
        # Вектор об'ємів i-ї стадії в момент перебігу процесу.
        # За базовою ідеєю, рознесення стадій зберігається, а всі циліндри рухаються одночасно.
        V_current = i * stage_width + displacement
        V_current = np.clip(V_current, a_min=V_end, a_max=V_start + stage_width)
        P_current = P_start * V_start / V_current
        force += P_current * S_area

    return force


def integrate_force(force: np.ndarray, displacement: np.ndarray) -> float:
    """Чисельна інтеграція роботи: W = ∫ F dx."""
    return float(np.trapz(force, displacement))


def main() -> None:
    # Однаковий хід в роботі для обох систем:
    # від 0 до (V_start - V_end) для однієї порції
    dx = np.linspace(0.0, V0 - Vf, 1000)

    F_A = system_a_force(dx)
    F_B = system_b_force(dx)

    W_A = integrate_force(F_A, dx)
    W_B = integrate_force(F_B, dx)

    # Тотальна теоретична робота одного об'єму газу з 1 м^3 до 0.1 м^3
    W_PV_single = pv_work_isothermal(P0, V0, Vf)
    W_PV_total = N * W_PV_single

    # Порівняння в межах однієї порції та всієї системи
    print("=== Ідеальна перевірка гіпотези каскадного стискання ===")
    print(f"Кількість циліндрів N                : {N}")
    print(f"Початковий тиск P0                  : {P0:.3f} атм")
    print(f"Об'єм однієї порції: V_start = {V0:.3f} м^3, V_end = {Vf:.3f} м^3")
    print(f"Повна теоретична робота P–V         : {W_PV_total:.6f} од. роботи")
    print(f"Інтегральна робота системи A        : {W_A:.6f} од. роботи")
    print(f"Інтегральна робота системи B        : {W_B:.6f} од. роботи")
    print(f"Різниця |W_B - W_A|                 : {abs(W_B - W_A):.6f}")
    print(f"Відносна різниця                    : {abs(W_B - W_A) / max(W_PV_total, 1e-9):.4%}")
    print()
    print("Інтерпретація:")
    print("- Якщо W_A ≈ W_B ≈ W_PV_total, то каскадна організація не створює прихованої енергії.")
    print("- Різниця полягає в розподілі сил по ходу, а не в збільшенні загальної роботи.")
    print("- У цьому режимі гіпотеза розглядається лише як зміна форми силового профілю.")

    # Графік
    fig, (ax_force, ax_work) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax_force.plot(dx, F_A, label="Система A (синхронна)", color="tab:orange", linewidth=2)
    ax_force.plot(dx, F_B, label="Система B (каскадна)", color="tab:blue", linewidth=2)
    ax_force.set_title("Порівняння сумарних сил у ідеальній моделі")
    ax_force.set_xlabel("Хід / шлях стискання")
    ax_force.set_ylabel("Сумарна сила")
    ax_force.grid(True, alpha=0.3)
    ax_force.legend()

    # Для графіка роботи інтегруємо кумулятивно
    cumulative_A = np.concatenate(([0.0], np.cumsum(0.5 * (F_A[:-1] + F_A[1:]) * np.diff(dx))))
    cumulative_B = np.concatenate(([0.0], np.cumsum(0.5 * (F_B[:-1] + F_B[1:]) * np.diff(dx))))

    ax_work.plot(dx, cumulative_A, label="W_A", color="tab:orange", linewidth=2)
    ax_work.plot(dx, cumulative_B, label="W_B", color="tab:blue", linewidth=2)
    ax_work.axhline(W_PV_total, color="black", linestyle="--", label="W_PV (теоретична)")
    ax_work.set_title("Накопичена робота")
    ax_work.set_xlabel("Хід / шлях стискання")
    ax_work.set_ylabel("Робота")
    ax_work.grid(True, alpha=0.3)
    ax_work.legend()

    plt.savefig("ideal_hypothesis_check.png", dpi=200)
    print("\nГрафік збережено: ideal_hypothesis_check.png")


if __name__ == "__main__":
    main()
