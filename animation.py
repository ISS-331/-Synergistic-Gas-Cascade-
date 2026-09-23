"""Анімація базового порівняння систем А і Б.

Це ілюстративна модель, що використовує ті самі залежності, що й
simulation.ру: синхронну криву сили для А та каскадну сумарну силу для Б.

Запуск:
    pip install numpy matplotlib
    python animation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

P1 = 10.0
V1 = 1.0
N = 10
S = 1.0
F_load = 23.99

# Локальна координата одного каскадного кроку.
dV_range = np.linspace(0.0, V1, 200)
F_B_curve = np.array([
    S * sum((P1 * V1) / (i * V1 + dV) for i in range(1, N + 1))
    for dV in dV_range
])

F_A_init = N * P1 * S
h_A_stall = F_A_init / F_load - 1.0
h_B_total = N * V1 / S
work_A_total = F_load * h_A_stall
work_B_total = F_load * h_B_total

# Кадры показывают последовательное прохождение каскадных рабочих позиций.
frames = 240
h_values = np.linspace(0.0, h_B_total, frames)

fig, (ax_force, ax_work) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle("Базовое сравнение системы А и каскадной системы Б")

ax_force.set_title("Суммарная сила и постоянная нагрузка")
ax_force.set_xlabel("Условный общий ход")
ax_force.set_ylabel("Сила")
ax_force.axhline(F_load, color="black", linestyle="--", label="F_load")
ax_force.set_xlim(0, h_B_total)
ax_force.set_ylim(0, max(F_A_init, F_B_curve.max()) * 1.1)

ax_work.set_title("Накопленная полезная работа")
ax_work.set_xlabel("Условный общий ход")
ax_work.set_ylabel("Работа")
ax_work.set_xlim(0, h_B_total)
ax_work.set_ylim(0, max(work_A_total, work_B_total) * 1.1)

line_b_force, = ax_force.plot([], [], color="tab:blue", label="Система Б")
line_a_force, = ax_force.plot([], [], color="tab:orange", label="Система А")
point_b_force, = ax_force.plot([], [], "o", color="tab:blue")
point_a_force, = ax_force.plot([], [], "o", color="tab:orange")
ax_force.legend(loc="upper right")

line_b_work, = ax_work.plot([], [], color="tab:blue", label="W_B")
line_a_work, = ax_work.plot([], [], color="tab:orange", label="W_A")
point_b_work, = ax_work.plot([], [], "o", color="tab:blue")
point_a_work, = ax_work.plot([], [], "o", color="tab:orange")
ax_work.legend(loc="upper left")

info = fig.text(0.02, 0.01, "", family="monospace")


def init():
    for artist in (
        line_b_force, line_a_force, point_b_force, point_a_force,
        line_b_work, line_a_work, point_b_work, point_a_work,
    ):
        artist.set_data([], [])
    info.set_text("")
    return (
        line_b_force, line_a_force, point_b_force, point_a_force,
        line_b_work, line_a_work, point_b_work, point_a_work, info,
    )


def update(frame):
    h = h_values[frame]
    # После каждой рабочей позиции каскад получает новую порцию; поэтому
    # локальная dV-координата повторяется в пределах каждого шага V1/S.
    local_dv = (h * S) % V1
    force_b = np.interp(local_dv, dV_range, F_B_curve)
    force_a = F_A_init / (1.0 + h) if h <= h_A_stall else 0.0

    x = h_values[: frame + 1]
    b_force_history = np.array([
        np.interp((value * S) % V1, dV_range, F_B_curve)
        for value in x
    ])
    a_force_history = np.array([
        F_A_init / (1.0 + value) if value <= h_A_stall else 0.0
        for value in x
    ])
    b_work_history = F_load * x
    a_work_history = F_load * np.minimum(x, h_A_stall)

    line_b_force.set_data(x, b_force_history)
    line_a_force.set_data(x, a_force_history)
    point_b_force.set_data([h], [force_b])
    point_a_force.set_data([h], [force_a])
    line_b_work.set_data(x, b_work_history)
    line_a_work.set_data(x, a_work_history)
    point_b_work.set_data([h], [F_load * h])
    point_a_work.set_data([h], [F_load * min(h, h_A_stall)])

    stage = min(N, int(h / (V1 / S)) + 1)
    info.set_text(
        f"ход = {h:5.2f} | стадия Б = {stage:2d} | "
        f"F_B = {force_b:5.2f} | F_A = {force_a:5.2f} | "
        f"W_B = {F_load * h:6.2f} | W_A = {F_load * min(h, h_A_stall):6.2f}"
    )
    return (
        line_b_force, line_a_force, point_b_force, point_a_force,
        line_b_work, line_a_work, point_b_work, point_a_work, info,
    )


animation = FuncAnimation(
    fig, update, init_func=init, frames=frames, interval=40, blit=True
)

plt.show()
