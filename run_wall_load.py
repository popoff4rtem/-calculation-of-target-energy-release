"""
run_wall_load.py — точка входа «поменял параметры и запустил».

Считает распределение тепловыделения на стенке теплосъёма (проекция Y/Z,
интеграл по оси пучка) от первичных и вторичных частиц, суммирует их,
переводит в полярные координаты и сохраняет в .npz + .json (+ картинку).

    python run_wall_load.py

Всё, что нужно менять, — в блоке ПАРАМЕТРЫ ниже. Вся физика и оптимизация —
в beam_heat_load.py.
"""

from beam_heat_load import (BeamParams, TargetParams, Config, compute_wall_load,
                            plot_wall_load, max_speed_m_per_s, sweep_length_mm)

# ============================================================================
#                                ПАРАМЕТРЫ
# ============================================================================

# --- исходные данные GEANT4/SRIM -------------------------------------------
PRIMARY_CSV = "Effective traks/primary_de_points_20_MeV.csv"
SECONDARY_CSV = "Effective traks/secondary_de_points_20_MeV.csv"   # None — только первичные
N_SIM_PROTONS = 10_000          # сколько протонов было в симуляции

# --- пучок ------------------------------------------------------------------
BEAM = BeamParams(
    sigma_y_mm=3.33,            # поперечный размер пучка по y
    sigma_z_mm=2.66,            # поперечный размер пучка по z
    peak_current_mA=100.0,      # ток в импульсе
    pulse_us=100.0,             # длительность импульса
    rep_rate_Hz=100.0,          # частота повторения
)

# --- мишенная сборка --------------------------------------------------------
# Болванка едет поперёк пучка вдоль оси y. За время импульса она проходит
# v * pulse_us, поэтому энерговыделение размазывается по траектории.
# Предельная скорость (зазор от центра пучка до края болванки = 5 sigma_y
# в начале и в конце импульса) печатается ниже; при меньшей скорости зазор больше.
TARGET = TargetParams(
    diameter_mm=50.0,           # диаметр болванки
    speed_m_per_s=167.0,        # скорость движения сборки (0 — неподвижна)
                                # 167 м/с — предел для этих параметров, см. печать ниже
    n_pucks=1,                  # болванок на колесе: в одну попадает rep_rate / n_pucks
    edge_margin_sigma=5.0,      # требуемый зазор до края болванки, в sigma_y
)

# --- единицы и сетки --------------------------------------------------------
UNITS = "J/mm3"                 # J/mm3 | J/cm3 | W/mm3 | W/cm3 (можно Дж/см3 и т.п.)
                                # J — за импульс, W — средняя мощность;
                                # проекция сохраняется в соответствующих
                                # поверхностных единицах (W/cm3 -> W/cm2)
BINS = (100, 100, 100)          # воксельная сетка по данным (x, y, z)
OUTPUT_STEP_MM = None           # шаг итоговой решётки мишени, мм
                                # None — авто (самый мелкий поперечный воксель
                                # первичных, но не мельче MAX_OUTPUT_BINS).
                                # Для теплофизики обычно хватает 0.1–0.25 мм
MAX_OUTPUT_BINS = 3000          # предел бинов по стороне (память: N² · 8 байт)
N_R, N_PHI = 400, 720           # полярная сетка (r, φ)

# --- вывод ------------------------------------------------------------------
OUTPUT = "results/wall_load_20MeV"   # база имён: .npz + .json (+ .png)
MAKE_PLOT = True
PLOT_ZOOM_MM = None             # масштаб картинки по y/z, None — вся мишень

# ============================================================================

cfg = Config(
    primary_csv=PRIMARY_CSV,
    secondary_csv=SECONDARY_CSV,
    n_sim_protons=N_SIM_PROTONS,
    bins=BINS,
    beam=BEAM,
    target=TARGET,
    units=UNITS,
    n_r=N_R,
    n_phi=N_PHI,
    output=OUTPUT,
    output_step_mm=OUTPUT_STEP_MM,
    max_output_bins=MAX_OUTPUT_BINS,
)

if __name__ == "__main__":
    v_max = max_speed_m_per_s(TARGET, BEAM)
    print(f"Предельная скорость сборки : {v_max:,.1f} м/с "
          f"(путь за импульс {v_max * BEAM.pulse_us * 1e-3:.2f} мм, "
          f"зазор {TARGET.edge_margin_sigma:g} sigma_y с обеих сторон)")
    print(f"Заданная скорость          : {TARGET.speed_m_per_s:,.1f} м/с "
          f"(путь за импульс {sweep_length_mm(TARGET, BEAM):.2f} мм)\n")

    result = compute_wall_load(cfg)

    if MAKE_PLOT:
        plot_wall_load(result, zoom_mm=PLOT_ZOOM_MM, save=f"{OUTPUT}.png")

    # Что лежит в result (и в .npz):
    #   y_mm, z_mm, y_edges_mm, z_edges_mm — сетка координат, мм
    #   q_primary, q_secondary, q_total    — поверхностная плотность, result.surface_units
    #   r_mm, phi_rad, q_polar_*           — то же в полярных координатах
    #   q_radial_mean                      — радиальный профиль (среднее по азимуту)
    #   x_mm, eps_axis_*                   — объёмная плотность на оси пучка, result.volume_units
    #   meta                               — все параметры и балансы энергии
