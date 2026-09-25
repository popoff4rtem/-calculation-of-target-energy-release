"""
beam_heat_load.py — расчёт распределения тепловой нагрузки на стенке теплосъёма.

Стенка перпендикулярна оси пучка (x), поэтому искомая величина — поверхностная
плотность энерговыделения на плоскости Y/Z, т.е. интеграл объёмного
энерговыделения вдоль оси распространения пучка:

    q(y, z) = ∫ eps(x, y, z) dx        [Дж/мм², Вт/см², ...]

ЦЕПОЧКА РАСЧЁТА
---------------
1.  CSV из GEANT4/SRIM (первичные и вторичные отдельно) -> воксельная карта
    H[x, y, z] = сумма dE в вокселе, [МэВ] на N_sim протонов симуляции.
    Сетка строится по фактическому охвату данных: bins=(100, 100, 100) по
    умолчанию. У первичных и вторичных частиц охват разный, поэтому и размер
    вокселя свой — он печатается и сохраняется в метаданных.
2.  Нормировка на один протон и переход к объёмной плотности:
        eps = H / (N_sim * dV)         [МэВ / (мм³ · протон)]
3.  Общая решётка мишени с шагом step (по умолчанию — самый мелкий поперечный
    воксель первичных). Карта переносится на неё точным перераспределением по
    площади (сумма сохраняется) и дополняется нулями до размера мишени по y и z.
    По оси пучка расширение не нужно. Общая решётка у первичных и вторичных
    означает, что их карты складываются напрямую, без пересетки.
4.  Свёртка с поперечным профилем пучка (эллиптический гаусс sigma_y, sigma_z).
    Ядро — точный интеграл гаусса по бину (erf), нормированный на полный гаусс:
    хвост, ушедший за край мишени, честно теряется и виден в балансе энергии.
5.  Масштабирование под параметры пучка (ток в импульсе, длительность, частота)
    и перевод в требуемые единицы: Дж/мм³, Дж/см³, Вт/мм³, Вт/см³
    (и соответствующие поверхностные Дж/мм², Вт/см² и т.д.).
6.  Проекция на Y/Z, суммирование первичных и вторичных, маска круглой мишени.
7.  Перевод в полярные координаты (r, phi) для круглой мишени + азимутально
    усреднённый радиальный профиль.
8.  Сохранение в .npz (сетка координат + карты) и .json (метаданные).

Дополнительно считается объёмная плотность на оси пучка eps(x, 0, 0) —
значение свёртки в одной точке, т.е. без построения расширенной 3D-карты.

ОПТИМИЗАЦИЯ ПАМЯТИ
------------------
Расширенная 3D-карта (100 x 1925 x 1925 при bins=100 и мишени Ø50 мм) заняла бы
~3 ГБ. Она не нужна: и свёртка по (y, z), и интегрирование по x линейны и
коммутируют, поэтому

    sum_x ( H[x] * K )  ==  ( sum_x H[x] ) * K

Сначала схлопываем по x (нативная сетка 100³ = 8 МБ), затем один раз свёртываем
2D-карту. Результат идентичен, разрешение воксельной сетки не снижается, а
шаг dx вообще сокращается в формуле — ответ от разрешения по оси пучка не
зависит (проверено: bit-identical при nx = 10...150). Сама свёртка считается как
'full' от компактной нативной карты (~100x100) с ядром размером с мишень, т.е.
массив размера мишени создаётся ровно один раз. Пик памяти — десятки МБ.

Размер итоговой решётки ограничен max_output_bins (по умолчанию 3000 бинов по
стороне, ~69 МБ на карту): шаг автоматически огрубляется, если охват данных даёт
воксель в единицы микрон. Для теплофизического расчёта осмысленный шаг — 0.1–0.25
мм (поле всё равно размыто пучком с sigma ~ 3 мм), он задаётся output_step_mm.

CSV читаются чанками (float32 для координат), результат вокселизации кешируется
на диск — повторные прогоны с другими параметрами пучка не перечитывают CSV.

Пример
------
    from beam_heat_load import BeamParams, TargetParams, Config, compute_wall_load

    res = compute_wall_load(Config(
        primary_csv="Effective traks/primary_de_points_20_MeV.csv",
        secondary_csv="Effective traks/secondary_de_points_20_MeV.csv",
        n_sim_protons=10000,
        bins=(100, 100, 100),
        beam=BeamParams(sigma_y_mm=3.33, sigma_z_mm=2.66,
                        peak_current_mA=100.0, pulse_us=100.0, rep_rate_Hz=100.0),
        target=TargetParams(diameter_mm=50.0),
        units="W/cm3",
        output="wall_load_20MeV",
    ))
    res.summary()
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from scipy.special import erf

# ============================================================================
#                          КОНСТАНТЫ И ЕДИНИЦЫ
# ============================================================================

E_CHARGE = 1.602176634e-19          # Кл, элементарный заряд (СИ, точное значение)
MEV_TO_J = E_CHARGE * 1e6           # Дж в 1 МэВ

_LEN_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}   # сколько мм в единице длины

_RU_UNITS = {"дж": "J", "вт": "W", "мм": "mm", "см": "cm", "м": "m"}

REQUIRED_COLUMNS = ("x_mm", "y_mm", "z_mm", "energy_deposition_mev")


def parse_units(units: str) -> tuple[str, str]:
    """'W/cm3' | 'Вт/см3' | 'J/mm^2' -> ('W', 'cm').

    Показатель степени в записи игнорируется: объёмная плотность всегда /длина³,
    поверхностная — /длина². Возвращается род величины ('J' — за импульс,
    'W' — средняя мощность) и единица длины.
    """
    s = units.strip().lower().replace("^", "").replace(" ", "")
    for ru, en in _RU_UNITS.items():
        s = s.replace(ru, en)
    if "/" not in s:
        raise ValueError(f"Не могу разобрать единицы: {units!r} (нужно вида 'W/cm3')")
    num, den = s.split("/", 1)
    kind = num.upper()
    if kind not in ("J", "W"):
        raise ValueError(f"Числитель должен быть J (Дж) или W (Вт), получено {num!r}")
    den = den.rstrip("0123456789")
    if den not in _LEN_MM:
        raise ValueError(f"Единица длины должна быть mm/cm/m, получено {den!r}")
    return kind, den


def _fmt_unit(kind: str, length: str, exp: int) -> str:
    return f"{kind}/{length}" if exp == 1 else f"{kind}/{length}{exp}"


# ============================================================================
#                              ПАРАМЕТРЫ
# ============================================================================

@dataclass
class BeamParams:
    """Параметры протонного пучка."""
    sigma_y_mm: float = 3.33
    sigma_z_mm: float = 2.66
    peak_current_mA: float = 100.0      # ток в импульсе
    pulse_us: float = 100.0             # длительность импульса
    rep_rate_Hz: float = 100.0          # частота повторения

    @property
    def duty_cycle(self) -> float:
        """Коэффициент заполнения D = t / T (доля времени под импульсом)."""
        return self.pulse_us * 1e-6 * self.rep_rate_Hz

    @property
    def duty_ratio(self) -> float:
        """Скважность S = T / t = 1 / D."""
        d = self.duty_cycle
        return float("inf") if d <= 0 else 1.0 / d

    @property
    def protons_per_pulse(self) -> float:
        return self.peak_current_mA * 1e-3 * self.pulse_us * 1e-6 / E_CHARGE

    @property
    def protons_per_second(self) -> float:
        return self.protons_per_pulse * self.rep_rate_Hz

    @property
    def average_current_mA(self) -> float:
        return self.peak_current_mA * self.duty_cycle

    @property
    def fwhm_y_mm(self) -> float:
        return 2.3548200450309493 * self.sigma_y_mm

    @property
    def fwhm_z_mm(self) -> float:
        return 2.3548200450309493 * self.sigma_z_mm


@dataclass
class TargetParams:
    """Мишенная сборка: круглая бериллиевая болванка на вращающемся колесе.

    Ось болванки совпадает с осью пучка. За время импульса болванка смещается
    поперёк пучка вдоль оси y со скоростью speed_m_per_s, поэтому энерговыделение
    размазывается по траектории. Колесо несёт n_pucks болванок, так что в одну и
    ту же болванку импульс попадает с частотой rep_rate / n_pucks.
    """
    diameter_mm: float = 50.0
    speed_m_per_s: float = 0.0      # скорость движения болванки поперёк пучка (0 — неподвижна)
    n_pucks: int = 1                # число болванок в сборке
    edge_margin_sigma: float = 5.0  # требуемый зазор от центра пучка до края болванки, в σ_y

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm

    @property
    def area_mm2(self) -> float:
        return math.pi * self.radius_mm ** 2


def sweep_length_mm(target: TargetParams, beam: BeamParams) -> float:
    """Путь, который болванка проходит за время импульса, мм."""
    return target.speed_m_per_s * beam.pulse_us * 1e-3


def max_speed_m_per_s(target: TargetParams, beam: BeamParams) -> float:
    """Предельная скорость: в начале и в конце импульса центр пучка должен
    отстоять от края болванки не меньше чем на edge_margin_sigma * sigma_y.
    """
    free_mm = 2.0 * (target.radius_mm - target.edge_margin_sigma * beam.sigma_y_mm)
    if free_mm <= 0:
        return 0.0
    return free_mm * 1e-3 / (beam.pulse_us * 1e-6)


def edge_gap_mm(target: TargetParams, beam: BeamParams) -> float:
    """Зазор от центра пучка до края болванки в крайнем положении, мм.

    Предполагается, что траектория отцентрована на болванке, т.е. зазоры с двух
    сторон одинаковые.
    """
    return target.radius_mm - 0.5 * sweep_length_mm(target, beam)


@dataclass
class Config:
    """Полная конфигурация расчёта."""
    primary_csv: str
    secondary_csv: str | None = None
    n_sim_protons: int = 10000
    bins: tuple[int, int, int] = (100, 100, 100)

    beam: BeamParams = field(default_factory=BeamParams)
    target: TargetParams = field(default_factory=TargetParams)

    units: str = "W/cm3"                # объёмные единицы; поверхностные выводятся автоматически
    n_r: int = 400                      # число узлов по радиусу в полярной сетке
    n_phi: int = 720                    # число узлов по углу

    output: str | None = None           # базовое имя файлов результата (без расширения)
    output_step_mm: float | None = None # шаг итоговой решётки (None = авто)
    mask_outside_target: bool = True    # обнулять всё за пределами радиуса мишени

    chunksize: int = 2_000_000          # строк CSV за раз
    cache_dir: str | None = ".voxel_cache"
    max_output_bins: int = 3000         # предел числа бинов итоговой решётки по стороне
    save_dtype: str = "float32"
    compress: bool = False
    verbose: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bins"] = list(self.bins)
        d["beam"]["protons_per_pulse"] = self.beam.protons_per_pulse
        d["beam"]["protons_per_second"] = self.beam.protons_per_second
        d["beam"]["duty_cycle"] = self.beam.duty_cycle
        d["beam"]["duty_ratio"] = self.beam.duty_ratio
        d["beam"]["average_current_mA"] = self.beam.average_current_mA
        d["target"]["radius_mm"] = self.target.radius_mm
        return d


# ============================================================================
#                       ВОКСЕЛИЗАЦИЯ (чанками + кеш)
# ============================================================================

@dataclass
class VoxelGrid:
    """H[x, y, z] — сумма dE в вокселе, МэВ (на N_sim протонов симуляции)."""
    H: np.ndarray
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    source: str
    n_rows: int
    sum_de_mev: float

    @property
    def voxel_size_mm(self) -> tuple[float, float, float]:
        return tuple(float(e[1] - e[0]) for e in self.edges)

    @property
    def voxel_volume_mm3(self) -> float:
        dx, dy, dz = self.voxel_size_mm
        return dx * dy * dz

    @property
    def extent_mm(self) -> dict:
        return {ax: (float(e[0]), float(e[-1]))
                for ax, e in zip("xyz", self.edges)}

    def describe(self) -> str:
        dx, dy, dz = self.voxel_size_mm
        ex = self.extent_mm
        return (
            f"  источник      : {self.source}\n"
            f"  точек         : {self.n_rows:,}\n"
            f"  ΣdE           : {self.sum_de_mev:,.1f} МэВ\n"
            f"  сетка         : {self.H.shape[0]} x {self.H.shape[1]} x {self.H.shape[2]}\n"
            f"  охват x/y/z   : [{ex['x'][0]:.4f}, {ex['x'][1]:.4f}] / "
            f"[{ex['y'][0]:.4f}, {ex['y'][1]:.4f}] / [{ex['z'][0]:.4f}, {ex['z'][1]:.4f}] мм\n"
            f"  размер вокселя: dx={dx*1e3:.3f} мкм, dy={dy*1e3:.3f} мкм, dz={dz*1e3:.3f} мкм "
            f"(dV={self.voxel_volume_mm3:.6g} мм³)\n"
            f"  ΣdE в сетке   : {self.H.sum():,.1f} МэВ"
        )


def _iter_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"x_mm": np.float32, "y_mm": np.float32, "z_mm": np.float32,
               "energy_deposition_mev": np.float64},
        chunksize=chunksize,
    )
    for chunk in reader:
        yield chunk


def _scan_bounds(path: str, chunksize: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Первый проход: границы по x, y, z, число строк и ΣdE."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n_rows = 0
    total = 0.0
    for chunk in _iter_chunks(path, chunksize):
        xyz = chunk[["x_mm", "y_mm", "z_mm"]].to_numpy()
        lo = np.minimum(lo, xyz.min(axis=0))
        hi = np.maximum(hi, xyz.max(axis=0))
        n_rows += len(chunk)
        total += float(chunk["energy_deposition_mev"].sum())
    if n_rows == 0:
        raise ValueError(f"Файл пуст или не содержит данных: {path}")
    return lo, hi, n_rows, total


def _cache_path(cache_dir: str, path: str, bins) -> Path:
    st = os.stat(path)
    tag = f"{Path(path).stem}__{bins[0]}x{bins[1]}x{bins[2]}__{int(st.st_mtime)}_{st.st_size}"
    return Path(cache_dir) / f"{tag}.npz"


def voxelize_csv(path: str, bins=(100, 100, 100), chunksize: int = 2_000_000,
                 cache_dir: str | None = ".voxel_cache", verbose: bool = True) -> VoxelGrid:
    """Строит воксельную карту энерговыделения из CSV (два прохода, чанками)."""
    path = str(path)
    bins = tuple(int(b) for b in bins)

    cache = _cache_path(cache_dir, path, bins) if cache_dir else None
    if cache is not None and cache.exists():
        with np.load(cache) as d:
            grid = VoxelGrid(
                H=d["H"], edges=(d["ex"], d["ey"], d["ez"]), source=path,
                n_rows=int(d["n_rows"]), sum_de_mev=float(d["sum_de_mev"]),
            )
        if verbose:
            print(f"[voxelize] из кеша: {cache}")
        return grid

    t0 = time.time()
    if verbose:
        print(f"[voxelize] {path} — проход 1/2 (границы)...", flush=True)
    lo, hi, n_rows, total = _scan_bounds(path, chunksize)

    edges = tuple(np.linspace(lo[i], hi[i], bins[i] + 1) for i in range(3))
    H = np.zeros(bins, dtype=np.float64)

    if verbose:
        print(f"[voxelize] {path} — проход 2/2 (гистограмма)...", flush=True)
    for chunk in _iter_chunks(path, chunksize):
        sample = chunk[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
        # страхуемся от вылета за границы из-за float32-округления
        np.clip(sample, lo, hi, out=sample)
        h, _ = np.histogramdd(sample, bins=edges,
                              weights=chunk["energy_deposition_mev"].to_numpy())
        H += h

    grid = VoxelGrid(H=H, edges=edges, source=path, n_rows=n_rows, sum_de_mev=total)

    if cache is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(cache, H=H, ex=edges[0], ey=edges[1], ez=edges[2],
                 n_rows=n_rows, sum_de_mev=total)
        if verbose:
            print(f"[voxelize] кеш записан: {cache}")

    if verbose:
        print(f"[voxelize] готово за {time.time() - t0:.1f} с")
    return grid


# ============================================================================
#         ОБЩАЯ РЕШЁТКА МИШЕНИ, РАСШИРЕНИЕ НУЛЯМИ И СВЁРТКА С ПУЧКОМ
# ============================================================================

def canonical_lattice(radius_mm: float, step_mm: float):
    """Общая решётка мишени: центры бинов — кратные шагу, ноль в центре бина.

    Одна и та же решётка используется для первичных и вторичных частиц, поэтому
    карты складываются напрямую, без пересетки.
    """
    n = int(math.ceil(radius_mm / step_mm))
    centers = np.arange(-n, n + 1, dtype=np.float64) * step_mm
    edges = (np.arange(-n, n + 2, dtype=np.float64) - 0.5) * step_mm
    return edges, centers, n


def _block_range(native_edges: np.ndarray, step_mm: float, n: int) -> tuple[int, int]:
    """Индексы бинов решётки, накрывающих охват исходных данных."""
    k_lo = int(math.floor(float(native_edges[0]) / step_mm + 0.5))
    k_hi = int(math.ceil(float(native_edges[-1]) / step_mm - 0.5))
    return max(k_lo, -n), min(k_hi, n)


def _rebin_axis(values: np.ndarray, src_edges: np.ndarray,
                dst_edges: np.ndarray, axis: int) -> np.ndarray:
    """Точное перераспределение по площади (сохраняет сумму) вдоль одной оси.

    Внутри исходного бина плотность считается постоянной: интегральная сумма
    интерполируется линейно по границам бинов, разности дают новые бины.
    """
    v = np.moveaxis(values, axis, 0)
    cum = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)], axis=0)
    idx, w, _ = _interp_weights(src_edges, dst_edges)
    c = cum[idx] * (1.0 - w).reshape((-1,) + (1,) * (v.ndim - 1)) \
        + cum[idx + 1] * w.reshape((-1,) + (1,) * (v.ndim - 1))
    return np.moveaxis(np.diff(c, axis=0), 0, axis)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _norm_cdf_int(x: np.ndarray) -> np.ndarray:
    """Первообразная нормального CDF: Psi(x) = x*Phi(x) + phi(x)."""
    return x * _norm_cdf(x) + np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _beam_cdf(u: np.ndarray, sigma_mm: float, sweep_mm: float) -> np.ndarray:
    """Функция распределения поперечного профиля пучка в системе болванки.

    Неподвижная болванка — обычный гаусс. Если болванка движется, за время
    импульса центр пучка равномерно проезжает отрезок длиной sweep_mm, поэтому
    профиль — свёртка гаусса с равномерным распределением. Для неё известна
    замкнутая форма: (sigma/L) * [Psi((u+L/2)/sigma) - Psi((u-L/2)/sigma)].
    """
    if sweep_mm <= 0:
        return _norm_cdf(u / sigma_mm)
    if sigma_mm <= 0:
        return np.clip(u / sweep_mm + 0.5, 0.0, 1.0)
    h = 0.5 * sweep_mm
    return (sigma_mm / sweep_mm) * (_norm_cdf_int((u + h) / sigma_mm)
                                    - _norm_cdf_int((u - h) / sigma_mm))


def _beam_bin_weights(offsets_mm: np.ndarray, sigma_mm: float, step_mm: float,
                      sweep_mm: float = 0.0) -> np.ndarray:
    """Доля пучка, попадающая в бин шириной step_mm со смещением offsets_mm.

    Считается как точный интеграл профиля по бину (разность функций
    распределения), а не выборка в центре бина. Нормировка — на полный профиль,
    поэтому хвост, обрезанный краем мишени, честно теряется, а не размазывается
    обратно.
    """
    if sigma_mm <= 0 and sweep_mm <= 0:
        w = np.zeros_like(offsets_mm)
        w[np.argmin(np.abs(offsets_mm))] = 1.0
        return w
    w = (_beam_cdf(offsets_mm + 0.5 * step_mm, sigma_mm, sweep_mm)
         - _beam_cdf(offsets_mm - 0.5 * step_mm, sigma_mm, sweep_mm))
    return np.maximum(w, 0.0)          # срезаем шум округления в дальних хвостах


def _beam_kernel_1d(sigma_mm: float, step_mm: float, half: int,
                    sweep_mm: float = 0.0) -> np.ndarray:
    """1D-ядро пучка длиной 2*half+1, узлы — целые смещения в вокселях."""
    offsets = np.arange(-half, half + 1, dtype=np.float64) * step_mm
    return _beam_bin_weights(offsets, sigma_mm, step_mm, sweep_mm)


def expand_and_convolve(block: np.ndarray,
                        n_low: tuple[int, int], n_lat: tuple[int, int],
                        step: tuple[float, float],
                        sigma_y_mm: float, sigma_z_mm: float,
                        sweep_y_mm: float = 0.0) -> np.ndarray:
    """Дополняет компактный блок нулями до решётки мишени и свёртывает с пучком.

    Реализовано как 'full'-свёртка компактного блока с ядром размером с мишень:
    массив размера мишени создаётся ровно один раз, расширенной копии исходной
    карты нет. Результат совпадает с «дополнить нулями, затем свернуть».

    block      — карта на решётке мишени, обрезанная до охвата данных
    n_low      — сколько бинов решётки лежит левее блока по y и z
    n_lat      — полное число бинов решётки по y и z
    sweep_y_mm — путь болванки за импульс (размазывает профиль вдоль y)
    """
    nb_y, nb_z = block.shape
    dy, dz = step
    n_high_y = n_lat[0] - n_low[0] - nb_y
    n_high_z = n_lat[1] - n_low[1] - nb_z

    # Полуширина ядра. Самый дальний воксель источника должен доставать до
    # самого дальнего края мишени: max|смещение| = max(n_low, n_high) + nb - 1.
    # Всё, что шире, лежит уже вне мишени и честно теряется (см. баланс энергии).
    my = max(n_low[0], n_high_y) + nb_y - 1
    mz = max(n_low[1], n_high_z) + nb_z - 1

    ky = _beam_kernel_1d(sigma_y_mm, dy, my, sweep_y_mm)
    kz = _beam_kernel_1d(sigma_z_mm, dz, mz)

    # сепарабельная свёртка: сначала по y, потом по z
    out = fftconvolve(block.astype(np.float64), ky[:, None], mode="full", axes=0)
    out = fftconvolve(out, kz[None, :], mode="full", axes=1)

    # центр выходного индекса n соответствует смещению (n - m) шагов от первого
    # бина блока, поэтому бину решётки k отвечает n = k - n_low + m
    oy, oz = my - n_low[0], mz - n_low[1]
    return np.ascontiguousarray(out[oy:oy + n_lat[0], oz:oz + n_lat[1]])


# ============================================================================
#                      ПОЛЯРНЫЕ КООРДИНАТЫ
# ============================================================================

def _interp_weights(src: np.ndarray, dst: np.ndarray):
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    w = (dst - src[idx]) / (src[idx + 1] - src[idx])
    inside = (dst >= src[0]) & (dst <= src[-1])
    return idx, np.clip(w, 0.0, 1.0), inside


def to_polar(S: np.ndarray, y_c: np.ndarray, z_c: np.ndarray,
             radius_mm: float, n_r: int, n_phi: int):
    """Карта (y, z) -> (r, phi). Билинейная интерполяция, вне сетки — 0."""
    r = np.linspace(0.0, radius_mm, n_r)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    Y = R * np.cos(PHI)
    Z = R * np.sin(PHI)

    dy = y_c[1] - y_c[0]
    dz = z_c[1] - z_c[0]
    coords = np.stack([(Y - y_c[0]) / dy, (Z - z_c[0]) / dz])
    polar = map_coordinates(S, coords, order=1, mode="constant", cval=0.0)
    return r, phi, polar


def depth_linear_profile(grid: "VoxelGrid", n_sim: int, scale_linear: float):
    """Линейная плотность энерговыделения по глубине: dE/dx, проинтегрированная
    по всему сечению. Именно её форма нужна, чтобы восстановить объёмный
    источник как q(y, z) * f(x); осевой профиль для этого не годится — он
    взвешен на ядро пучка.
    """
    dx = grid.voxel_size_mm[0]
    prof = grid.H.sum(axis=(1, 2)) / (n_sim * dx) * scale_linear
    x_c = 0.5 * (grid.edges[0][1:] + grid.edges[0][:-1])
    return x_c, prof


def axis_volumetric_profile(grid: "VoxelGrid", n_sim: int,
                            sigma_y_mm: float, sigma_z_mm: float,
                            scale_volume: float, sweep_y_mm: float = 0.0):
    """Объёмная плотность энерговыделения на оси пучка (y=z=0) вдоль оси x.

    Значение свёртки в одной точке — это взвешенная сумма, поэтому считается
    за миллисекунды и не требует расширенной 3D-карты. Для центрированного
    пучка и центрированного источника это и есть максимум по (y, z).
    """
    dx, dy, dz = grid.voxel_size_mm
    y_c = 0.5 * (grid.edges[1][1:] + grid.edges[1][:-1])
    z_c = 0.5 * (grid.edges[2][1:] + grid.edges[2][:-1])
    wy = _beam_bin_weights(-y_c, sigma_y_mm, dy, sweep_y_mm)
    wz = _beam_bin_weights(-z_c, sigma_z_mm, dz)
    prof = np.einsum("ijk,j,k->i", grid.H, wy, wz)
    prof = prof / (n_sim * dx * dy * dz) * scale_volume
    x_c = 0.5 * (grid.edges[0][1:] + grid.edges[0][:-1])
    return x_c, prof


# ============================================================================
#                              РЕЗУЛЬТАТ
# ============================================================================

@dataclass
class WallLoad:
    """Результат расчёта тепловой нагрузки на стенке."""
    y_mm: np.ndarray                 # центры бинов, мм
    z_mm: np.ndarray
    y_edges_mm: np.ndarray
    z_edges_mm: np.ndarray
    q_primary: np.ndarray            # поверхностная плотность, surface_units
    q_secondary: np.ndarray
    q_total: np.ndarray
    r_mm: np.ndarray
    phi_rad: np.ndarray
    q_polar_total: np.ndarray        # (n_r, n_phi)
    q_polar_primary: np.ndarray
    q_polar_secondary: np.ndarray
    q_radial_mean: np.ndarray        # азимутально усреднённый профиль, (n_r,)
    x_mm: np.ndarray                 # центры вокселей по оси пучка, мм
    eps_axis_primary: np.ndarray     # объёмная плотность на оси пучка, volume_units
    eps_axis_secondary: np.ndarray
    eps_axis_total: np.ndarray
    lin_depth_primary: np.ndarray    # dE/dx по всему сечению, linear_units
    lin_depth_secondary: np.ndarray
    lin_depth_total: np.ndarray
    surface_units: str
    volume_units: str
    linear_units: str
    meta: dict

    # ---- сводные величины -------------------------------------------------
    @property
    def total_power(self) -> float:
        return self.meta["totals"]["total_on_target"]

    def summary(self) -> None:
        m = self.meta
        b, t = m["config"]["beam"], m["config"]["target"]
        su, vu = self.surface_units, self.volume_units
        print("=" * 78)
        print("ТЕПЛОВАЯ НАГРУЗКА НА СТЕНКУ (проекция Y/Z, интеграл по оси пучка)")
        print("=" * 78)
        print(f"Пучок        : σy={b['sigma_y_mm']} мм, σz={b['sigma_z_mm']} мм "
              f"(FWHM {m['beam']['fwhm_y_mm']:.2f} x {m['beam']['fwhm_z_mm']:.2f} мм)")
        print(f"               {b['peak_current_mA']} мА в импульсе x {b['pulse_us']} мкс "
              f"x {b['rep_rate_Hz']} Гц, средний ток {b['average_current_mA']:.4f} мА")
        print(f"               коэффициент заполнения D = t/T = {b['duty_cycle']*100:.3f} %, "
              f"скважность S = T/t = {1/b['duty_cycle']:,.1f}")
        print(f"               {b['protons_per_pulse']:.4e} протонов/импульс, "
              f"{b['protons_per_second']:.4e} протонов/с")
        print(f"Мишень       : Ø{t['diameter_mm']} мм (R={t['radius_mm']} мм), "
              f"площадь {math.pi*t['radius_mm']**2:,.1f} мм²")
        mo, rt = m.get("motion"), m.get("rates")
        if mo:
            if mo["speed_m_per_s"] > 0:
                print(f"Движение     : {mo['speed_m_per_s']:,.4g} м/с вдоль {mo['axis']}, "
                      f"путь за импульс {mo['sweep_mm']:.3f} мм")
                print(f"               зазор до края {mo['edge_gap_mm']:.3f} мм "
                      f"= {mo['edge_gap_sigma']:.2f} σy   "
                      f"(предельная скорость {mo['max_speed_m_per_s']:,.4g} м/с)")
            else:
                print(f"Движение     : нет (болванка неподвижна), "
                      f"предельная скорость {mo['max_speed_m_per_s']:,.4g} м/с")
        if rt:
            print(f"Попадания    : болванок в сборке {rt['n_pucks']}, "
                  f"в одну болванку {rt['hit_rate_Hz']:,.4g} Гц "
                  f"(коэффициент заполнения {rt['puck_duty_cycle']*100:.4f} %, "
                  f"скважность {1/rt['puck_duty_cycle']:,.1f})")
        print(f"Нормировка   : {m['config']['n_sim_protons']:,} протонов в симуляции")
        print("-" * 78)
        for name in ("primary", "secondary"):
            g = m["grids"].get(name)
            if g is None:
                continue
            print(f"{name.upper()}")
            print(f"  ΣdE        : {g['sum_de_mev']:,.1f} МэВ  "
                  f"({g['sum_de_mev']/m['config']['n_sim_protons']:.4f} МэВ/протон)")
            print(f"  воксель    : dx={g['voxel_size_mm'][0]*1e3:.2f} мкм, "
                  f"dy={g['voxel_size_mm'][1]*1e3:.2f} мкм, "
                  f"dz={g['voxel_size_mm'][2]*1e3:.2f} мкм "
                  f"(сетка {g['bins'][0]}x{g['bins'][1]}x{g['bins'][2]})")
        print("-" * 78)
        tot, pw = m["totals"], m["powers"]
        kind, length = parse_units(su)
        hit = m["rates"]["hit_rate_Hz"]
        to_pulse = 1.0 if kind == "J" else 1.0 / hit          # величина файла -> за импульс
        tau_us = m["config"]["beam"]["pulse_us"]
        print(f"ЭНЕРГИЯ ЗА ОДИН ИМПУЛЬС ({tau_us:g} мкс), Дж")
        print(f"  первичные  : {tot['primary'] * to_pulse:,.3f}")
        print(f"  вторичные  : {tot['secondary'] * to_pulse:,.3f}   "
              f"({100*tot['secondary']/max(tot['primary'], 1e-300):.2f}% от первичных)")
        print(f"  ВСЕГО      : {pw['energy_per_pulse_J']:,.3f}")
        print(f"  потери за краем болванки: {tot['outside_fraction']*100:.4f}%")
        print("-" * 78)
        print("МОЩНОСТЬ — та же энергия, разные интервалы усреднения")
        print(f"  1) за длительность импульса ({tau_us:g} мкс) : "
              f"{pw['during_pulse_W']/1e3:>12,.2f} кВт   пиковая нагрузка")
        print(f"  2) за период источника ({m['rates']['rep_rate_Hz']:g} Гц)     : "
              f"{pw['per_period_W']/1e3:>12,.4f} кВт   на всю сборку")
        print(f"  3) за оборот колеса (N={m['rates']['n_pucks']} болванок)  : "
              f"{pw['per_puck_W']/1e3:>12,.4f} кВт   на одну болванку  <-- в картах")
        print("-" * 78)
        m_per_unit = _LEN_MM[length] * 1e-3
        peak_pulse = self.q_total.max() / m_per_unit ** 2 * to_pulse       # Дж/м² за импульс
        print(f"Поверхностная плотность, пик по карте:")
        print(f"  за импульс                    : {peak_pulse/1e4:>12,.2f} Дж/см²")
        print(f"  за длительность импульса      : "
              f"{peak_pulse/(tau_us*1e-6)/1e10:>12,.3f} МВт/см²")
        print(f"  за оборот колеса (на болванку): {peak_pulse*hit/1e4:>12,.0f} Вт/см²")
        print(f"  в единицах файла [{su}]  : {self.q_total.max():,.6g} "
              f"(первичные {self.q_primary.max():,.6g}, вторичные {self.q_secondary.max():,.6g})")
        og = m["output_grid"]
        print(f"  решётка мишени      : шаг {og['step_mm']*1e3:.3f} мкм"
              f"{' (огрублён автоматически)' if og['auto_coarsened'] else ''}, "
              f"{self.q_total.shape[0]} x {self.q_total.shape[1]}")
        if m.get("peak_volumetric") is not None:
            print(f"Объёмная плотность на оси пучка [{vu}]:")
            print(f"  пик (суммарно)      : {m['peak_volumetric']:,.6g} "
                  f"при x = {m['peak_volumetric_at_x_mm']:.4f} мм")
            print(f"  пик (первичные)     : {m['peak_volumetric_primary']:,.6g}")
        print("=" * 78)

    # ---- сохранение -------------------------------------------------------
    def save(self, base: str, compress: bool = False, dtype: str = "float32") -> Path:
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        npz = base.with_suffix(".npz")
        cast = np.dtype(dtype).type
        payload = dict(
            y_mm=self.y_mm.astype(np.float64),
            z_mm=self.z_mm.astype(np.float64),
            y_edges_mm=self.y_edges_mm.astype(np.float64),
            z_edges_mm=self.z_edges_mm.astype(np.float64),
            q_primary=self.q_primary.astype(cast),
            q_secondary=self.q_secondary.astype(cast),
            q_total=self.q_total.astype(cast),
            r_mm=self.r_mm.astype(np.float64),
            phi_rad=self.phi_rad.astype(np.float64),
            q_polar_total=self.q_polar_total.astype(cast),
            q_polar_primary=self.q_polar_primary.astype(cast),
            q_polar_secondary=self.q_polar_secondary.astype(cast),
            q_radial_mean=self.q_radial_mean.astype(np.float64),
            x_mm=self.x_mm.astype(np.float64),
            eps_axis_primary=self.eps_axis_primary.astype(np.float64),
            eps_axis_secondary=self.eps_axis_secondary.astype(np.float64),
            eps_axis_total=self.eps_axis_total.astype(np.float64),
            lin_depth_primary=self.lin_depth_primary.astype(np.float64),
            lin_depth_secondary=self.lin_depth_secondary.astype(np.float64),
            lin_depth_total=self.lin_depth_total.astype(np.float64),
            surface_units=np.array(self.surface_units),
            volume_units=np.array(self.volume_units),
            linear_units=np.array(self.linear_units),
            meta_json=np.array(json.dumps(self.meta, ensure_ascii=False, indent=2)),
        )
        (np.savez_compressed if compress else np.savez)(npz, **payload)
        base.with_suffix(".json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return npz


def load_wall_load(npz_path: str) -> WallLoad:
    """Читает результат, сохранённый WallLoad.save()."""
    with np.load(npz_path, allow_pickle=False) as d:
        return WallLoad(
            y_mm=d["y_mm"], z_mm=d["z_mm"],
            y_edges_mm=d["y_edges_mm"], z_edges_mm=d["z_edges_mm"],
            q_primary=d["q_primary"], q_secondary=d["q_secondary"], q_total=d["q_total"],
            r_mm=d["r_mm"], phi_rad=d["phi_rad"],
            q_polar_total=d["q_polar_total"], q_polar_primary=d["q_polar_primary"],
            q_polar_secondary=d["q_polar_secondary"], q_radial_mean=d["q_radial_mean"],
            x_mm=d["x_mm"], eps_axis_primary=d["eps_axis_primary"],
            eps_axis_secondary=d["eps_axis_secondary"], eps_axis_total=d["eps_axis_total"],
            lin_depth_primary=d["lin_depth_primary"],
            lin_depth_secondary=d["lin_depth_secondary"],
            lin_depth_total=d["lin_depth_total"],
            surface_units=str(d["surface_units"]), volume_units=str(d["volume_units"]),
            linear_units=str(d["linear_units"]),
            meta=json.loads(str(d["meta_json"])),
        )


# ============================================================================
#                            ОСНОВНОЙ РАСЧЁТ
# ============================================================================

def choose_output_step(grids: Sequence[VoxelGrid], cfg: Config) -> tuple[float, bool]:
    """Шаг общей решётки мишени.

    По умолчанию — самый мелкий поперечный воксель ПЕРВИЧНЫХ частиц, но не
    мельче, чем позволяет max_output_bins. Ограничение нужно, потому что размер
    вокселя задаётся охватом данных: у компактного набора (например, редких
    вторичных) он может оказаться в единицы микрон, что на мишени Ø50 мм дало бы
    решётку в десятки тысяч бинов по стороне.
    """
    if cfg.output_step_mm:
        return float(cfg.output_step_mm), False
    native = min(grids[0].voxel_size_mm[1], grids[0].voxel_size_mm[2])
    limit = 2.0 * cfg.target.radius_mm / cfg.max_output_bins
    if native < limit:
        return limit, True
    return native, False


def project_to_lattice(grid: VoxelGrid, cfg: Config, step: float, n_lat: int,
                       scale_surface: float, sweep_y_mm: float = 0.0):
    """Воксельная карта -> поверхностная плотность на решётке мишени.

    q(y, z) = ∫ eps dx = (Σ_x H) / (N_sim · dy · dz) · scale
    Шаг dx сокращается, поэтому разрешение по оси пучка на ответ не влияет.
    Затем компактный блок переносится на общую решётку (точное перераспределение
    по площади), дополняется нулями до размера мишени и свёртывается с пучком.
    """
    E = grid.H.sum(axis=0) / cfg.n_sim_protons          # МэВ/протон в нативном бине
    e_native = float(E.sum())

    ky_lo, ky_hi = _block_range(grid.edges[1], step, n_lat)
    kz_lo, kz_hi = _block_range(grid.edges[2], step, n_lat)
    y_block = (np.arange(ky_lo, ky_hi + 2) - 0.5) * step
    z_block = (np.arange(kz_lo, kz_hi + 2) - 0.5) * step

    E = _rebin_axis(E, grid.edges[1], y_block, axis=0)
    E = _rebin_axis(E, grid.edges[2], z_block, axis=1)
    if float(E.sum()) < 0.999 * e_native:
        print(f"[warning] {Path(grid.source).name}: часть данных лежит за пределами "
              f"мишени и отброшена ({100*(1 - E.sum()/e_native):.2f}% энерговыделения)")

    block = E / (step * step) * scale_surface           # поверхностная плотность
    n_bins = 2 * n_lat + 1
    q = expand_and_convolve(
        block, (ky_lo + n_lat, kz_lo + n_lat), (n_bins, n_bins), (step, step),
        cfg.beam.sigma_y_mm, cfg.beam.sigma_z_mm, sweep_y_mm,
    )
    return q


def compute_wall_load(cfg: Config) -> WallLoad:
    """Полный расчёт: CSV -> карта Y/Z (+ полярная) в заданных единицах."""
    t_start = time.time()
    kind, length = parse_units(cfg.units)
    vol_units = _fmt_unit(kind, length, 3)
    surf_units = _fmt_unit(kind, length, 2)

    # --- движение мишенной сборки -----------------------------------------
    sweep_mm = sweep_length_mm(cfg.target, cfg.beam)
    v_max = max_speed_m_per_s(cfg.target, cfg.beam)
    gap_mm = edge_gap_mm(cfg.target, cfg.beam)
    margin_mm = cfg.target.edge_margin_sigma * cfg.beam.sigma_y_mm
    if cfg.verbose and cfg.target.speed_m_per_s > 0:
        print(f"[motion] скорость {cfg.target.speed_m_per_s:.4g} м/с -> путь за импульс "
              f"{sweep_mm:.3f} мм, зазор до края {gap_mm:.3f} мм "
              f"({gap_mm/cfg.beam.sigma_y_mm:.2f} σy), предел {v_max:.4g} м/с")
    if gap_mm < margin_mm:
        print(f"[warning] зазор {gap_mm:.3f} мм меньше требуемых "
              f"{cfg.target.edge_margin_sigma:g} σy = {margin_mm:.3f} мм: "
              f"пучок заденет край болванки. Предельная скорость {v_max:.4g} м/с")

    # --- частота попаданий в ОДНУ болванку --------------------------------
    hit_rate_Hz = cfg.beam.rep_rate_Hz / max(cfg.target.n_pucks, 1)
    puck_duty = cfg.beam.pulse_us * 1e-6 * hit_rate_Hz

    # --- множитель перехода МэВ/протон -> рабочие единицы ------------------
    beam_factor = cfg.beam.protons_per_pulse * MEV_TO_J        # Дж на (МэВ/протон)
    if kind == "W":
        beam_factor *= hit_rate_Hz            # средняя мощность на одну болванку
    scale_surface = beam_factor * _LEN_MM[length] ** 2         # /мм² -> /длина²
    scale_volume = beam_factor * _LEN_MM[length] ** 3          # /мм³ -> /длина³
    scale_linear = beam_factor * _LEN_MM[length]               # /мм  -> /длина

    # --- вокселизация ------------------------------------------------------
    g_pri = voxelize_csv(cfg.primary_csv, cfg.bins, cfg.chunksize, cfg.cache_dir, cfg.verbose)
    if cfg.verbose:
        print("[primary]\n" + g_pri.describe())
    grids = [g_pri]

    g_sec = None
    if cfg.secondary_csv:
        g_sec = voxelize_csv(cfg.secondary_csv, cfg.bins, cfg.chunksize, cfg.cache_dir, cfg.verbose)
        if cfg.verbose:
            print("[secondary]\n" + g_sec.describe())
        grids.append(g_sec)

    # --- общая решётка мишени ----------------------------------------------
    step, coarsened = choose_output_step(grids, cfg)
    y_t, y_c, n_lat = canonical_lattice(cfg.target.radius_mm, step)
    z_t, z_c, _ = canonical_lattice(cfg.target.radius_mm, step)
    if cfg.verbose:
        note = " (огрублён под max_output_bins)" if coarsened else ""
        print(f"[grid] решётка мишени: шаг {step*1e3:.3f} мкм{note}, "
              f"{len(y_c)} x {len(z_c)} бинов "
              f"(~{len(y_c)*len(z_c)*8/2**20:.0f} МБ на карту)")
    if min(cfg.beam.sigma_y_mm, cfg.beam.sigma_z_mm) < 4 * step:
        print(f"[warning] шаг решётки {step:.4g} мм великоват для σ="
              f"{min(cfg.beam.sigma_y_mm, cfg.beam.sigma_z_mm):.4g} мм — "
              f"профиль пучка разрешён грубо")

    # --- проекция + свёртка -------------------------------------------------
    meta_grids: dict = {}
    q_pri = project_to_lattice(g_pri, cfg, step, n_lat, scale_surface, sweep_mm)
    meta_grids["primary"] = {
        "source": g_pri.source, "n_rows": g_pri.n_rows, "sum_de_mev": g_pri.sum_de_mev,
        "bins": list(g_pri.H.shape), "voxel_size_mm": list(g_pri.voxel_size_mm),
        "voxel_volume_mm3": g_pri.voxel_volume_mm3, "extent_mm": g_pri.extent_mm,
    }

    x_mm, eps_pri = axis_volumetric_profile(
        g_pri, cfg.n_sim_protons, cfg.beam.sigma_y_mm, cfg.beam.sigma_z_mm,
        scale_volume, sweep_mm)
    _, lin_pri = depth_linear_profile(g_pri, cfg.n_sim_protons, scale_linear)

    if g_sec is not None:
        q_sec = project_to_lattice(g_sec, cfg, step, n_lat, scale_surface, sweep_mm)
        meta_grids["secondary"] = {
            "source": g_sec.source, "n_rows": g_sec.n_rows, "sum_de_mev": g_sec.sum_de_mev,
            "bins": list(g_sec.H.shape), "voxel_size_mm": list(g_sec.voxel_size_mm),
            "voxel_volume_mm3": g_sec.voxel_volume_mm3, "extent_mm": g_sec.extent_mm,
        }
        x_sec, eps_sec_own = axis_volumetric_profile(
            g_sec, cfg.n_sim_protons, cfg.beam.sigma_y_mm, cfg.beam.sigma_z_mm,
            scale_volume, sweep_mm)
        eps_sec = (eps_sec_own if np.array_equal(x_sec, x_mm)
                   else np.interp(x_mm, x_sec, eps_sec_own, left=0.0, right=0.0))
        _, lin_sec_own = depth_linear_profile(g_sec, cfg.n_sim_protons, scale_linear)
        lin_sec = (lin_sec_own if np.array_equal(x_sec, x_mm)
                   else np.interp(x_mm, x_sec, lin_sec_own, left=0.0, right=0.0))
    else:
        q_sec = np.zeros_like(q_pri)
        eps_sec = np.zeros_like(eps_pri)
        lin_sec = np.zeros_like(lin_pri)

    dy_out = float(y_c[1] - y_c[0])
    dz_out = float(z_c[1] - z_c[0])
    cell = dy_out * dz_out / (_LEN_MM[length] ** 2)   # площадь ячейки в единицах length²

    total_before_mask = float((q_pri.sum() + q_sec.sum()) * cell)

    # --- маска круглой мишени ----------------------------------------------
    RR = np.hypot(y_c[:, None], z_c[None, :])
    inside = RR <= cfg.target.radius_mm
    if cfg.mask_outside_target:
        q_pri = np.where(inside, q_pri, 0.0)
        q_sec = np.where(inside, q_sec, 0.0)

    q_tot = q_pri + q_sec

    tot_pri = float(q_pri.sum() * cell)
    tot_sec = float(q_sec.sum() * cell)
    tot_all = tot_pri + tot_sec

    # ожидаемое полное значение (вся энергия, без потерь за краем мишени)
    expected = (g_pri.sum_de_mev + (g_sec.sum_de_mev if g_sec is not None else 0.0)) \
        / cfg.n_sim_protons * beam_factor
    outside_fraction = 1.0 - tot_all / expected if expected > 0 else 0.0

    # --- полярные координаты -----------------------------------------------
    r, phi, pol_tot = to_polar(q_tot, y_c, z_c, cfg.target.radius_mm, cfg.n_r, cfg.n_phi)
    _, _, pol_pri = to_polar(q_pri, y_c, z_c, cfg.target.radius_mm, cfg.n_r, cfg.n_phi)
    _, _, pol_sec = to_polar(q_sec, y_c, z_c, cfg.target.radius_mm, cfg.n_r, cfg.n_phi)
    q_radial = pol_tot.mean(axis=1)

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.as_dict(),
        "beam": {
            "protons_per_pulse": cfg.beam.protons_per_pulse,
            "protons_per_second": cfg.beam.protons_per_second,
            "duty_cycle": cfg.beam.duty_cycle,          # D = t/T, коэффициент заполнения
            "duty_ratio": cfg.beam.duty_ratio,          # S = T/t, скважность
            "average_current_mA": cfg.beam.average_current_mA,
            "fwhm_y_mm": cfg.beam.fwhm_y_mm,
            "fwhm_z_mm": cfg.beam.fwhm_z_mm,
        },
        "motion": {
            "speed_m_per_s": cfg.target.speed_m_per_s,
            "sweep_mm": sweep_mm,
            "edge_gap_mm": gap_mm,
            "edge_gap_sigma": gap_mm / cfg.beam.sigma_y_mm,
            "required_margin_mm": margin_mm,
            "max_speed_m_per_s": v_max,
            "axis": "y",
        },
        "rates": {
            "rep_rate_Hz": cfg.beam.rep_rate_Hz,
            "n_pucks": cfg.target.n_pucks,
            "hit_rate_Hz": hit_rate_Hz,
            "puck_duty_cycle": puck_duty,               # D для одной болванки
            "puck_duty_ratio": (1.0 / puck_duty) if puck_duty > 0 else None,
        },
        "grids": meta_grids,
        "units": {"surface": surf_units, "volume": vol_units,
                  "linear": _fmt_unit(kind, length, 1),
                  "coordinates": "mm", "angle": "rad"},
        "totals_units": ("J/импульс" if kind == "J"
                         else "W (усреднение по обороту колеса, на одну болванку)"),
        "powers": {
            "comment": ("одна и та же энергия, разные интервалы усреднения; "
                        "в картах и выгрузке используется per_puck_W"),
            "energy_per_pulse_J": None,      # заполняется ниже
            "during_pulse_W": None,          # 1) энергия / длительность импульса
            "per_period_W": None,            # 2) энергия / период источника (вся сборка)
            "per_puck_W": None,              # 3) энергия / (N периодов) — на одну болванку
        },
        "totals": {
            "primary": tot_pri,
            "secondary": tot_sec,
            "total_on_target": tot_all,
            "total_before_mask": total_before_mask,
            "expected_full": expected,
            "outside_fraction": outside_fraction,
        },
        "output_grid": {"step_mm": step, "dy_mm": dy_out, "dz_mm": dz_out,
                        "shape": list(q_tot.shape), "auto_coarsened": coarsened,
                        "n_r": cfg.n_r, "n_phi": cfg.n_phi},
        "peak_volumetric": float(np.max(eps_pri + eps_sec)),
        "peak_volumetric_primary": float(np.max(eps_pri)),
        "peak_volumetric_at_x_mm": float(x_mm[int(np.argmax(eps_pri + eps_sec))]),
        "runtime_s": None,
    }

    res = WallLoad(
        y_mm=y_c, z_mm=z_c, y_edges_mm=y_t, z_edges_mm=z_t,
        q_primary=q_pri, q_secondary=q_sec, q_total=q_tot,
        r_mm=r, phi_rad=phi,
        q_polar_total=pol_tot, q_polar_primary=pol_pri, q_polar_secondary=pol_sec,
        q_radial_mean=q_radial,
        x_mm=x_mm, eps_axis_primary=eps_pri, eps_axis_secondary=eps_sec,
        eps_axis_total=eps_pri + eps_sec,
        lin_depth_primary=lin_pri, lin_depth_secondary=lin_sec,
        lin_depth_total=lin_pri + lin_sec,
        surface_units=surf_units, volume_units=vol_units,
        linear_units=_fmt_unit(kind, length, 1), meta=meta,
    )
    e_pulse_J = tot_all / (1.0 if kind == "J" else hit_rate_Hz)
    res.meta["powers"].update({
        "energy_per_pulse_J": e_pulse_J,
        "during_pulse_W": e_pulse_J / (cfg.beam.pulse_us * 1e-6),
        "per_period_W": e_pulse_J * cfg.beam.rep_rate_Hz,
        "per_puck_W": e_pulse_J * hit_rate_Hz,
    })
    res.meta["runtime_s"] = time.time() - t_start

    if cfg.output:
        path = res.save(cfg.output, compress=cfg.compress, dtype=cfg.save_dtype)
        if cfg.verbose:
            print(f"[save] {path}  (+ {Path(cfg.output).with_suffix('.json')})")

    if cfg.verbose:
        res.summary()
        print(f"Время расчёта: {res.meta['runtime_s']:.1f} с")
    return res


# ============================================================================
#                      БЫСТРАЯ ВИЗУАЛИЗАЦИЯ (необязательная)
# ============================================================================

# Последовательная шкала (одна светлота -> тёмная, без «радуги») — как в остальных
# графиках проекта. Линии — категориальные цвета, различимые при дальтонизме.
_CMAP = "inferno"
_C_TOTAL = "#1a1a19"
_C_PRIMARY = "#2a78d6"
_C_SECONDARY = "#eb6834"


def plot_wall_load(res: WallLoad, log: bool = False, zoom_mm: float | None = None,
                   save: str | None = None, dpi: int = 130):
    """Обзорная картинка результата: карта Y/Z, полярная карта и профили."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    R = res.r_mm[-1]
    q = res.q_total
    vmax = float(q.max())
    norm = LogNorm(vmin=max(vmax * 1e-4, np.nextafter(0, 1)), vmax=vmax) if log else None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11), dpi=dpi)
    fig.suptitle(f"Тепловая нагрузка на стенку теплосъёма  •  "
                 f"{res.meta['totals']['total_on_target']:,.4g} "
                 f"{res.meta['totals_units']} на мишени Ø{2*R:.0f} мм",
                 fontsize=14, fontweight="bold")

    # --- (1) декартова карта -------------------------------------------------
    ax = axes[0, 0]
    im = ax.pcolormesh(res.y_edges_mm, res.z_edges_mm, q.T,
                       cmap=_CMAP, norm=norm, vmin=None if log else 0, vmax=None if log else vmax,
                       shading="auto", rasterized=True)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="white", lw=1.2, alpha=0.8))
    lim = zoom_mm or R
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="y (мм)", ylabel="z (мм)",
           title="Проекция Y/Z (интеграл по оси пучка)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=res.surface_units, fraction=0.046, pad=0.04)

    # --- (2) полярная карта --------------------------------------------------
    ax = fig.add_subplot(2, 2, 2, projection="polar")
    axes[0, 1].remove()
    phi_e = np.append(res.phi_rad, 2 * np.pi)
    dr = res.r_mm[1] - res.r_mm[0]
    r_e = np.append(res.r_mm - dr / 2, res.r_mm[-1] + dr / 2)
    r_e[0] = 0.0
    im = ax.pcolormesh(phi_e, r_e, res.q_polar_total,
                       cmap=_CMAP, norm=norm, vmin=None if log else 0,
                       vmax=None if log else vmax, shading="auto", rasterized=True)
    ax.set_rmax(zoom_mm or R)
    ax.set_title("Полярные координаты (r, φ)", pad=16)
    ax.grid(alpha=0.25, color="white", lw=0.6)
    fig.colorbar(im, ax=ax, label=res.surface_units, fraction=0.046, pad=0.10)

    # --- (3) радиальный профиль ---------------------------------------------
    ax = axes[1, 0]
    for arr, c, lab in ((res.q_polar_total.mean(axis=1), _C_TOTAL, "суммарно"),
                        (res.q_polar_primary.mean(axis=1), _C_PRIMARY, "первичные"),
                        (res.q_polar_secondary.mean(axis=1), _C_SECONDARY, "вторичные")):
        ax.plot(res.r_mm, arr, color=c, lw=2, label=lab)
    ax.set(xlabel="r (мм)", ylabel=res.surface_units,
           title="Радиальный профиль (среднее по азимуту)",
           xlim=(0, zoom_mm or R))
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.2, lw=0.6)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # --- (4) объёмная плотность вдоль оси пучка ------------------------------
    ax = axes[1, 1]
    for arr, c, lab in ((res.eps_axis_total, _C_TOTAL, "суммарно"),
                        (res.eps_axis_primary, _C_PRIMARY, "первичные"),
                        (res.eps_axis_secondary, _C_SECONDARY, "вторичные")):
        ax.plot(res.x_mm, arr, color=c, lw=2, label=lab)
    ax.set(xlabel="x (мм), ось пучка", ylabel=res.volume_units,
           title="Объёмная плотность на оси пучка")
    ax.grid(alpha=0.2, lw=0.6)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[plot] {save}")
    return fig


# ============================================================================
#                                  CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    d = Config(primary_csv="")
    p = argparse.ArgumentParser(
        description="Тепловая нагрузка на стенку теплосъёма (проекция Y/Z)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--primary", required=True, help="CSV с dE первичных частиц")
    p.add_argument("--secondary", default=None, help="CSV с dE вторичных частиц")
    p.add_argument("--n-sim", type=int, default=d.n_sim_protons, help="протонов в симуляции")
    p.add_argument("--bins", type=int, nargs=3, default=list(d.bins), metavar=("NX", "NY", "NZ"))
    p.add_argument("--sigma-y", type=float, default=d.beam.sigma_y_mm)
    p.add_argument("--sigma-z", type=float, default=d.beam.sigma_z_mm)
    p.add_argument("--current-mA", type=float, default=d.beam.peak_current_mA,
                   help="ток в импульсе")
    p.add_argument("--pulse-us", type=float, default=d.beam.pulse_us)
    p.add_argument("--rep-Hz", type=float, default=d.beam.rep_rate_Hz)
    p.add_argument("--diameter", type=float, default=d.target.diameter_mm, help="диаметр болванки, мм")
    p.add_argument("--speed", type=float, default=d.target.speed_m_per_s,
                   help="скорость движения болванки поперёк пучка (вдоль y), м/с")
    p.add_argument("--n-pucks", type=int, default=d.target.n_pucks,
                   help="число болванок в сборке (делит частоту попаданий в одну болванку)")
    p.add_argument("--edge-margin-sigma", type=float, default=d.target.edge_margin_sigma,
                   help="требуемый зазор от центра пучка до края болванки, в единицах sigma_y")
    p.add_argument("--units", default=d.units, help="J/mm3 | J/cm3 | W/mm3 | W/cm3 (и Дж/см3 и т.п.)")
    p.add_argument("--n-r", type=int, default=d.n_r)
    p.add_argument("--n-phi", type=int, default=d.n_phi)
    p.add_argument("--output-step", type=float, default=None,
                   help="шаг итоговой решётки, мм (по умолчанию — самый мелкий поперечный воксель)")
    p.add_argument("--max-output-bins", type=int, default=d.max_output_bins,
                   help="предел числа бинов итоговой решётки по стороне")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--compress", action="store_true")
    p.add_argument("-o", "--output", default=None, help="базовое имя файлов результата")
    return p


def main(argv: Sequence[str] | None = None) -> WallLoad:
    a = _build_parser().parse_args(argv)
    cfg = Config(
        primary_csv=a.primary, secondary_csv=a.secondary, n_sim_protons=a.n_sim,
        bins=tuple(a.bins),
        beam=BeamParams(a.sigma_y, a.sigma_z, a.current_mA, a.pulse_us, a.rep_Hz),
        target=TargetParams(a.diameter, a.speed, a.n_pucks, a.edge_margin_sigma),
        units=a.units, n_r=a.n_r, n_phi=a.n_phi,
        output=a.output, output_step_mm=a.output_step,
        max_output_bins=a.max_output_bins,
        cache_dir=None if a.no_cache else ".voxel_cache",
        compress=a.compress,
    )
    return compute_wall_load(cfg)


if __name__ == "__main__":
    main()
