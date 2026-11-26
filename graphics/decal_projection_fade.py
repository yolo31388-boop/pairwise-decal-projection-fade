"""贴花投影与深度。

六个入口分别对应题目里的六项修复：

===============================  ===================================================
``project_tangent_space``        投射按切线空间（不再随相机俯仰退化成长条）
``decal_depth_test``             贴花用场景深度做深度测试（不再穿过角色腿）
``pool_visibility_evict``        池按可见性淘汰（不再按固定 256 FIFO 覆盖）
``grow_instance_buffer``         实例缓冲按需扩容（不再写满就丢新贴花）
``normal_follow_object``         法线随物体（逆转置，不再世界空间混合）
``sort_batch_by_material``       批次按材质排序（同材质连续、状态切换最少）
===============================  ===================================================

其余入口组成一条可复现的贴花管线：``load -> project -> draw -> present``，
``validate_decal`` / ``metrics`` / ``audit`` / ``fallback`` / ``determinism``
做自检、指标、审计、降级与可重复性验证。

约定
----
* 矩阵按 **行向量** 存储：``v' = v @ M``（``v`` 为齐次行向量），即常见
  列向量写法的转置。``normal_follow_object`` 同时支持 ``convention="column"``。
* 深度缓冲取值 ``[0, 1]``，0 为近裁剪面；``depth_encoding="reversed"``
  表示反向 Z，比较方向自动翻转。
* 所有入口不抛异常：输入不合法时返回 ``ok=False`` 的报告。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "tangent_basis",
    "tangent_projection_matrix",
    "project_tangent_space",
    "decal_depth_test",
    "pool_visibility_evict",
    "grow_instance_buffer",
    "normal_follow_object",
    "sort_batch_by_material",
    "load",
    "project",
    "draw",
    "present",
    "validate_decal",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_EPS = 1e-9


# ======================================================================================
# 通用工具
# ======================================================================================
def _arr(value: Any, dtype: type = float) -> np.ndarray:
    """尽量转成 ndarray，失败返回空数组。"""
    if value is None:
        return np.empty((0,), dtype=dtype)
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return np.empty((0,), dtype=dtype)


def _num(value: Any, default: float = 0.0) -> float:
    """安全浮点：非数值/非有限一律回退。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _vec3(value: Any) -> np.ndarray | None:
    """解析成 3 维向量；不合法返回 None。"""
    arr = _arr(value)
    if arr.size != 3:
        return None
    arr = arr.reshape(3)
    return arr if np.all(np.isfinite(arr)) else None


def _normalize(value: Any, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> tuple[np.ndarray, bool]:
    """归一化；退化时回退 ``fallback`` 并返回 ``ok=False``。"""
    vec = _vec3(value)
    candidate = fallback if vec is None else vec
    norm = float(np.linalg.norm(candidate))
    if not math.isfinite(norm) or norm < _EPS:
        return np.asarray(fallback, dtype=float), False
    return np.asarray(candidate, dtype=float) / norm, vec is not None


def _as_list(value: Any) -> list[Any]:
    """宽松列表化：None 为空表，标量包成单元素，其余按可迭代展开。"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [value.item()] if value.ndim == 0 else list(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _rows(value: Any, width: int) -> np.ndarray:
    """把标量 / (n,) / (n,width) / 可迭代对象统一成 ``(n, width)``。"""
    if value is not None and not isinstance(value, (np.ndarray, list, tuple, Mapping, str, bytes)):
        if isinstance(value, Iterable):
            value = list(value)
    arr = _arr(value)
    if arr.size == 0:
        return np.empty((0, width), dtype=float)
    if arr.ndim == 0:
        return np.full((1, width), _num(arr), dtype=float)
    if arr.ndim == 1:
        if arr.size == width:
            return arr.reshape(1, width)
        return np.repeat(arr.reshape(-1, 1), width, axis=1)
    if arr.ndim == 2 and arr.shape[1] == width:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return np.repeat(arr, width, axis=1)
    if arr.size % width:
        return np.empty((0, width), dtype=float)
    return arr.reshape(-1, width)


def _per_item(value: Any, count: int, default: float) -> np.ndarray:
    """标量 / (count,) 参数铺开成 ``(count,)``。"""
    arr = _arr(value)
    flat = arr.reshape(-1) if arr.size else np.empty(0)
    if flat.size == 0:
        return np.full(count, default, dtype=float)
    if flat.size == 1:
        val = _num(flat[0], default)
        return np.full(count, val, dtype=float)
    if flat.size == count and np.all(np.isfinite(flat)):
        return np.asarray(flat, dtype=float)
    return np.full(count, default, dtype=float)


def _bools(value: Any, count: int, default: bool = True) -> np.ndarray:
    arr = _arr(value) if value is not None else np.empty(0)
    flat = arr.reshape(-1) if arr.size else np.empty(0)
    if flat.size == 0:
        return np.full(count, default, dtype=bool)
    if flat.size == 1:
        return np.full(count, bool(flat[0]), dtype=bool)
    if flat.size != count:
        return np.full(count, default, dtype=bool)
    return flat.astype(bool)


def _round(value: Any, digits: int = 9) -> Any:
    """浮点收敛到固定精度，保证不同路径的摘要可比。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        return round(val, digits) if math.isfinite(val) else None
    if isinstance(value, np.ndarray):
        return [_round(v, digits) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return value


def _digest(obj: Any) -> str:
    text = json.dumps(_round(obj), sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# ======================================================================================
# 1. 投射按切线空间
# ======================================================================================
def tangent_basis(normal: Any, up_hint: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由曲面法线构造正交切线基 ``(t, b, n)``。

    基只取决于曲面法线（``up_hint`` 仅消除绕法线的旋转二义性），与相机
    朝向无关，所以相机俯仰不会改变贴花在表面上的朝向与比例。
    """
    n, ok = _normalize(normal)
    up = _vec3(up_hint)
    if up is None or not ok:
        up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, n))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(up, n))) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
    t, _ = _normalize(np.cross(up, n))
    b, _ = _normalize(np.cross(n, t))
    return t, b, n


def _box_frame(axis: np.ndarray, up_hint: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由投射轴构造盒子的正交框架：``(side, up, axis)``。"""
    a, _ = _normalize(axis)
    up = _vec3(up_hint)
    if up is None or abs(float(np.dot(up, a))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(up, a))) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
    side, _ = _normalize(np.cross(up, a))
    up2, _ = _normalize(np.cross(a, side))
    return side, up2, a


def _extent_along(direction: np.ndarray, axes: Sequence[np.ndarray], halves: Sequence[float]) -> float:
    """单位立方体与过中心的平面相交后，沿 ``direction`` 的半宽——乘 2 即全宽。

    解 ``|λ (w·e_i)| <= h_i``，取各约束中最紧的那个：``λ_max = min_i h_i/|w·e_i|``。
    垂直于某个轴的平面不受该轴约束（分母为 0 时跳过）。
    """
    w, ok = _normalize(direction)
    if not ok:
        return 0.0
    best = math.inf
    for axis, half in zip(axes, halves):
        cos = abs(float(np.dot(w, axis)))
        if cos > 1e-9:
            best = min(best, half / cos)
    return 2.0 * best if math.isfinite(best) else 0.0


def _surface_footprint(
    normal: np.ndarray, axes: Sequence[np.ndarray], halves: Sequence[float]
) -> tuple[tuple[float, float], float | None]:
    """贴花盒在**曲面切平面**上的足迹尺寸与长宽比。

    沿投射轴在切平面上的投影方向量一次（拉伸方向），垂直于它再量一次。
    切线空间投射时两者相等、长宽比为 1；屏幕空间投射在斜面上时，拉伸
    方向会被 ``1/cosθ`` 拉长（同时受盒子厚度截断），这就是"弹孔变划痕"。
    """
    side, up, axis = axes
    n, _ = _normalize(normal)
    projected = axis - n * float(np.dot(axis, n))
    if float(np.linalg.norm(projected)) < 1e-9:
        w1, w2 = side, up
    else:
        w1, _ = _normalize(projected)
        w2, _ = _normalize(np.cross(n, w1))
    ext1 = _extent_along(w1, (side, up, axis), halves)
    ext2 = _extent_along(w2, (side, up, axis), halves)
    longest, shortest = max(ext1, ext2), min(ext1, ext2)
    aspect = (longest / shortest) if shortest > _EPS else None
    return (ext1, ext2), aspect


def tangent_projection_matrix(
    center: Any,
    normal: Any,
    size: float = 1.0,
    depth: float = 1.0,
    *,
    axis: Any = None,
    up_hint: Any = None,
    basis: Any = None,
) -> np.ndarray:
    """世界 -> 贴花单位立方体 ``[-1, 1]^3`` 的行向量矩阵。

    盒子横向为 ``size × size``，沿投射轴厚度为 ``depth``。``axis`` 缺省
    等于曲面法线（切线空间投射）；传入相机 forward 之类的方向就是被修复的
    屏幕空间投射，盒子横向随相机，斜面上足迹会按 ``1/cosθ`` 拉长。
    ``basis`` 可直接指定切线空间 ``(t, b, n)``。
    """
    n, _ = _normalize(normal)
    if basis is not None:
        # 调用方直接给框架：以 third 轴为投射轴，其余两轴正交化
        try:
            vecs = [_normalize(v)[0] for v in basis]
        except (TypeError, ValueError):
            vecs = []
        if len(vecs) == 3:
            side = vecs[0] - vecs[2] * float(np.dot(vecs[0], vecs[2]))
            side, _ = _normalize(side, vecs[1])
            up, _ = _normalize(np.cross(vecs[2], side), vecs[1])
            a = vecs[2]
        else:
            side, up, a = tangent_basis(n, up_hint)
    elif axis is None:
        side, up, a = tangent_basis(n, up_hint)
    else:
        axis_vec = _vec3(axis)
        side, up, a = _box_frame(axis_vec if axis_vec is not None else n, up_hint)
    half_size = max(_num(size, 1.0), _EPS) * 0.5
    half_depth = max(_num(depth, 1.0), _EPS) * 0.5
    c = _vec3(center)
    if c is None:
        c = np.zeros(3)

    mat = np.eye(4)
    mat[:3, 0] = side / half_size
    mat[:3, 1] = up / half_size
    mat[:3, 2] = a / half_depth
    mat[3, :3] = -np.array(
        [
            float(np.dot(c, side)) / half_size,
            float(np.dot(c, up)) / half_size,
            float(np.dot(c, a)) / half_depth,
        ]
    )
    return mat


def project_tangent_space(
    targets: Any = None,
    center: Any = None,
    normal: Any = None,
    size: Any = 1.0,
    depth: Any = 1.0,
    *,
    mode: str = "tangent",
    project_dir: Any = None,
    up_hint: Any = None,
    basis: Any = None,
) -> dict[str, Any]:
    """把世界空间点投射到贴花空间，返回逐点 ``u/v/w``、命中掩码与足迹。

    参数
    ----
    targets : (N, 3)
        待投射的点（也叫 gbuffer 世界坐标重建结果）。
    center, normal : (3,) 或 (M, 3)
        贴花中心与曲面法线。
    size, depth : 标量或 (M,)
        盒子横向边长与沿投射轴的厚度（世界单位）。
    mode
        ``"tangent"``（默认，修复后）：盒子轴 = 曲面法线，横向随切线空间；
        ``"screen"``（缺陷复现）：盒子轴 = ``project_dir``（缺省相机 -Z），
        横向随相机，斜面上足迹按 ``1/cosθ`` 拉长。
    basis
        直接给 ``(t, b, n)``，优先于自动构造。
    up_hint
        ``mode="screen"`` 时是相机 up；切线空间里只用来消除旋转二义性。

    返回
    ----
    ``hit`` / ``u`` / ``v`` / ``w`` / ``decal``（重叠时归属的贴花索引）/
    ``coverage``，以及每贴花的 ``matrix``、``stretch``（``1/cosθ``）、
    ``footprint``（命中点在**曲面切平面**上的外接尺寸）与 ``aspect``。
    斜面上被拉成长条时 ``aspect`` 会显著偏离 1。
    """
    report: dict[str, Any] = {
        "ok": True,
        "mode": str(mode).lower(),
        "hit": np.zeros(0, dtype=bool),
        "u": np.zeros(0, dtype=float),
        "v": np.zeros(0, dtype=float),
        "w": np.zeros(0, dtype=float),
        "decal": np.zeros(0, dtype=int),
        "coverage": np.zeros(0, dtype=int),
        "decals": [],
        "hits": 0,
        "total": 0,
        "warnings": [],
    }
    pts = _rows(targets, 3)
    if pts.shape[0] == 0:
        report["ok"] = False
        report["warnings"].append("no_targets")
        return report

    centers = _rows(center, 3)
    normals = _rows(normal, 3)
    if centers.shape[0] == 0 and basis is None:
        report["ok"] = False
        report["warnings"].append("no_decal")
        return report

    basis_vectors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if basis is not None:
        try:
            vecs = [_normalize(v)[0] for v in basis]
        except TypeError:
            vecs = []
        if len(vecs) == 3:
            basis_vectors = (vecs[0], vecs[1], vecs[2])
        else:
            report["warnings"].append("invalid_basis_ignored")

    if centers.shape[0] == 0:
        centers = np.zeros((1, 3))
    if normals.shape[0] == 0:
        normals = np.array([[0.0, 0.0, 1.0]])
    count = max(centers.shape[0], normals.shape[0])
    if centers.shape[0] == 1 and count > 1:
        centers = np.repeat(centers, count, axis=0)
    if normals.shape[0] == 1 and count > 1:
        normals = np.repeat(normals, count, axis=0)
    count = min(centers.shape[0], normals.shape[0])

    sizes = _per_item(size, count, 1.0)
    depths = _per_item(depth, count, 1.0)
    dirs = _rows(project_dir, 3) if project_dir is not None else np.empty((0, 3))

    mode = str(mode).lower()
    if mode not in {"tangent", "screen"}:
        report["warnings"].append(f"unknown_mode:{mode}_using_tangent")
        mode = "tangent"
    report["mode"] = mode

    n_pts = pts.shape[0]
    best_w = np.full(n_pts, np.inf)
    hit = np.zeros(n_pts, dtype=bool)
    u_out = np.full(n_pts, np.nan)
    v_out = np.full(n_pts, np.nan)
    w_out = np.full(n_pts, np.nan)
    owner = np.full(n_pts, -1, dtype=int)
    coverage = np.zeros(n_pts, dtype=int)

    for i in range(count):
        c_i = centers[i]
        n_i, n_ok = _normalize(normals[i])
        if not n_ok:
            report["warnings"].append(f"degenerate_normal:{i}")

        # 盒子的三个轴：切线空间用曲面法线做投射轴，屏幕空间用相机方向
        if basis_vectors is not None:
            box_axes = tuple(np.asarray(v, dtype=float) for v in basis_vectors)
            axis = box_axes[2]
            n = box_axes[2]
        elif mode == "screen":
            if dirs.shape[0]:
                axis = dirs[i % dirs.shape[0]]
            else:
                axis = np.array([0.0, 0.0, -1.0])  # 相机默认看向 -Z
            axis, axis_ok = _normalize(axis, n_i)
            if not axis_ok:
                axis = n_i
            box_axes = _box_frame(axis, up_hint)
            n = n_i  # 屏幕空间下曲面法线仍是 n_i，不是盒子轴
        else:
            box_axes = tangent_basis(n_i, up_hint)
            axis = box_axes[2]
            n = box_axes[2]
        t, b = box_axes[0], box_axes[1]

        cos_theta = abs(float(np.dot(axis, n)))
        stretch = 1.0 / max(cos_theta, 1.0 / 4096.0)
        mat = tangent_projection_matrix(
            c_i, n, sizes[i], depths[i], axis=axis, up_hint=up_hint, basis=box_axes
        )
        # 盒半尺寸，顺序与 box_axes 的 (side, up, axis) 一致
        halves = (sizes[i] * 0.5, sizes[i] * 0.5, depths[i] * 0.5)

        local = np.hstack([pts, np.ones((n_pts, 1))]) @ mat
        inside = np.all(np.abs(local[:, :3]) <= 1.0 + 1e-12, axis=1)
        footprint, aspect = _surface_footprint(n, box_axes, halves)
        if inside.any():
            coverage += inside.astype(int)

            w_local = local[:, 2]
            take = inside & (np.abs(w_local) < best_w - 1e-12)
            if take.any():
                best_w[take] = np.abs(w_local[take])
                u_out[take] = local[take, 0] * 0.5 + 0.5
                v_out[take] = local[take, 1] * 0.5 + 0.5
                w_out[take] = w_local[take] * depths[i] * 0.5
                owner[take] = i
                hit[take] = True

        report["decals"].append(
            {
                "index": i,
                "center": c_i,
                "normal": n,
                "axis": axis,
                "basis": box_axes,
                "size": float(sizes[i]),
                "depth": float(depths[i]),
                "matrix": mat,
                "stretch": float(stretch),
                "grazing": bool(cos_theta < 0.15),
                "hits": int(inside.sum()),
                "footprint": footprint,
                "aspect": aspect,
            }
        )

    report.update(
        {
            "hit": hit,
            "u": u_out,
            "v": v_out,
            "w": w_out,
            "decal": owner,
            "coverage": coverage,
            "hits": int(hit.sum()),
            "total": int(n_pts),
        }
    )
    return report


# ======================================================================================
# 2. 贴花参与深度
# ======================================================================================
def _depth_buffer(value: Any) -> tuple[np.ndarray, int, int]:
    """把深度缓冲的多种写法统一成 ``(h, w)`` 数组。

    接受 ``(h, w)`` 数组、扁平 ``w*h`` 数组（尽力推断成方形）以及带
    ``width`` / ``height`` 的映射。无法确定尺寸时返回空数组，由调用方报错。
    """
    if isinstance(value, Mapping):
        w = int(_num(value.get("width", 0)))
        h = int(_num(value.get("height", 0)))
        arr = _arr(value.get("data", value.get("depth")))
        if arr.ndim == 1 and w > 0 and h > 0 and arr.size == w * h:
            arr = arr.reshape(h, w)
        elif arr.ndim == 2 and (w == 0 or h == 0):
            w, h = int(arr.shape[1]), int(arr.shape[0])
        return (arr, w, h) if arr.ndim == 2 and w > 0 and h > 0 else (np.empty((0, 0)), 0, 0)
    arr = _arr(value)
    if arr.ndim == 1 and arr.size > 1:
        side = int(round(math.sqrt(arr.size)))
        if side * side == arr.size:
            arr = arr.reshape(side, side)
    if arr.ndim != 2:
        return np.empty((0, 0)), 0, 0
    return arr, int(arr.shape[1]), int(arr.shape[0])


def decal_depth_test(
    scene_depth: Any = None,
    decal_world: Any = None,
    view_proj: Any = None,
    *,
    camera: Any = None,
    inv_view_proj: Any = None,
    width: int | None = None,
    height: int | None = None,
    tolerance: float = 1e-4,
    bias: float = 0.0,
    policy: str = "less_equal",
    depth_encoding: str = "float01",
) -> dict[str, Any]:
    """用**场景深度**判定贴花片元是否可见。

    旧实现拿贴花自身深度和自己比，恒通过，贴花于是画到角色腿上。这里把
    贴花点投到屏幕像素，取该像素的场景深度比较：

    * ``less_equal``：贴花深度 ``<= 场景深度 + tolerance - bias`` 则可见；
    * 反向 Z（``depth_encoding="reversed"``）自动翻转比较方向；
    * ``policy="always"`` 复现旧的"不做深度测试"行为，仅供回归对比。

    相机矩阵为**世界 -> 裁剪**的行向量矩阵 ``v_clip = [x y z 1] @ view_proj``
    （透视投影请传列向量矩阵的转置）。也可传 ``inv_view_proj``，这里自动求逆
    后再用。``camera`` 可以是矩阵，或含 ``view_proj`` / ``matrix`` 的映射。

    像素取整用向下取整（与光栅化一致），屏幕外的点记入 ``offscreen``；
    采样到非有限场景深度的点记入 ``no_scene`` 并保守判为不可见。
    """
    report: dict[str, Any] = {
        "ok": True,
        "tested": False,
        "policy": str(policy),
        "depth_encoding": str(depth_encoding),
        "pass": np.zeros(0, dtype=bool),
        "reject": np.zeros(0, dtype=bool),
        "offscreen": np.zeros(0, dtype=bool),
        "no_scene": np.zeros(0, dtype=bool),
        "decal_depth": np.zeros(0, dtype=float),
        "scene_depth": np.zeros(0, dtype=float),
        "index": (np.zeros(0, dtype=int), np.zeros(0, dtype=int)),
        "tolerance": _num(tolerance),
        "bias": _num(bias),
        "width": int(width or 0),
        "height": int(height or 0),
        "passed": 0,
        "rejected": 0,
        "warnings": [],
    }
    pts = _rows(decal_world, 3)
    if pts.shape[0] == 0 or scene_depth is None:
        report["ok"] = False
        report["warnings"].append("no_inputs")
        return report

    buf, w, h = _depth_buffer(scene_depth)
    if w == 0 or h == 0:
        report["ok"] = False
        report["warnings"].append("invalid_scene_depth")
        return report
    if width:
        w = int(width)
    if height:
        h = int(height)
    if buf.shape != (h, w):
        try:
            buf = buf.reshape(h, w)
        except ValueError:
            report["ok"] = False
            report["warnings"].append("scene_depth_shape_mismatch")
            return report
    report["width"], report["height"] = w, h

    n = pts.shape[0]
    policy = str(policy).lower()
    if policy == "always":
        report.update(
            {
                "pass": np.ones(n, dtype=bool),
                "decal_depth": np.full(n, np.nan),
                "scene_depth": np.full(n, np.nan),
                "index": (np.zeros(n, dtype=int), np.zeros(n, dtype=int)),
                "passed": n,
            }
        )
        report["warnings"].append("depth_test_disabled")
        return report
    if policy not in {"less_equal", "greater_equal"}:
        report["warnings"].append(f"unknown_policy:{policy}_using_less_equal")
        policy = "less_equal"

    mat: Any = view_proj
    if mat is None and inv_view_proj is not None:
        # 调用方给的是逆矩阵，这里求逆得到世界 -> 裁剪
        try:
            mat = np.linalg.inv(_arr(inv_view_proj).reshape(4, 4))
        except (np.linalg.LinAlgError, ValueError):
            report["ok"] = False
            report["warnings"].append("singular_inv_view_proj")
            return report
    if mat is None and isinstance(camera, Mapping):
        mat = camera.get("view_proj", camera.get("matrix"))
    elif mat is None:
        mat = camera
    if mat is None:
        report["ok"] = False
        report["warnings"].append("no_camera")
        return report
    mat = _arr(mat)
    if mat.size == 1:
        mat = np.eye(4)
    if mat.size != 16:
        report["ok"] = False
        report["warnings"].append("invalid_camera_matrix")
        return report
    mat = mat.reshape(4, 4)

    clip = np.hstack([pts, np.ones((n, 1))]) @ mat
    wclip = clip[:, 3]
    valid = np.abs(wclip) > _EPS
    ndc = clip[:, :3] / np.where(valid, wclip, 1.0)[:, None]
    fx = ndc[:, 0] * 0.5 + 0.5
    fy = 0.5 - ndc[:, 1] * 0.5  # 行号向下增长
    fz = ndc[:, 2]

    reversed_z = str(depth_encoding).lower() in {"reversed", "reverse", "reversed_z"}
    if reversed_z and policy == "less_equal":
        policy = "greater_equal"
        report["warnings"].append("reversed_z_inverted_test")
    report["policy"] = policy
    d = fz if reversed_z else fz * 0.5 + 0.5

    inside = (
        valid
        & (fx >= 0.0)
        & (fx < 1.0)
        & (fy >= 0.0)
        & (fy < 1.0)
        & (fz >= 0.0)
        & (fz <= 1.0)
    )
    ix = np.zeros(n, dtype=int)
    iy = np.zeros(n, dtype=int)
    if inside.any():
        ix[inside] = np.floor(np.clip(fx[inside], 0.0, np.nextafter(1.0, 0.0)) * w).astype(int)
        iy[inside] = np.floor(np.clip(fy[inside], 0.0, np.nextafter(1.0, 0.0)) * h).astype(int)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    sampled = np.full(n, np.nan)
    if inside.any():
        sampled[inside] = buf[iy[inside], ix[inside]]
    # 场景深度缺失（离屏点采样到 NaN 等）：保守判为不可见，并单独统计
    no_scene = inside & ~np.isfinite(sampled)
    has_scene = inside & ~no_scene
    if no_scene.any():
        report["warnings"].append("missing_scene_depth")

    if policy == "greater_equal":
        passed = has_scene & (d >= sampled - _num(tolerance) + _num(bias))
    else:
        passed = has_scene & (d <= sampled + _num(tolerance) - _num(bias))

    report.update(
        {
            "tested": True,
            "pass": passed,
            "reject": has_scene & ~passed,
            "offscreen": ~inside,
            "no_scene": no_scene,
            "decal_depth": d,
            "scene_depth": sampled,
            "index": (ix, iy),
            "passed": int(passed.sum()),
            "rejected": int((has_scene & ~passed).sum()),
        }
    )
    return report


# ======================================================================================
# 3. 贴花池按可见性淘汰
# ======================================================================================
def _entry_id(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return entry.get("id")
    return getattr(entry, "id", entry)


def _hashable_id(value: Any) -> str:
    """id 可能不可哈希（list/dict），统一成字符串参与集合比对。"""
    return str(value)


def _entry_born(entry: Any, fallback: int = 0) -> int:
    if isinstance(entry, Mapping):
        return int(_num(entry.get("born", entry.get("frame", fallback)), fallback))
    return int(_num(getattr(entry, "born", fallback), fallback))


def _entry_visible(entry: Any, frame_visible: set[str] | None) -> bool:
    """本帧可见性：显式集合优先（按字符串 id 比对），其次读条目的 visible 标记。"""
    if frame_visible is not None:
        return _hashable_id(_entry_id(entry)) in frame_visible
    if isinstance(entry, Mapping):
        return bool(entry.get("visible", True))
    return bool(getattr(entry, "visible", True))


def _age(entry: Any) -> tuple[int, str]:
    return (_entry_born(entry), str(_entry_id(entry)))


def pool_visibility_evict(
    pool: Any = None,
    candidates: Any = None,
    *,
    visible: Iterable[Any] | None = None,
    capacity: int = 256,
    now: int = 0,
    allow_visible_eviction: bool = True,
) -> dict[str, Any]:
    """按可见性回收贴花池槽位。

    旧实现是固定 256 的 FIFO 环，写满就覆盖最老的，玩家回头时墙上的弹孔
    已经没了。修复后的规则：

    1. **不可见贴花优先回收**：需要腾槽位时先回收本帧不可见（离屏/被剔除）
       的贴花，按 ``(born, id)`` 从老到新——它们才是"暂时看不到"的那批；
    2. 池内有可用槽位时，不可见贴花**不会**被回收，玩家回头仍能看到；
    3. 池里全是可见贴花仍不够时，才淘汰最老的可见贴花（``evicted_visible``）；
    4. ``allow_visible_eviction=False`` 为无损模式：宁可丢弃超出容量的新贴花
       （``dropped``）也不覆盖屏幕上的贴花。

    新贴花总按提交顺序入池。``evicted_avoidable`` 单独记录"池里明明还有
    不可见贴花却淘汰了可见贴花"的次数，正常情况下恒为空。
    """
    cap = int(_num(capacity, 0))
    report: dict[str, Any] = {
        "ok": True,
        "capacity": cap,
        "pool": [],
        "accepted": [],
        "recycled": [],
        "evicted": [],
        "evicted_visible": [],
        "evicted_avoidable": [],
        "dropped": [],
        "occupancy": {},
        "frame_visible": [],
        "warnings": [],
    }
    if cap <= 0:
        report["ok"] = False
        report["warnings"].append("invalid_capacity")
        return report
    if cap > 256:
        report["warnings"].append("capacity_exceeds_budget:256")

    frame_visible: set[str] | None = None
    if visible is not None:
        # id 可能是不可哈希对象；统一按 str 比对，与 _entry_visible 一致
        frame_visible = {_hashable_id(v) for v in _as_list(visible)}
    entries = list(_as_list(pool))
    cands = _as_list(candidates)
    report["frame_visible"] = sorted(str(v) for v in (frame_visible or ()))

    # 用下标操作，避免 id() 在重复的小整数/驻留字符串上碰撞
    result = list(entries)
    deficit = max(0, len(result) + len(cands) - cap)
    if deficit:
        # 1) 先回收不可见（离屏/被剔除）的，从老到新
        invisible = sorted(
            (idx for idx, e in enumerate(result) if not _entry_visible(e, frame_visible)),
            key=lambda idx: _age(result[idx]),
        )
        drop = set(invisible[:deficit])
        report["recycled"] = [_entry_id(result[i]) for i in sorted(drop)]
        result = [e for i, e in enumerate(result) if i not in drop]
        deficit -= len(drop)
        # 2) 仍不够才动可见的（最老的先走）
        if deficit and allow_visible_eviction:
            still_visible = sorted(
                (idx for idx, e in enumerate(result) if _entry_visible(e, frame_visible)),
                key=lambda idx: _age(result[idx]),
            )
            drop = set(still_visible[:deficit])
            evicted = [_entry_id(result[i]) for i in sorted(drop)]
            result = [e for i, e in enumerate(result) if i not in drop]
            report["evicted"] = evicted
            report["evicted_visible"] = evicted
            # 池里还有不可见贴花却仍淘汰可见的，才是真正可避免的 bug
            has_invisible_left = any(not _entry_visible(e, frame_visible) for e in result)
            if has_invisible_left:
                report["evicted_avoidable"] = list(evicted)
            deficit -= len(evicted)
        if deficit:
            report["warnings"].append("pool_full_visible")

    room = max(0, cap - len(result))
    accepted: list[Any] = []
    for offset, cand in enumerate(cands):
        if len(accepted) >= room:
            report["dropped"].append(_entry_id(cand))
            continue
        if isinstance(cand, Mapping):
            item = dict(cand)
            item.setdefault("id", f"new{offset}")
            item.setdefault("born", int(now) + offset)
            item.setdefault("visible", True)
        else:
            item = cand
        accepted.append(item)
    if report["dropped"]:
        report["warnings"].append("dropped_new_decals")

    report["accepted"] = [_entry_id(e) for e in accepted]
    report["pool"] = result + accepted
    visible_now = sum(1 for e in report["pool"] if _entry_visible(e, frame_visible))
    report["occupancy"] = {
        "used": len(report["pool"]),
        "capacity": cap,
        "visible": int(visible_now),
        "free": int(cap - len(report["pool"])),
    }
    return report


# ======================================================================================
# 4. 实例缓冲按需扩容
# ======================================================================================
def grow_instance_buffer(
    buffer: Any = None,
    incoming: Any = 0,
    *,
    growth: float = 2.0,
    alignment: int = 256,
    stride: int = 64,
    max_capacity: int | None = None,
) -> dict[str, Any]:
    """写入前保证容量，按需扩容而不是丢新实例。

    旧实现把"写满"当丢弃条件，连射后半段没有弹孔。这里按
    ``used + incoming`` 计算需求，不足则几何增长（可设上限），并按
    ``alignment`` 字节对齐；容量只增不减，避免每帧反复重建。
    """
    if isinstance(buffer, (int, float, np.integer, np.floating)):
        cap, used, grow_events = max(0, int(_num(buffer, 0))), 0, 0
    elif isinstance(buffer, Mapping):
        cap = int(_num(buffer.get("capacity", buffer.get("size", 0)), 0))
        used = int(_num(buffer.get("used", buffer.get("count", 0)), 0))
        grow_events = int(_num(buffer.get("grow_events", 0), 0))
    else:
        cap, used, grow_events = 0, 0, 0
    cap = max(0, cap)
    used = max(0, used)
    if isinstance(incoming, (int, np.integer)):
        incoming_count = max(0, int(incoming))
    else:
        incoming_count = len(_as_list(incoming))
    stride = max(1, int(_num(stride, 64)))
    alignment = max(1, int(_num(alignment, 256)))
    growth = max(1.0, _num(growth, 2.0))

    required = used + incoming_count
    report: dict[str, Any] = {
        "ok": True,
        "capacity_before": cap,
        "capacity": cap,
        "used": min(used, cap),
        "used_after": min(used, cap),
        "incoming": incoming_count,
        "required": required,
        "grown": False,
        "grow_events": grow_events,
        "stride": stride,
        "alignment": alignment,
        "bytes_before": cap * stride,
        "bytes_after": cap * stride,
        "reserved": 0,
        "aligned": True,
        "dropped": 0,
        "warnings": [],
    }
    if required <= cap:
        report["used"] = used
        report["used_after"] = required
        report["reserved"] = cap - required
        return report

    target = required
    if max_capacity is not None:
        target = min(target, max(0, int(max_capacity)))
        if required > target:
            report["dropped"] = required - target
            report["warnings"].append("max_capacity_reached")

    if target <= cap:
        report["used"] = min(used, cap)
        report["used_after"] = min(required, cap)
        report["dropped"] = max(report["dropped"], required - cap)
        report["reserved"] = 0
        if report["dropped"] and "max_capacity_reached" not in report["warnings"]:
            report["warnings"].append("max_capacity_reached")
        return report

    new_cap = max(cap, 1)
    new_cap = max(int(math.ceil(new_cap * growth)), target)
    if alignment > 1:
        # 保证总字节数对齐：capacity 必须是 lcm(stride, alignment)/stride 的整数倍
        unit = max(1, math.lcm(stride, alignment) // stride)
        new_cap = int(math.ceil(new_cap / unit)) * unit
    if max_capacity is not None:
        new_cap = min(new_cap, max(int(max_capacity), target))

    report.update(
        {
            "capacity": new_cap,
            "grown": True,
            "grow_events": grow_events + 1,
            "bytes_after": new_cap * stride,
            "used": min(used, new_cap),
            "used_after": min(required, new_cap),
            "reserved": max(0, new_cap - min(required, new_cap)),
        }
    )
    report["bytes_after"] = new_cap * stride
    report["aligned"] = alignment <= 1 or (new_cap * stride) % alignment == 0
    if not report["aligned"]:
        report["warnings"].append("unaligned_capacity")
    if new_cap < required:
        report["dropped"] = required - new_cap
        report["warnings"].append("still_short_after_growth")
    return report


# ======================================================================================
# 5. 法线随物体
# ======================================================================================
def normal_follow_object(
    model: Any = None,
    normal_local: Any = None,
    *,
    convention: str = "row",
    normalize: bool = True,
) -> dict[str, Any]:
    """把物体空间的贴花法线变换到世界空间（逆转置）。

    旧实现把法线当方向在世界空间直接混合，物体一转法线不跟着转，光照方向
    反了、弹孔看起来是凸的。正确做法是用模型矩阵左上 3x3 的**逆转置**
    ``inv(M3).T``：

    * ``convention="row"``（本模块约定，``p_w = p_l @ M``）：``n' = n @ inv(M3).T``；
    * ``convention="column"``（``p_w = M @ p_l``）：``n' = inv(M3).T @ n``。

    纯旋转时 ``inv(M3).T == M3``，所以法线跟着物体一起转；非均匀缩放下
    逆转置仍保证法线与表面垂直。镜像（``det < 0``）置 ``mirrored``。
    """
    report: dict[str, Any] = {
        "ok": True,
        "normal": np.zeros(3),
        "normal_matrix": np.eye(4),
        "det": 1.0,
        "mirrored": False,
        "normalized": bool(normalize),
        "transformed": False,
        "warnings": [],
    }
    mat = _arr(model)
    if mat.size == 9:
        mat = mat.reshape(3, 3)
    elif mat.size == 16:
        mat = mat.reshape(4, 4)
    else:
        report["ok"] = False
        report["warnings"].append("invalid_model_matrix")
        return report
    normal = _vec3(normal_local)
    if normal is None:
        report["ok"] = False
        report["warnings"].append("invalid_normal")
        return report

    m3 = np.asarray(mat[:3, :3], dtype=float)
    det = float(np.linalg.det(m3))
    report["det"] = det
    report["mirrored"] = det < 0.0
    conv = str(convention).lower()
    if conv not in {"row", "column", "col"}:
        report["warnings"].append(f"unknown_convention:{conv}_using_row")
        conv = "row"

    try:
        inv = np.linalg.inv(m3)
        if not np.all(np.isfinite(inv)):
            raise np.linalg.LinAlgError("non-finite inverse")
        n_mat = inv.T
    except np.linalg.LinAlgError:
        # 退化矩阵：用最优正交近似（极点分解）保证至少跟随旋转
        u_svd, _, vt_svd = np.linalg.svd(m3)
        n_mat = u_svd @ vt_svd
        report["warnings"].append("singular_model_matrix_orthonormalized")

    out = normal @ n_mat if conv == "row" else n_mat @ normal

    if not np.all(np.isfinite(out)):
        report["ok"] = False
        report["warnings"].append("non_finite_result")
        out = normal
    if normalize:
        out, ok = _normalize(out, tuple(normal))
        if not ok:
            report["warnings"].append("degenerate_normal")
    report["normal"] = out
    if conv == "row":
        report["normal_matrix"][:3, :3] = n_mat
    else:
        report["normal_matrix"][:3, :3] = n_mat.T
    report["transformed"] = bool(np.any(np.abs(m3 - np.eye(3)) > 1e-12))
    return report


# ======================================================================================
# 6. 批次按材质排序
# ======================================================================================
def _instance_material(inst: Any, key: str | None = None) -> Any:
    if isinstance(inst, Mapping):
        if key is not None:
            return inst.get(key, inst.get("material", 0))
        return inst.get("material", inst.get("mat", 0))
    if isinstance(inst, (tuple, list)) and inst:
        return inst[0]
    return getattr(inst, "material", 0)


def sort_batch_by_material(
    instances: Any = None,
    *,
    material_order: Sequence[Any] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """按材质重排实例：同材质连续，绘制状态切换最少。

    稳定排序：同材质内保持提交顺序，结果只由输入决定，因此可复现。
    ``material_order`` 指定优先顺序，未列出的材质按首次出现顺序排在后面。
    """
    items = _as_list(instances)
    report: dict[str, Any] = {
        "ok": True,
        "instances": items,
        "batches": [],
        "batch_count": 0,
        "materials": [],
        "state_changes": 0,
        "state_changes_before": 0,
        "warnings": [],
    }
    if not items:
        return report

    def mat_of(inst: Any) -> Any:
        return _instance_material(inst, key)

    def mat_key(inst_or_mat: Any, is_material: bool = False) -> Any:
        """材质键可能不可哈希（列表/字典），统一降级为可哈希表示。"""
        m = inst_or_mat if is_material else mat_of(inst_or_mat)
        try:
            hash(m)
            return m
        except TypeError:
            return repr(m)

    ranks: dict[Any, int] = {}
    for m in material_order or []:
        ranks.setdefault(mat_key(m, True), len(ranks))
    for inst in items:
        ranks.setdefault(mat_key(inst), len(ranks))

    order = sorted(range(len(items)), key=lambda i: ranks[mat_key(items[i])])  # 稳定
    ordered = [items[i] for i in order]

    def transitions(seq: list[Any]) -> int:
        changes = 0
        prev = object()
        for inst in seq:
            m = mat_key(inst)
            if m != prev:
                changes += 1
                prev = m
        return max(0, changes - 1)

    batches: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(ordered) + 1):
        if i == len(ordered) or mat_key(ordered[i]) != mat_key(ordered[start]):
            batches.append(
                {
                    "material": mat_of(ordered[start]),
                    "start": start,
                    "end": i,
                    "count": i - start,
                    "instances": ordered[start:i],
                }
            )
            start = i

    report.update(
        {
            "instances": ordered,
            "batches": batches,
            "batch_count": len(batches),
            "materials": [b["material"] for b in batches],
            "state_changes": transitions(ordered),
            "state_changes_before": transitions(items),
        }
    )
    return report


# ======================================================================================
# 管线：load / project / draw / present
# ======================================================================================
_DATA_KEYS = ("decals", "targets", "instances", "visible", "decal_world", "scene_depth", "camera")


def _new_state(config: Any = None) -> dict[str, Any]:
    if isinstance(config, Mapping):
        cfg = dict(config)
    elif isinstance(config, (int, np.integer)) and int(config) > 0:
        cfg = {"capacity": int(config)}
    else:
        cfg = {}
    material_order = cfg.get("material_order", [])
    state: dict[str, Any] = {
        "config": {
            "capacity": max(1, int(_num(cfg.get("capacity", 256), 256))),
            "mode": str(cfg.get("mode", "tangent")),
            "depth_encoding": str(cfg.get("depth_encoding", "float01")),
            "tolerance": _num(cfg.get("tolerance", 1e-4), 1e-4),
            "bias": _num(cfg.get("bias", 0.0)),
            "allow_visible_eviction": bool(cfg.get("allow_visible_eviction", True)),
            "material_order": list(material_order) if isinstance(material_order, (list, tuple)) else [],
        },
        "frame": 0,
        "presented": 0,
        "pool": [],
        "instances": [],
        "decals": [],
        "targets": None,
        "visible": None,
        "decal_world": None,
        "scene_depth": None,
        "camera": None,
        "buffer": {
            "capacity": int(_num(cfg.get("buffer_capacity", 0), 0)),
            "used": 0,
            "stride": max(1, int(_num(cfg.get("stride", 64), 64))),
            "alignment": max(1, int(_num(cfg.get("alignment", 256), 256))),
            "grow_events": 0,
        },
        "projected": [],
        "last_draw": None,
        "last_present": None,
        "history": [],
        "stats": {
            "drawn": 0,
            "depth_passed": 0,
            "depth_rejected": 0,
            "depth_skipped_frames": 0,
            "recycled": 0,
            "evicted": 0,
            "evicted_visible": 0,
            "evicted_avoidable": 0,
            "dropped_decals": 0,
            "dropped_instances": 0,
            "buffer_grow_events": 0,
            "batches": 0,
            "materials": [],
            "frames": 0,
        },
        "warnings": [],
    }
    for name in _DATA_KEYS:
        if name in cfg:
            value = cfg[name]
            state[name] = list(value) if name in {"decals", "instances"} and value is not None else value
    return state


def log_warning(state: dict[str, Any], *warnings: str) -> None:
    """去重写入状态告警。"""
    for warn in warnings:
        if warn and warn not in state["warnings"]:
            state["warnings"].append(warn)


def load(config: Any = None) -> dict[str, Any]:
    """初始化贴花管线：池、实例缓冲、深度缓冲、相机与统计。"""
    state = _new_state(config)
    state["loaded"] = True
    state["ok"] = True
    return state


def _sub_map(obj: Any, name: str) -> Mapping[str, Any]:
    """安全取子映射：字段不是映射时返回空映射，避免 ``.get`` 打在字符串上。"""
    if isinstance(obj, Mapping):
        value = obj.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _sub_list(obj: Any, name: str) -> list[Any]:
    """安全取子列表：非列表时返回空列表。"""
    if isinstance(obj, Mapping):
        value = obj.get(name)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


_STATE_SHAPE: dict[str, tuple[str, ...]] = {
    "config": (
        "capacity",
        "mode",
        "depth_encoding",
        "tolerance",
        "bias",
        "allow_visible_eviction",
        "material_order",
    ),
    "stats": (
        "drawn",
        "depth_passed",
        "depth_rejected",
        "depth_skipped_frames",
        "recycled",
        "evicted",
        "evicted_visible",
        "evicted_avoidable",
        "dropped_decals",
        "dropped_instances",
        "buffer_grow_events",
        "batches",
        "materials",
        "frames",
    ),
    "buffer": ("capacity", "used", "stride", "alignment", "grow_events"),
}


def _coerce_state(state: Any) -> dict[str, Any] | None:
    """把管线入口的输入规整成状态字典。

    * 真正的状态（``load`` 产生的 dict、字段完整）原样返回，保证链式调用
      返回同一对象；
    * 普通映射/字段残缺的状态补全默认值后再用；
    * ``None`` 新建空状态；其他类型返回 ``None`` 由调用方降级。
    """
    if state is None:
        return _new_state()
    if not isinstance(state, Mapping):
        return None
    if isinstance(state, dict):
        complete = (
            all(
                isinstance(state.get(name), Mapping) and set(keys) <= set(state[name])
                for name, keys in _STATE_SHAPE.items()
            )
            and isinstance(state.get("pool"), list)
            and isinstance(state.get("history"), list)
            and isinstance(state.get("warnings"), list)
        )
        if complete:
            return state
    seeded = _new_state(state)
    for name in ("config", "stats", "buffer"):
        merged = dict(seeded[name])
        for key, value in _sub_map(state, name).items():
            default = merged.get(key)
            # 类型对不上就用默认值，避免 "capacity": "bad" 之类污染后续运算
            if key == "material_order":
                merged[key] = list(value) if isinstance(value, (list, tuple)) else merged[key]
            elif isinstance(default, bool):
                merged[key] = bool(value)
            elif isinstance(default, str):
                merged[key] = value if isinstance(value, str) else default
            elif isinstance(default, (int, float)):
                merged[key] = _num(value, float(default))
            else:
                merged[key] = value
        seeded[name] = merged
    seeded["pool"] = _sub_list(state, "pool")
    seeded["decals"] = _sub_list(state, "decals")
    seeded["instances"] = _sub_list(state, "instances")
    seeded["history"] = _sub_list(state, "history")
    seeded["warnings"] = _sub_list(state, "warnings")
    for name in ("frame", "presented"):
        if name in state:
            seeded[name] = int(_num(state.get(name), 0))
    for name in ("targets", "visible", "decal_world", "scene_depth", "camera", "projected", "last_draw"):
        if name in state:
            seeded[name] = state[name]
    return seeded


def project(state: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """对状态里的目标点做（默认切线空间）投射，结果写入 ``projected``。"""
    st = _coerce_state(state)
    if st is None:
        return project_tangent_space(state, *args, **kwargs)
    targets = kwargs.pop("targets", st.get("targets"))
    if targets is None:
        st["projected"] = []
        st["projected_ok"] = False
        log_warning(st, "project:no_targets")
        return st
    decals = _as_list(kwargs.pop("decals", st.get("decals")))
    kwargs.setdefault("mode", st["config"]["mode"])
    results: list[dict[str, Any]] = []
    for decal in decals:
        if not isinstance(decal, Mapping):
            continue
        res = project_tangent_space(
            targets,
            decal.get("center"),
            decal.get("normal"),
            decal.get("size", 1.0),
            decal.get("depth", 1.0),
            up_hint=decal.get("up", kwargs.get("up_hint")),
            project_dir=decal.get("project_dir"),
            **kwargs,
        )
        res["id"] = decal.get("id")
        res["material"] = decal.get("material", 0)
        results.append(res)
        log_warning(st, *res["warnings"])
    st["projected"] = results
    st["projected_ok"] = bool(results) and all(r["ok"] for r in results)
    return st


def draw(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """一帧绘制：材质排序 -> 缓冲扩容 -> 深度测试 -> 可见性入池。"""
    st = _coerce_state(state)
    if st is None:
        st = _new_state()
    cfg = st["config"]
    for name in _DATA_KEYS:
        if name in kwargs and kwargs[name] is not None:
            value = kwargs.pop(name)
            st[name] = list(value) if name in {"decals", "instances"} else value
    decal_world = kwargs.pop("decal_world", st.get("decal_world"))
    scene_depth = kwargs.pop("scene_depth", st.get("scene_depth"))
    camera = kwargs.pop("camera", st.get("camera"))
    instances = _as_list(st.get("instances"))
    decals = _as_list(st.get("decals"))

    # 6) 批次按材质排序
    batches = sort_batch_by_material(instances, material_order=_as_list(cfg.get("material_order")))

    # 4) 实例缓冲按需扩容
    grow = grow_instance_buffer(
        st["buffer"],
        len(instances),
        stride=st["buffer"].get("stride", 64),
        alignment=st["buffer"].get("alignment", 256),
        max_capacity=kwargs.pop("max_buffer_capacity", None),
    )
    st["buffer"].update(
        {
            "capacity": grow["capacity"],
            "used": grow["used_after"],
            "grow_events": grow["grow_events"],
            "bytes": grow["bytes_after"],
        }
    )
    st["stats"]["buffer_grow_events"] += 1 if grow["grown"] else 0
    st["stats"]["dropped_instances"] += int(grow["dropped"])

    # 2) 贴花参与深度测试
    depth = decal_depth_test(
        scene_depth,
        decal_world,
        camera,
        tolerance=_num(cfg.get("tolerance", 1e-4), 1e-4),
        bias=_num(cfg.get("bias", 0.0)),
        depth_encoding=str(cfg.get("depth_encoding", "float01")),
    )
    if depth["tested"]:
        st["stats"]["depth_passed"] += depth["passed"]
        st["stats"]["depth_rejected"] += depth["rejected"]
    else:
        st["stats"]["depth_skipped_frames"] += 1

    # 3) 池按可见性淘汰
    pool = pool_visibility_evict(
        st["pool"],
        decals,
        visible=st.get("visible"),
        capacity=cfg.get("capacity", 256),
        now=st["frame"],
        allow_visible_eviction=bool(cfg.get("allow_visible_eviction", True)),
    )
    st["pool"] = pool["pool"]
    st["stats"]["recycled"] += len(pool["recycled"])
    st["stats"]["evicted"] += len(pool["evicted"])
    st["stats"]["evicted_visible"] += len(pool["evicted_visible"])
    # 只有"池里还有不可见贴花却淘汰了可见贴花"才算真正违反可见性淘汰
    st["stats"]["evicted_avoidable"] += len(pool["evicted_avoidable"])
    st["stats"]["dropped_decals"] += len(pool["dropped"])
    st["stats"]["drawn"] += len(batches["instances"])
    st["stats"]["batches"] = batches["batch_count"]
    st["stats"]["materials"] = batches["materials"]
    log_warning(st, *batches["warnings"], *grow["warnings"], *depth["warnings"], *pool["warnings"])

    st["last_draw"] = {
        "frame": st["frame"],
        "batches": batches["batch_count"],
        "materials": batches["materials"],
        "state_changes": batches["state_changes"],
        "state_changes_before": batches["state_changes_before"],
        "instances": len(batches["instances"]),
        "buffer": {
            "capacity": grow["capacity"],
            "used": grow["used"],
            "grown": grow["grown"],
            "dropped": grow["dropped"],
        },
        "depth": {
            "tested": depth["tested"],
            "passed": depth["passed"],
            "rejected": depth["rejected"],
            "policy": depth["policy"],
        },
        "pool": pool["occupancy"],
        "recycled": len(pool["recycled"]),
        "evicted_visible": list(pool["evicted_visible"]),
        "dropped_decals": len(pool["dropped"]),
    }
    return st


def present(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """提交一帧：推进帧号并记录带摘要的历史，供 determinism 比对。"""
    st = _coerce_state(state)
    if st is None:
        st = _new_state()
    st["frame"] = int(_num(st.get("frame", 0), 0)) + 1
    st["presented"] = int(_num(st.get("presented", 0), 0)) + 1
    stats = st["stats"]
    stats["frames"] = st["presented"]
    entry = {
        "frame": st["frame"],
        "drawn": int(_num(stats.get("drawn", 0), 0)),
        "batches": int(_num(stats.get("batches", 0), 0)),
        "materials": list(stats.get("materials") or []),
        "depth_rejected": int(_num(stats.get("depth_rejected", 0), 0)),
        "pool_used": len(st.get("pool") or []),
        "buffer_capacity": int(_num(st["buffer"].get("capacity", 0), 0)),
        "buffer_used": int(_num(st["buffer"].get("used", 0), 0)),
    }
    entry["digest"] = _digest(entry)
    history = st.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        st["history"] = history
    history.append(entry)
    del history[:-64]
    st["last_present"] = entry
    return st


# ======================================================================================
# 自检：validate / metrics / audit / fallback / determinism
# ======================================================================================
def _iter_decals(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        pool = value.get("pool", value.get("decals"))
        return _as_list(pool)
    return _as_list(value)


def validate_decal(target: Any = None, **kwargs: Any) -> dict[str, Any]:
    """检查贴花数据不变量，返回逐项布尔结果而非抛异常。"""
    decals = _iter_decals(target)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []

    bad_normals = 0
    for decal in decals:
        if isinstance(decal, Mapping) and "normal" in decal:
            n = _vec3(decal["normal"])
            if n is None or abs(float(np.linalg.norm(n)) - 1.0) > 1e-3:
                bad_normals += 1
    checks["normals_unit"] = bad_normals == 0
    if bad_normals:
        errors.append(f"non_unit_normals:{bad_normals}")

    if isinstance(target, Mapping):
        bad_uv = 0
        for res in _sub_list(target, "projected"):
            if not isinstance(res, Mapping):
                continue
            u = _arr(res.get("u")).reshape(-1)
            v = _arr(res.get("v")).reshape(-1)
            if u.size != v.size:
                bad_uv += 1
                continue
            hit = _bools(res.get("hit"), u.size, False)
            if hit.any():
                if not np.all(np.isfinite(u[hit])) or not np.all(np.isfinite(v[hit])):
                    bad_uv += 1
                elif np.any((u[hit] < -1e-9) | (u[hit] > 1 + 1e-9)) or np.any(
                    (v[hit] < -1e-9) | (v[hit] > 1 + 1e-9)
                ):
                    bad_uv += 1
        checks["uv_in_range"] = bad_uv == 0
        if bad_uv:
            errors.append(f"uv_out_of_range:{bad_uv}")

        cfg = _sub_map(target, "config")
        cap = int(_num(cfg.get("capacity", 256), 256))
        pool = _sub_list(target, "pool")
        checks["pool_within_capacity"] = len(pool) <= cap
        if len(pool) > cap:
            errors.append(f"pool_overflow:{len(pool)}>{cap}")
        if cap > 256:
            warnings.append("pool_capacity_exceeds_budget")

        buf = _sub_map(target, "buffer")
        used = int(_num(buf.get("used", 0), 0))
        bcap = int(_num(buf.get("capacity", 0), 0))
        checks["buffer_within_capacity"] = used <= bcap
        if used > bcap:
            errors.append("buffer_over_capacity")

        ids = [str(_entry_id(e)) for e in pool]
        checks["pool_ids_unique"] = len(ids) == len(set(ids))
        if not checks["pool_ids_unique"]:
            errors.append("duplicate_pool_ids")

        stats = _sub_map(target, "stats")
        checks["no_visible_eviction"] = int(_num(stats.get("evicted_avoidable", 0), 0)) == 0
        if not checks["no_visible_eviction"]:
            warnings.append("visible_decal_evicted")
        checks["no_instance_drop"] = int(_num(stats.get("dropped_instances", 0), 0)) == 0
        if not checks["no_instance_drop"]:
            errors.append("instances_dropped")
        checks["depth_tested"] = int(_num(stats.get("depth_skipped_frames", 0), 0)) == 0
        if not checks["depth_tested"]:
            warnings.append("decals_without_depth_test")
    else:
        for name in (
            "uv_in_range",
            "pool_within_capacity",
            "buffer_within_capacity",
            "pool_ids_unique",
            "no_visible_eviction",
            "no_instance_drop",
            "depth_tested",
        ):
            checks[name] = True

    return {
        "ok": not errors,
        "checked": len(decals),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def metrics(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """贴花池 / 缓冲 / 批次 / 深度测试的量化指标。"""
    if not isinstance(state, Mapping):
        return {
            "ok": False,
            "pool": {},
            "buffer": {},
            "batches": {},
            "depth": {},
            "warnings": ["not_a_state"],
        }
    stats = _sub_map(state, "stats")
    cfg = _sub_map(state, "config")
    pool = _sub_list(state, "pool")
    cap = max(1, int(_num(cfg.get("capacity", 256), 256)))
    buf = _sub_map(state, "buffer")
    bcap = int(_num(buf.get("capacity", 0), 0))
    tested = int(_num(stats.get("depth_passed", 0), 0)) + int(_num(stats.get("depth_rejected", 0), 0))
    rejected = int(_num(stats.get("depth_rejected", 0), 0))
    return {
        "ok": True,
        "frame": int(_num(state.get("frame", 0), 0)),
        "presented": int(_num(state.get("presented", 0), 0)),
        "pool": {
            "used": len(pool),
            "capacity": cap,
            "fill": round(len(pool) / cap, 6),
            "visible": int(sum(1 for e in pool if _entry_visible(e, None))),
        },
        "buffer": {
            "capacity": bcap,
            "used": int(_num(buf.get("used", 0), 0)),
            "fill": round(int(_num(buf.get("used", 0), 0)) / bcap, 6) if bcap else 0.0,
            "grow_events": int(_num(buf.get("grow_events", 0), 0)),
            "bytes": bcap * int(_num(buf.get("stride", 64), 64)),
        },
        "batches": {
            "count": int(_num(stats.get("batches", 0), 0)),
            "state_changes": max(0, int(_num(stats.get("batches", 0), 0)) - 1),
            "materials": list(stats.get("materials") or []),
        },
        "depth": {
            "passed": int(_num(stats.get("depth_passed", 0), 0)),
            "rejected": rejected,
            "skipped_frames": int(_num(stats.get("depth_skipped_frames", 0), 0)),
            "reject_ratio": round(rejected / tested, 6) if tested else 0.0,
        },
        "churn": {
            "recycled": int(_num(stats.get("recycled", 0), 0)),
            "evicted": int(_num(stats.get("evicted", 0), 0)),
            "evicted_visible": int(_num(stats.get("evicted_visible", 0), 0)),
            "evicted_avoidable": int(_num(stats.get("evicted_avoidable", 0), 0)),
            "dropped_decals": int(_num(stats.get("dropped_decals", 0), 0)),
            "dropped_instances": int(_num(stats.get("dropped_instances", 0), 0)),
        },
        "warnings": list(state.get("warnings") or []),
    }


def audit(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """按六项修复逐条审计，返回可断言的布尔表与失败清单。"""
    if not isinstance(state, Mapping):
        return {"ok": False, "checks": {}, "failures": ["not_a_state"], "frame": 0, "score": 0.0}
    stats = _sub_map(state, "stats")
    cfg = _sub_map(state, "config")
    pool = _sub_list(state, "pool")
    buf = _sub_map(state, "buffer")
    last = _sub_map(state, "last_draw")

    tested = int(_num(stats.get("depth_passed", 0), 0)) + int(_num(stats.get("depth_rejected", 0), 0))
    checks: dict[str, bool] = {
        # 投射确实走切线空间
        "tangent_projection": str(cfg.get("mode", "tangent")) == "tangent",
        # 深度测试启用且真的跑过
        "depth_test_active": int(_num(stats.get("depth_skipped_frames", 0), 0)) == 0 and tested > 0,
        # 没有可见贴花被顶掉
        "visibility_evict": int(_num(stats.get("evicted_avoidable", 0), 0)) == 0,
        # 实例一个没丢（缓冲能长）
        "buffer_growth": int(_num(stats.get("dropped_instances", 0), 0)) == 0,
        # 池内法线保持单位长度（随物体变换后不应退化）
        "normal_follow": all(
            _vec3(e.get("normal")) is not None
            and abs(float(np.linalg.norm(_vec3(e["normal"]))) - 1.0) <= 1e-3
            for e in pool
            if isinstance(e, Mapping) and "normal" in e
        ),
        # 每个材质只出一个批次，且切换次数不高于排序前
        "material_sorted": (
            len(list(stats.get("materials") or [])) == len(set(map(str, stats.get("materials") or [])))
            and int(_num(last.get("state_changes", 0), 0)) <= int(_num(last.get("state_changes_before", 0), 0))
        ),
        "pool_within_capacity": len(pool) <= int(_num(cfg.get("capacity", 256), 256)),
        "buffer_within_capacity": int(_num(buf.get("used", 0), 0)) <= int(_num(buf.get("capacity", 0), 0)),
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "frame": int(_num(state.get("frame", 0), 0)),
        "score": round(sum(1 for ok in checks.values() if ok) / len(checks), 6),
    }


def fallback(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """资源不足时的降级：先淘汰不可见（老→新），仍不够才淘汰最老的可见，
    并把淘汰清单完整写进报告，绝不静默丢数据。"""
    if not isinstance(state, Mapping):
        return {"ok": False, "applied": False, "reason": "not_a_state", "plan": {}, "pool": [], "dropped": []}
    cfg = _sub_map(state, "config")
    cap = max(0, int(_num(kwargs.get("capacity", cfg.get("capacity", 256)), 256)))
    pool = _sub_list(state, "pool")
    plan: dict[str, Any] = {"capacity": cap, "mode": "visibility_first", "steps": []}
    if len(pool) <= cap:
        return {
            "ok": True,
            "applied": False,
            "reason": "within_capacity",
            "plan": plan,
            "pool": pool,
            "dropped": [],
            "complete": True,
        }

    # 按原始下标操作：id 重复时也不会误删其余条目
    invisible = sorted(
        (i for i, e in enumerate(pool) if not _entry_visible(e, None)),
        key=lambda i: _age(pool[i]),
    )
    visible = sorted(
        (i for i, e in enumerate(pool) if _entry_visible(e, None)),
        key=lambda i: _age(pool[i]),
    )
    need = len(pool) - cap
    drop = set((invisible + visible)[:need])
    dropped_entries = [e for i, e in enumerate(pool) if i in drop]
    survivors = [e for i, e in enumerate(pool) if i not in drop]
    dropped_visible = [e for i, e in enumerate(pool) if i in drop and _entry_visible(e, None)]

    plan["steps"].append({"action": "evict_invisible", "count": len(dropped_entries) - len(dropped_visible)})
    complete = not dropped_visible
    if dropped_visible:
        plan["steps"].append({"action": "evict_oldest_visible", "count": len(dropped_visible)})
        plan["warnings"] = ["visible_decals_dropped"]
        # 只有在可见贴花本身超过容量时才算"被迫"；否则说明淘汰顺序有问题
        if len(visible) > cap:
            plan["warnings"].append("visible_over_capacity_unavoidable")

    return {
        "ok": True,
        "applied": True,
        "reason": "pool_over_capacity",
        "plan": plan,
        "pool": survivors,
        "dropped": [_entry_id(e) for e in dropped_entries],
        "complete": complete,
    }


def determinism(state: Any = None, runs: int = 3, **kwargs: Any) -> dict[str, Any]:
    """同一输入重复执行整条绘制流程，比较稳定摘要，验证结果可复现。

    对状态的深拷贝重复调用一次 ``draw``（若已投射过则连同 ``project``），
    把批次顺序、缓冲布局、池内容、深度测试统计全部纳入摘要。只要有任何
    隐藏的随机性（迭代顺序、哈希顺序、未初始化状态），摘要就会分叉。
    """
    runs = max(2, min(int(_num(runs, 3)), 32))
    if not isinstance(state, Mapping):
        return {
            "ok": False,
            "deterministic": False,
            "runs": runs,
            "hashes": [],
            "hash": None,
            "unique": 0,
            "reason": "not_a_state",
        }
    hashes: list[str] = []
    for _ in range(runs):
        try:
            clone = copy.deepcopy(dict(state))
        except Exception:  # noqa: BLE001 - 无法深拷贝时退化为浅拷贝
            clone = dict(state)
        replay = _coerce_state(clone)
        if replay is None:
            replay = _new_state()
        replay["frame"] = 0
        replay["stats"] = dict(_new_state()["stats"])
        replay["pool"] = []
        # 缓冲从空开始重放，但保留调用方配置的 layout 参数
        src_buf = _sub_map(state, "buffer")
        replay["buffer"] = {
            "capacity": 0,
            "used": 0,
            "stride": max(1, int(_num(src_buf.get("stride", 64), 64))),
            "alignment": max(1, int(_num(src_buf.get("alignment", 256), 256))),
            "grow_events": 0,
        }
        if replay.get("targets") is not None:
            replay = project(replay)
        replay = draw(replay)
        snapshot = {
            "batches": [(str(b["material"]), b["count"]) for b in sort_batch_by_material(
                _as_list(replay.get("instances")),
                material_order=_as_list(_sub_map(replay, "config").get("material_order")),
            )["batches"]],
            "buffer": {
                "capacity": int(_num(_sub_map(replay, "buffer").get("capacity", 0), 0)),
                "used": int(_num(_sub_map(replay, "buffer").get("used", 0), 0)),
            },
            "pool": [str(_entry_id(e)) for e in _sub_list(replay, "pool")],
            "depth": _round(_sub_map(replay, "last_draw").get("depth")),
            "projected": [
                _round({k: r.get(k) for k in ("id", "hits", "ok")})
                for r in _sub_list(replay, "projected")
                if isinstance(r, Mapping)
            ],
        }
        hashes.append(_digest(snapshot))
    unique = sorted(set(hashes))
    return {
        "ok": True,
        "deterministic": len(unique) == 1,
        "runs": runs,
        "hashes": hashes,
        "hash": hashes[0],
        "unique": len(unique),
    }
