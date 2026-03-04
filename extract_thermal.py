#!/usr/bin/env python3
"""
Extract per-pixel temperatures from DJI thermal JPEG files.

Radiometric JPEG (R-JPEG) vs normal JPEG:
- Normal JPEG stores only display image pixels (8-bit visual content).
- DJI thermal JPEG embeds additional thermal payload + calibration/measurement
  metadata, enabling physical temperature reconstruction in Celsius.

Processing strategy in this script:
1) Planck path (for cameras/files that expose Planck tags via exiftool)
2) DJI DIRP SDK path (for M3T-style files that store calibration in DJI iirp
   binary blocks and do not expose Planck tags directly)

Outputs saved next to each source image:
- *_temp.npy    float32 Celsius matrix (HxW)
- *_temp.tiff   optional uint16 centi-Kelvin export
- *_preview.png visualization with colorbar

Usage:
  python extract_thermal.py
  python extract_thermal.py --root /path/to/Thermal3D/Data
  python extract_thermal.py --dirp-lib /path/to/libdirp.so
  python extract_thermal.py --strict-rjpeg-pattern
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    print("ERROR: numpy is required. Install with: pip install numpy", file=sys.stderr)
    raise


# Optional dependencies (script still runs without them)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import tifffile
except Exception:
    tifffile = None


PLANCK_KEYS = {
    "planckr1": "PlanckR1",
    "planckr2": "PlanckR2",
    "planckb": "PlanckB",
    "planckf": "PlanckF",
    "plancko": "PlanckO",
}


class ThermalExtractionError(RuntimeError):
    """Raised when thermal extraction fails for a candidate image."""


class ThermalSkipError(ThermalExtractionError):
    """Raised when an input file is intentionally skipped (non-thermal image)."""


class _DirpResolution(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


_DIRP_LIB_CACHE: Dict[str, ctypes.CDLL] = {}


def _canonical_key(key: str) -> str:
    key = key.split(":")[-1]  # strip group prefixes like EXIF:TagName
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _flatten_json(obj, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        if len(obj) == 1:
            out.update(_flatten_json(obj[0], prefix))
        else:
            out[prefix] = obj
    else:
        out[prefix] = obj
    return out


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        return _parse_float(value[0])
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _lookup_text(meta: Dict[str, object], candidate_keys: Iterable[str]) -> Optional[str]:
    cands = {_canonical_key(k) for k in candidate_keys}
    for key, value in meta.items():
        if _canonical_key(key) in cands:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, list) and value:
                return str(value[0])
    return None


def _run_exiftool(args: List[str], image_path: Path, expect_binary: bool = False):
    exiftool_bin = shutil.which("exiftool")
    if not exiftool_bin:
        raise ThermalExtractionError(
            "exiftool not found. Install it first (Linux: sudo apt install libimage-exiftool-perl)."
        )

    cmd = [exiftool_bin, *args, str(image_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise ThermalExtractionError(f"exiftool failed ({proc.returncode}): {stderr}")

    return proc.stdout if expect_binary else proc.stdout.decode("utf-8", errors="ignore")


def _read_exiftool_metadata(image_path: Path) -> Dict[str, object]:
    text = _run_exiftool(["-j", "-n", "-a", "-u", "-g1"], image_path, expect_binary=False)
    data = json.loads(text)
    if not data:
        raise ThermalExtractionError("No EXIF metadata returned by exiftool.")
    return _flatten_json(data[0])


def _lookup_float(meta: Dict[str, object], candidate_keys: Iterable[str]) -> Optional[float]:
    cands = {_canonical_key(k) for k in candidate_keys}
    for key, value in meta.items():
        if _canonical_key(key) in cands:
            parsed = _parse_float(value)
            if parsed is not None:
                return parsed
    return None


def _extract_raw_thermal_blob_exiftool(image_path: Path) -> bytes:
    """Extract thermal payload with exiftool binary output."""
    tag_candidates = ["RawThermalImage", "RawData", "ThermalData"]
    errors: List[str] = []
    for tag in tag_candidates:
        try:
            blob = _run_exiftool(["-b", f"-{tag}"], image_path, expect_binary=True)
            if blob:
                return blob
        except ThermalExtractionError as exc:
            errors.append(f"{tag}: {exc}")

    joined = " | ".join(errors) if errors else "no thermal tag returned data"
    raise ThermalExtractionError(f"Unable to extract RawThermalImage payload ({joined})")


def _is_tiff_blob(blob: bytes) -> bool:
    return len(blob) >= 4 and (blob[:4] == b"II*\x00" or blob[:4] == b"MM\x00*")


def _decode_payload_to_raw_uint16(blob: bytes) -> np.ndarray:
    if _is_tiff_blob(blob):
        if tifffile is None:
            raise ThermalExtractionError(
                "Thermal payload is TIFF but tifffile is not installed. Install with: pip install tifffile"
            )
        try:
            arr = tifffile.imread(io.BytesIO(blob))
        except Exception as exc:
            raise ThermalExtractionError(f"Failed to decode embedded TIFF payload: {exc}") from exc
        arr = np.asarray(arr)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        return arr.reshape(-1)

    if len(blob) % 2 != 0:
        raise ThermalExtractionError(f"Thermal payload has odd byte length ({len(blob)}).")

    return np.frombuffer(blob, dtype="<u2")


def _infer_dimensions(n_pixels: int, meta: Dict[str, object]) -> Tuple[int, int]:
    width = _lookup_float(meta, ["RawThermalImageWidth", "ThermalImageWidth", "ImageWidth", "RawImageWidth"])
    height = _lookup_float(meta, ["RawThermalImageHeight", "ThermalImageHeight", "ImageHeight", "RawImageHeight"])

    if width is not None and height is not None:
        w = int(round(width))
        h = int(round(height))
        if w > 0 and h > 0 and w * h == n_pixels:
            return h, w

    root = int(math.sqrt(n_pixels))
    candidates: List[Tuple[float, int, int]] = []
    for h in range(1, root + 1):
        if n_pixels % h != 0:
            continue
        w = n_pixels // h
        hh, ww = (h, w) if h <= w else (w, h)
        ratio = ww / hh
        if ratio < 1.0 or ratio > 3.0:
            continue
        score = min(abs(ratio - 4 / 3), abs(ratio - 5 / 4), abs(ratio - 16 / 9))
        score += 0.01 * ((ww % 2) + (hh % 2))
        candidates.append((score, hh, ww))

    if not candidates:
        raise ThermalExtractionError(f"Could not infer dimensions from payload size (pixels={n_pixels}).")

    _, h_best, w_best = sorted(candidates, key=lambda x: x[0])[0]
    return h_best, w_best


def _raw_from_temperature_c(temp_c: float, r1: float, r2: float, b: float, f: float, o: float) -> float:
    temp_k = temp_c + 273.15
    expo = math.exp(b / max(temp_k, 1e-6))
    denom = max(r2 * (expo - f), 1e-12)
    return (r1 / denom) - o


def raw_to_temperature_c(
    raw_uint16: np.ndarray,
    planck_r1: float,
    planck_r2: float,
    planck_b: float,
    planck_f: float,
    planck_o: float,
    emissivity: Optional[float] = None,
    reflected_app_temp_c: Optional[float] = None,
) -> np.ndarray:
    """
    Convert raw thermal values to Celsius with the Planck inversion:
      T(K) = B / ln( R1 / (R2 * (Raw + O)) + F )
    """
    raw = raw_uint16.astype(np.float64, copy=False)

    eps = float(emissivity) if emissivity is not None else 1.0
    eps = min(max(eps, 1e-4), 1.0)

    raw_corrected = raw
    if reflected_app_temp_c is not None and eps < 0.9999:
        raw_refl = _raw_from_temperature_c(
            reflected_app_temp_c,
            r1=planck_r1,
            r2=planck_r2,
            b=planck_b,
            f=planck_f,
            o=planck_o,
        )
        raw_corrected = raw - ((1.0 - eps) / eps) * raw_refl

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = np.maximum(planck_r2 * (raw_corrected + planck_o), 1e-12)
        log_arg = (planck_r1 / denom) + planck_f
        log_arg = np.maximum(log_arg, 1.000001)
        temp_k = planck_b / np.log(log_arg)
        temp_c = temp_k - 273.15

    temp_c = np.where(np.isfinite(temp_c), temp_c, np.nan)
    return temp_c.astype(np.float32)


def _extract_dji_image_source_from_file(image_path: Path) -> Optional[str]:
    try:
        data = image_path.read_bytes()
    except Exception:
        return None
    m = re.search(rb'drone-dji:ImageSource="([^"]+)"', data)
    if not m:
        return None
    try:
        return m.group(1).decode("utf-8", errors="ignore")
    except Exception:
        return None


def _looks_like_thermal_candidate(image_path: Path, image_source: Optional[str]) -> bool:
    name = image_path.name.upper()
    if name.endswith("_T.JPG") or name.endswith("_T.JPEG"):
        return True
    if name.endswith("_R.JPG") or name.endswith("_R.JPEG"):
        return True
    if image_source and image_source.lower() == "infraredcamera":
        return True
    return False


def _load_dirp_lib(dirp_lib_path: Optional[str]) -> ctypes.CDLL:
    candidates: List[Path] = []

    if dirp_lib_path:
        candidates.append(Path(dirp_lib_path).expanduser())

    env_lib = os.environ.get("DJI_DIRP_LIB")
    if env_lib:
        candidates.append(Path(env_lib).expanduser())

    # Common locations users may place DJI Thermal SDK binaries.
    candidates.extend(
        [
            Path.cwd() / "libdirp.so",
            Path.cwd() / "linux" / "libdirp.so",
            Path.cwd() / "linux" / "release_x64" / "libdirp.so",
            Path.cwd() / "dji_thermal_sdk" / "linux" / "libdirp.so",
            Path("/usr/local/lib/libdirp.so"),
        ]
    )

    resolved: Optional[Path] = None
    for c in candidates:
        if c.exists():
            resolved = c.resolve()
            break

    if resolved is None:
        raise ThermalExtractionError(
            "DJI libdirp.so not found. Provide --dirp-lib /path/to/libdirp.so or set DJI_DIRP_LIB."
        )

    key = str(resolved)
    if key in _DIRP_LIB_CACHE:
        return _DIRP_LIB_CACHE[key]

    lib_dir = str(resolved.parent)
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld}" if old_ld else lib_dir

    # Load dependent DJI helper libs first when present.
    for dep in ["libv_iirp.so", "libv_dirp.so", "libv_girp.so"]:
        dep_path = resolved.parent / dep
        if dep_path.exists():
            try:
                ctypes.CDLL(str(dep_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    try:
        lib = ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise ThermalExtractionError(f"Failed to load libdirp: {exc}") from exc

    # Configure the function signatures we use.
    lib.dirp_create_from_rjpeg.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
    lib.dirp_create_from_rjpeg.restype = ctypes.c_int

    lib.dirp_destroy.argtypes = [ctypes.c_void_p]
    lib.dirp_destroy.restype = ctypes.c_int

    lib.dirp_get_rjpeg_resolution.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DirpResolution)]
    lib.dirp_get_rjpeg_resolution.restype = ctypes.c_int

    if hasattr(lib, "dirp_measure_ex"):
        lib.dirp_measure_ex.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.dirp_measure_ex.restype = ctypes.c_int
    if hasattr(lib, "dirp_measure"):
        lib.dirp_measure.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.dirp_measure.restype = ctypes.c_int

    _DIRP_LIB_CACHE[key] = lib
    return lib


def _measure_temperature_with_dirp(image_path: Path, dirp_lib_path: Optional[str]) -> np.ndarray:
    """
    Measure Celsius with DJI DIRP API.

    This is the precise path for M3T-style thermal JPEGs that embed iirp data
    but do not expose Planck constants in EXIF tags.
    """
    lib = _load_dirp_lib(dirp_lib_path)

    rjpeg = image_path.read_bytes()
    if not rjpeg:
        raise ThermalExtractionError("Empty image file.")

    rjpeg_buf = (ctypes.c_uint8 * len(rjpeg)).from_buffer_copy(rjpeg)
    handle = ctypes.c_void_p()
    ret = lib.dirp_create_from_rjpeg(ctypes.cast(rjpeg_buf, ctypes.c_void_p), ctypes.c_int32(len(rjpeg)), ctypes.byref(handle))
    if ret != 0:
        raise ThermalExtractionError(f"dirp_create_from_rjpeg failed with code {ret} for {image_path.name}")

    try:
        res = _DirpResolution()
        ret = lib.dirp_get_rjpeg_resolution(handle, ctypes.byref(res))
        if ret != 0:
            raise ThermalExtractionError(f"dirp_get_rjpeg_resolution failed with code {ret}")
        if res.width <= 0 or res.height <= 0:
            raise ThermalExtractionError(f"Invalid DIRP resolution: width={res.width}, height={res.height}")

        h = int(res.height)
        w = int(res.width)

        if hasattr(lib, "dirp_measure_ex"):
            temp = np.empty((h, w), dtype=np.float32)
            ret = lib.dirp_measure_ex(
                handle,
                temp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int32(temp.nbytes),
            )
            if ret != 0:
                raise ThermalExtractionError(f"dirp_measure_ex failed with code {ret}")
            return temp

        if hasattr(lib, "dirp_measure"):
            temp_i16 = np.empty((h, w), dtype=np.int16)
            ret = lib.dirp_measure(
                handle,
                temp_i16.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int32(temp_i16.nbytes),
            )
            if ret != 0:
                raise ThermalExtractionError(f"dirp_measure failed with code {ret}")
            return (temp_i16.astype(np.float32) / 10.0)

        raise ThermalExtractionError("Loaded DIRP library does not expose dirp_measure_ex/dirp_measure.")
    finally:
        try:
            lib.dirp_destroy(handle)
        except Exception:
            pass


def _write_preview_png(temp_c: np.ndarray, out_png: Path) -> None:
    if plt is None:
        print(f"[WARN] matplotlib unavailable; skipping preview: {out_png}")
        return

    finite_vals = temp_c[np.isfinite(temp_c)]
    if finite_vals.size == 0:
        print(f"[WARN] no finite temperatures; skipping preview: {out_png}")
        return

    vmin = float(np.percentile(finite_vals, 2))
    vmax = float(np.percentile(finite_vals, 98))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    im = ax.imshow(temp_c, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title("Thermal Temperature (°C)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Temperature (°C)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _write_optional_tiff(temp_c: np.ndarray, out_tiff: Path) -> None:
    if tifffile is None:
        print(f"[WARN] tifffile unavailable; skipping TIFF: {out_tiff}")
        return
    temp_ck = np.clip(np.round((temp_c + 273.15) * 100.0), 0, 65535).astype(np.uint16)
    tifffile.imwrite(out_tiff, temp_ck)


def _collect_planck_and_env(meta: Dict[str, object]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key_canon, key_out in PLANCK_KEYS.items():
        out[key_out] = _lookup_float(meta, [key_out, key_canon])

    out["Emissivity"] = _lookup_float(meta, ["Emissivity", "ObjectEmissivity"])
    out["ReflectedApparentTemperature"] = _lookup_float(
        meta,
        [
            "ReflectedApparentTemperature",
            "ReflectedTemperature",
            "ReflectedTemp",
            "ReflectedAppTemp",
            "ReflectedTemperature2",
        ],
    )
    return out


def _has_full_planck(calib: Dict[str, Optional[float]]) -> bool:
    required = ["PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO"]
    return all(calib.get(k) is not None for k in required)


def _save_outputs(image_path: Path, temp_c: np.ndarray, write_tiff: bool) -> None:
    stem = image_path.with_suffix("")
    npy_path = Path(f"{stem}_temp.npy")
    tiff_path = Path(f"{stem}_temp.tiff")
    png_path = Path(f"{stem}_preview.png")

    np.save(npy_path, temp_c.astype(np.float32, copy=False))
    if write_tiff:
        _write_optional_tiff(temp_c, tiff_path)
    _write_preview_png(temp_c, png_path)


def process_rjpeg(
    image_path: Path,
    save_outputs: bool = True,
    write_tiff: bool = True,
    dirp_lib_path: Optional[str] = None,
    enable_dirp: bool = True,
) -> np.ndarray:
    """Process one DJI thermal image and return float32 Celsius matrix (HxW)."""
    # Try metadata via exiftool, but don't fail immediately if unavailable.
    meta: Dict[str, object] = {}
    meta_error: Optional[Exception] = None
    try:
        meta = _read_exiftool_metadata(image_path)
    except Exception as exc:
        meta_error = exc

    image_source = _lookup_text(meta, ["ImageSource", "drone-dji:ImageSource"]) or _extract_dji_image_source_from_file(
        image_path
    )

    if image_source and image_source.lower() in {"widecamera", "zoomcamera", "visiblecamera"}:
        raise ThermalSkipError(f"ImageSource={image_source} (visible stream, not thermal)")

    if image_path.name.startswith("._"):
        raise ThermalSkipError("Apple metadata sidecar file")

    calib = _collect_planck_and_env(meta) if meta else {}

    # Path A: classic Planck conversion (works for cameras that expose Planck tags)
    planck_err: Optional[Exception] = None
    if calib and _has_full_planck(calib):
        try:
            blob = _extract_raw_thermal_blob_exiftool(image_path)
            raw_flat = _decode_payload_to_raw_uint16(blob)
            h, w = _infer_dimensions(raw_flat.size, meta)
            raw = raw_flat.reshape(h, w)
            temp_c = raw_to_temperature_c(
                raw,
                planck_r1=float(calib["PlanckR1"]),
                planck_r2=float(calib["PlanckR2"]),
                planck_b=float(calib["PlanckB"]),
                planck_f=float(calib["PlanckF"]),
                planck_o=float(calib["PlanckO"]),
                emissivity=calib.get("Emissivity"),
                reflected_app_temp_c=calib.get("ReflectedApparentTemperature"),
            )
            method = "planck"
        except Exception as exc:
            planck_err = exc
        else:
            if save_outputs:
                _save_outputs(image_path, temp_c, write_tiff=write_tiff)
            finite = temp_c[np.isfinite(temp_c)]
            tmin = float(np.nanmin(finite)) if finite.size else float("nan")
            tmax = float(np.nanmax(finite)) if finite.size else float("nan")
            tmean = float(np.nanmean(finite)) if finite.size else float("nan")
            print(
                f"[OK] {image_path}\n"
                f"     method={method} shape={temp_c.shape}  min={tmin:.3f}C  max={tmax:.3f}C  mean={tmean:.3f}C"
            )
            return temp_c

    # Path B: DJI DIRP SDK (precise M3T path for _T/JPEG with iirp blocks)
    if enable_dirp and _looks_like_thermal_candidate(image_path, image_source):
        temp_c = _measure_temperature_with_dirp(image_path, dirp_lib_path=dirp_lib_path)
        temp_c = np.asarray(temp_c, dtype=np.float32)

        if temp_c.ndim != 2:
            raise ThermalExtractionError(f"DIRP returned invalid shape: {temp_c.shape}")

        if save_outputs:
            _save_outputs(image_path, temp_c, write_tiff=write_tiff)

        finite = temp_c[np.isfinite(temp_c)]
        tmin = float(np.nanmin(finite)) if finite.size else float("nan")
        tmax = float(np.nanmax(finite)) if finite.size else float("nan")
        tmean = float(np.nanmean(finite)) if finite.size else float("nan")
        print(
            f"[OK] {image_path}\n"
            f"     method=dirp_measure_ex shape={temp_c.shape}  min={tmin:.3f}C  max={tmax:.3f}C  mean={tmean:.3f}C"
        )
        return temp_c

    # If we got here, neither path succeeded.
    if not _looks_like_thermal_candidate(image_path, image_source):
        raise ThermalSkipError("not recognized as thermal candidate (_T/_R suffix or InfraredCamera source)")

    messages: List[str] = []
    if planck_err:
        messages.append(f"Planck path failed: {planck_err}")
    if meta_error:
        messages.append(f"Metadata read issue: {meta_error}")
    if not enable_dirp:
        messages.append("DIRP path disabled (--disable-dirp)")

    if not messages:
        missing = [k for k in ["PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO"] if calib.get(k) is None]
        if missing:
            messages.append(f"Missing Planck metadata: {', '.join(missing)}")

    raise ThermalExtractionError("; ".join(messages))


def load_thermal_tensor(path: str):
    """
    Load thermal data as a PyTorch tensor (float32, HxW).

    Supported inputs:
    - *_temp.npy
    - DJI thermal JPEG (extracts on demand if needed)
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for load_thermal_tensor(). Install with: pip install torch") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        inferred_npy = Path(f"{p.with_suffix('')}_temp.npy")
        if inferred_npy.exists():
            arr = np.load(inferred_npy)
        else:
            arr = process_rjpeg(p, save_outputs=False)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW thermal array, got shape={arr.shape}")

    return torch.from_numpy(arr)


def _find_input_root(user_root: Optional[str]) -> Path:
    if user_root:
        return Path(user_root).expanduser().resolve()

    candidates = [
        Path("/Volumes/SHD/NIPS2026/Thermal3D/Data"),
        Path("/Volumes/SDH/NIPS2026/Thermal3D/Data"),
        Path.cwd() / "Thermal3D" / "Data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return Path.cwd().resolve()


def _iter_target_images(root: Path, strict_rjpeg_pattern: bool) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        name = p.name.upper()
        if strict_rjpeg_pattern:
            if name.endswith("_R.JPG") or name.endswith("_R.JPEG"):
                yield p
            continue

        yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DJI thermal temperatures to numpy/tiff/preview.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory to recursively scan (default: auto-detect Thermal3D/Data)",
    )
    parser.add_argument(
        "--strict-rjpeg-pattern",
        action="store_true",
        help="Only process files matching *_R.JPG / *_R.JPEG",
    )
    parser.add_argument("--no-tiff", action="store_true", help="Skip optional TIFF output")
    parser.add_argument(
        "--dirp-lib",
        type=str,
        default=None,
        help="Path to DJI Thermal SDK libdirp.so (used when Planck tags are unavailable)",
    )
    parser.add_argument(
        "--disable-dirp",
        action="store_true",
        help="Disable DJI DIRP fallback (Planck-only mode)",
    )
    args = parser.parse_args()

    root = _find_input_root(args.root)
    if not root.exists():
        print(f"ERROR: Input root does not exist: {root}", file=sys.stderr)
        return 2

    files = sorted(_iter_target_images(root, strict_rjpeg_pattern=args.strict_rjpeg_pattern))
    if not files:
        patt = "*_R.JPG" if args.strict_rjpeg_pattern else "all JPGs"
        print(f"No matching images found under {root} (pattern={patt}).")
        return 0

    print(f"Scanning {root}")
    print(f"Found {len(files)} candidate images")

    ok = 0
    fail = 0
    skip = 0

    for image_path in files:
        try:
            process_rjpeg(
                image_path,
                save_outputs=True,
                write_tiff=not args.no_tiff,
                dirp_lib_path=args.dirp_lib,
                enable_dirp=not args.disable_dirp,
            )
            ok += 1
        except ThermalSkipError as exc:
            skip += 1
            print(f"[SKIP] {image_path}: {exc}")
        except ThermalExtractionError as exc:
            fail += 1
            print(f"[ERR] {image_path}: {exc}")
        except Exception as exc:
            fail += 1
            print(f"[ERR] {image_path}: Unexpected error: {exc}")

    print(f"Done. success={ok}, failed={fail}, skipped={skip}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
