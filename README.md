# Thermal3D DJI Mavic 3T Extraction Pipeline

## Project Purpose
This project extracts per-pixel temperature maps from DJI thermal JPEG files for 3D thermal reconstruction workflows.

The main script is:
- `/Volumes/SHD/NIPS2026/extract_thermal.py`

The script supports two extraction paths:
1. Planck metadata path (when `PlanckR1/R2/B/F/O` are present in metadata)
2. DJI DIRP SDK path (required for M3T-style `_T.JPG` files where Planck tags are not exposed)

## Project Composition
Current top-level layout:

```text
/Volumes/SHD/NIPS2026
├── extract_thermal.py
├── Thermal3D/
│   ├── Data/
│   │   ├── DJI_*_T.JPG         # thermal frames
│   │   ├── DJI_*_V.JPG         # visible frames
│   │   └── ._*                 # macOS sidecar files (ignored)
│   └── papers/
└── __pycache__/
```

## Code Structure (extract_thermal.py)
Main workflow:
- `main()`: CLI entry point, recursive scan, per-file processing, summary counts
- `process_rjpeg(...)`: chooses Planck path or DIRP path, writes outputs, prints stats

Metadata and parsing helpers:
- `_read_exiftool_metadata(...)`, `_lookup_float(...)`, `_collect_planck_and_env(...)`
- `_extract_dji_image_source_from_file(...)`, `_looks_like_thermal_candidate(...)`

Planck conversion path:
- `_extract_raw_thermal_blob_exiftool(...)`
- `_decode_payload_to_raw_uint16(...)`
- `_infer_dimensions(...)`
- `raw_to_temperature_c(...)`

DIRP (DJI SDK) path:
- `_load_dirp_lib(...)`
- `_measure_temperature_with_dirp(...)`

Output helpers:
- `_save_outputs(...)`
- `_write_optional_tiff(...)`
- `_write_preview_png(...)`
- `load_thermal_tensor(path)` for PyTorch (`float32`, `H x W`)

## Output Files
For each processed thermal JPEG:
- `*_temp.npy` (float32, Celsius)
- `*_temp.tiff` (optional uint16 centi-Kelvin)
- `*_preview.png` (colormap + colorbar)

The script also prints:
- shape
- min temperature
- max temperature
- mean temperature

## Important Runtime Requirements
Python packages:
- `numpy`
- `matplotlib` (preview PNG)
- `tifffile` (TIFF output)
- `torch` (for `load_thermal_tensor`)

External tools/libraries:
- `exiftool`
- DJI Thermal SDK Linux libraries for precise M3T thermal conversion:
  - `libdirp.so`
  - usually companion libs in the same folder (`libv_iirp.so`, `libv_dirp.so`, `libv_girp.so`)

## Why macOS Failed for Precise Extraction
The current M3T `_T.JPG` files contain thermal payload but do not expose Planck tags directly.  
Precise temperature therefore needs DJI DIRP runtime. This is typically available for Linux/Windows binaries, not native macOS ARM.

## Narval (Compute Canada) Deployment Notes
Target environment should be Linux x86_64.

### Recommended Run Steps
1. Create/activate env on Narval
2. Install Python deps (`numpy`, `matplotlib`, `tifffile`, `torch`)
3. Ensure `exiftool` is available
4. Place DJI Thermal SDK Linux libs in a project folder, for example:
   - `/path/to/dji_sdk/linux/release_x64/libdirp.so`
5. Export runtime variables:
   - `export DJI_DIRP_LIB=/path/to/libdirp.so`
   - `export LD_LIBRARY_PATH=$(dirname "$DJI_DIRP_LIB"):$LD_LIBRARY_PATH`
6. Run:
   - `python /path/to/extract_thermal.py --root /path/to/Thermal3D/Data`

### Example Slurm Job Script (Narval)
```bash
#!/bin/bash
#SBATCH --job-name=thermal_extract
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=thermal_extract_%j.out

set -euo pipefail

module load python/3.11

source ~/envs/thermal3d/bin/activate

export DJI_DIRP_LIB=/home/$USER/dji_thermal_sdk/linux/release_x64/libdirp.so
export LD_LIBRARY_PATH=$(dirname "$DJI_DIRP_LIB"):$LD_LIBRARY_PATH

python /home/$USER/NIPS2026/extract_thermal.py \
  --root /home/$USER/NIPS2026/Thermal3D/Data
```

## Future Agent Notes (Please Address in Next Modifications)
1. Add a `requirements.txt` and optional `environment.yml` for reproducible setup.
2. Add a startup check command (`--check-env`) that validates:
   - `exiftool` present
   - `libdirp.so` loadable
   - required Python packages present
3. Add a `--thermal-only` mode to only include `_T.JPG` and skip `_V.JPG` earlier.
4. Add CSV/JSON summary output (per image stats + method used: `planck` or `dirp_measure_ex`).
5. Add tests with 1-2 small sample files to validate:
   - file classification
   - output shape
   - save paths
6. Add robust handling for cluster environments where `LD_LIBRARY_PATH` is sanitized by scheduler modules.
7. Add documentation for data transfer and absolute path remapping from local Mac paths to Narval paths.

## Current Status Snapshot
- `_T.JPG` files are detected as thermal candidates.
- `_V.JPG` files are correctly skipped as visible stream.
- End-to-end precise extraction will work once DJI Linux DIRP libs are available on target runtime.
