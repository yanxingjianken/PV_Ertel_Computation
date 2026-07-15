#!/usr/bin/env python
"""Download ERA5 hourly pressure-level data for Ertel PV validation.

Downloads u, v, T, q, and native PV at all 37 pressure levels for
2025-01-08 00Z using the CDS API.

Also downloads surface pressure from ERA5 single levels for reference.

Usage
-----
    micromamba run -n fourcastnetv2 python download_era5.py
"""

import cdsapi
from pathlib import Path

# ---- config ----
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "era5_2025-01-08_00z_pl.nc"

# All 37 ERA5 pressure levels [hPa]
PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50",
    "70", "100", "125", "150", "175", "200", "225",
    "250", "300", "350", "400", "450", "500", "550",
    "600", "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000",
]

# ---- download ----
c = cdsapi.Client()

print("Downloading ERA5 pressure-level data for 2025-01-08 00Z ...")
print(f"  Output: {OUT_FILE}")
print(f"  Levels: {len(PRESSURE_LEVELS)} ({PRESSURE_LEVELS[0]}–{PRESSURE_LEVELS[-1]} hPa)")

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "potential_vorticity",
            "specific_humidity",
        ],
        "pressure_level": PRESSURE_LEVELS,
        "year": "2025",
        "month": "01",
        "day": "08",
        "time": "00:00",
    },
    str(OUT_FILE),
)

print(f"✓ Download complete: {OUT_FILE}")
print(f"  Size: {OUT_FILE.stat().st_size / 1e6:.1f} MB")

# ---- also download surface pressure (single levels) ----
SP_FILE = OUT_DIR / "era5_2025-01-08_00z_sp.nc"
print(f"\nDownloading ERA5 surface pressure ...")
print(f"  Output: {SP_FILE}")

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": ["surface_pressure"],
        "year": "2025",
        "month": "01",
        "day": "08",
        "time": "00:00",
    },
    str(SP_FILE),
)
print(f"✓ Download complete: {SP_FILE}")
print(f"  Size: {SP_FILE.stat().st_size / 1e6:.1f} MB")
