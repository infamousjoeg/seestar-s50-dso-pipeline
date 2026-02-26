#!/usr/bin/env python3
"""
Seestar S50 Astrophotography Pipeline

Automated stacking and post-processing pipeline for ZWO Seestar S50 smart telescope.
Takes raw .fits sub-frames and produces a processed, plate-solved image plus AstroBin
metadata files. One command, zero manual intervention.

Usage:
    python astro_pipeline.py <subs_directory> [options]
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from astropy.io import fits
from PIL import Image

log = logging.getLogger("astro_pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        log.warning("Config file not found: %s, using defaults", config_path)
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_profiles(profiles_path: Path) -> dict[str, Any]:
    """Load processing profiles from YAML file."""
    if not profiles_path.exists():
        log.error("Profiles file not found: %s", profiles_path)
        sys.exit(1)
    with open(profiles_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_tool_path(config: dict[str, Any], tool_name: str) -> str:
    """Get tool path from config, with sensible defaults."""
    defaults = {
        "siril_cli": "siril-cli",
        "graxpert": "graxpert",
        "cosmic_clarity": "SetiAstroCosmicClarity.exe",
    }
    tools = config.get("tools", {})
    return tools.get(tool_name, defaults.get(tool_name, tool_name))


# ---------------------------------------------------------------------------
# External process helpers
# ---------------------------------------------------------------------------

def run_command(
    cmd: list[str],
    description: str,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run an external command with logging."""
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        log.error("%s failed (exit code %d)", description, result.returncode)
        if result.stdout:
            log.error("stdout:\n%s", result.stdout[-2000:])
        if result.stderr:
            log.error("stderr:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    if result.stdout:
        log.debug("stdout:\n%s", result.stdout[-1000:])
    return result


def run_siril_script(siril_path: str, script_path: Path, working_dir: Path) -> None:
    """Run a Siril .ssf script via siril-cli."""
    # Siril's -d flag needs the directory path without a trailing backslash
    work_dir_str = str(working_dir).rstrip("\\")
    run_command(
        [siril_path, "-s", str(script_path), "-d", work_dir_str],
        f"Siril script {script_path.name}",
    )


def write_siril_script(script_path: Path, commands: list[str]) -> None:
    """Write a temporary Siril script file."""
    with open(script_path, "w", newline="\r\n") as f:
        for cmd in commands:
            f.write(cmd + "\n")


# ---------------------------------------------------------------------------
# SIMBAD query with local cache
# ---------------------------------------------------------------------------

def load_simbad_cache(cache_path: Path) -> dict[str, Any]:
    """Load SIMBAD query cache from disk."""
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_simbad_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    """Save SIMBAD query cache to disk."""
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def query_simbad(object_name: str, cache_path: Path) -> dict[str, Any]:
    """Query SIMBAD for object classification, with caching."""
    cache = load_simbad_cache(cache_path)

    if object_name in cache:
        log.info("Using cached SIMBAD result for '%s'", object_name)
        return cache[object_name]

    log.info("Querying SIMBAD for '%s'...", object_name)
    try:
        from astroquery.simbad import Simbad as SimbadQuery

        simbad = SimbadQuery()
        simbad.add_votable_fields("otype")
        result_table = simbad.query_object(object_name)

        if result_table is None:
            log.warning("SIMBAD returned no results for '%s'", object_name)
            return {"otype": "Unknown", "main_id": object_name}

        entry = {
            "main_id": str(result_table["MAIN_ID"][0]),
            "otype": str(result_table["OTYPE"][0]),
        }

        log.info("  SIMBAD MAIN_ID: %s", entry["main_id"])
        log.info("  SIMBAD OTYPE: %s", entry["otype"])

        cache[object_name] = entry
        save_simbad_cache(cache_path, cache)
        return entry

    except Exception as e:
        log.warning("SIMBAD query failed: %s", e)
        return {"otype": "Unknown", "main_id": object_name}


def select_profile(
    otype: str,
    profiles: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Select processing profile based on SIMBAD object type."""
    for profile_name, profile in profiles.items():
        if otype in profile.get("otypes", []):
            return profile_name, profile
    return "generic_dso", profiles.get("generic_dso", {})


def parse_target_from_dirname(dir_path: Path) -> Optional[str]:
    """Extract target name from directory name (e.g., 'M42_subs' -> 'M42')."""
    dirname = dir_path.name
    for suffix in ["_subs", "_lights", "_sub", "_light", "_frames"]:
        if dirname.lower().endswith(suffix):
            dirname = dirname[: -len(suffix)]
            break
    return dirname if dirname else None


# ---------------------------------------------------------------------------
# FITS <-> TIFF format conversion (for Cosmic Clarity)
# ---------------------------------------------------------------------------

def fits_to_tiff(fits_path: Path, tiff_path: Path) -> dict:
    """Convert FITS to 16-bit TIFF for Cosmic Clarity.  Returns original FITS header as dict."""
    log.info("Converting FITS -> TIFF: %s", fits_path.name)
    with fits.open(str(fits_path)) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = dict(hdul[0].header)

    # Normalize to 0-65535
    if data.max() > 0:
        if data.max() <= 1.0:
            data = data * 65535.0
        elif data.max() > 65535:
            data = (data / data.max()) * 65535.0

    data = np.clip(data, 0, 65535).astype(np.uint16)

    # FITS colour layout is (C, H, W); PIL expects (H, W, C)
    if data.ndim == 3 and data.shape[0] in (3, 4):
        data = np.transpose(data, (1, 2, 0))

    img = Image.fromarray(data)
    img.save(str(tiff_path), compression=None)
    return header


def tiff_to_fits(tiff_path: Path, fits_path: Path, original_header: dict) -> None:
    """Convert TIFF back to FITS, restoring the original header metadata."""
    log.info("Converting TIFF -> FITS: %s", tiff_path.name)
    img = Image.open(str(tiff_path))
    data = np.array(img, dtype=np.float32)

    # PIL gives (H, W, C); FITS needs (C, H, W)
    if data.ndim == 3 and data.shape[2] in (3, 4):
        data = np.transpose(data, (2, 0, 1))

    # Normalize to [0, 1] (Siril convention)
    if data.max() > 1.0:
        data = data / 65535.0

    header = fits.Header()
    for key, val in original_header.items():
        if key and key not in ("COMMENT", "HISTORY", "") and len(key) <= 8:
            try:
                header[key] = val
            except (ValueError, KeyError):
                pass

    fits.writeto(str(fits_path), data, header, overwrite=True)


# ---------------------------------------------------------------------------
# Phase 0 – Object Identification & Profile Selection
# ---------------------------------------------------------------------------

def phase_0_identify(
    subs_dir: Path,
    config: dict[str, Any],
    profiles: dict[str, Any],
    target_override: Optional[str] = None,
    profile_override: Optional[str] = None,
) -> dict[str, Any]:
    """Read FITS headers, query SIMBAD, select processing profile."""
    log.info("=" * 60)
    log.info("Phase 0: Object Identification & Profile Selection")
    log.info("=" * 60)

    # Locate sub-frames
    lights_dir = subs_dir / "lights"
    search_dir = lights_dir if lights_dir.exists() else subs_dir

    fits_files = sorted(
        list(search_dir.glob("*.fits")) + list(search_dir.glob("*.fit"))
    )
    if not fits_files:
        log.error("No .fits/.fit files found in %s", search_dir)
        sys.exit(1)

    log.info("Found %d sub-frames in %s", len(fits_files), search_dir)

    # Read first sub header
    first_sub = fits_files[0]
    log.info("Reading FITS header from: %s", first_sub)
    header = fits.getheader(str(first_sub))

    seestar = config.get("seestar", {})
    object_name = header.get("OBJECT", "").strip()
    ra = header.get("RA")
    dec = header.get("DEC")
    bayer = header.get("BAYERPAT", seestar.get("default_bayer", "GBRG"))
    exptime = header.get("EXPTIME", seestar.get("typical_exposure", 10.0))
    gain = header.get("GAIN", seestar.get("typical_gain", 80))

    log.info("  OBJECT: %s", object_name or "(not set)")
    log.info("  RA: %s | DEC: %s", ra, dec)
    log.info("  BAYERPAT: %s | EXPTIME: %ss | GAIN: %s", bayer, exptime, gain)

    # Target identification chain
    if target_override:
        object_name = target_override
        log.info("  Using target override: %s", object_name)
    elif not object_name:
        object_name = parse_target_from_dirname(subs_dir)
        if object_name:
            log.info("  Detected target from folder name: %s", object_name)
        else:
            log.warning("  Could not determine target name")
            object_name = "Unknown"

    # SIMBAD classification
    cache_path = Path(config.get("simbad", {}).get("cache_file", "simbad_cache.json"))
    if not cache_path.is_absolute():
        cache_path = Path(__file__).parent / cache_path

    simbad_result = query_simbad(object_name, cache_path)
    otype = simbad_result.get("otype", "Unknown")
    main_id = simbad_result.get("main_id", object_name)

    # Profile selection
    if profile_override:
        if profile_override not in profiles:
            log.error(
                "Unknown profile: %s. Available: %s",
                profile_override,
                list(profiles.keys()),
            )
            sys.exit(1)
        profile_name = profile_override
        profile = profiles[profile_name]
        log.info("  Using profile override: %s", profile_name)
    else:
        profile_name, profile = select_profile(otype, profiles)

    log.info("Selected profile: %s", profile_name)
    log.info(
        "  Denoise: %s | Deconv: %s @ %s | Star split: %s",
        profile.get("denoise_strength"),
        profile.get("deconv_mode"),
        profile.get("deconv_amount"),
        profile.get("star_split"),
    )

    return {
        "object_name": object_name,
        "main_id": main_id,
        "otype": otype,
        "ra": ra,
        "dec": dec,
        "bayer": bayer,
        "exptime": exptime,
        "gain": gain,
        "profile_name": profile_name,
        "profile": profile,
        "fits_files": fits_files,
        "search_dir": search_dir,
    }


# ---------------------------------------------------------------------------
# Phase 1 – Stacking
# ---------------------------------------------------------------------------

def phase_1_stack(
    subs_dir: Path,
    results_dir: Path,
    info: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Stack sub-frames using Siril (debayer, register, sigma-clip)."""
    log.info("=" * 60)
    log.info("Phase 1: Stacking %d sub-frames", len(info["fits_files"]))
    log.info("=" * 60)

    output_file = results_dir / "result.fit"
    if output_file.exists():
        log.info("Stacked result already exists, skipping")
        return output_file

    # Set up Siril directory structure
    lights_dir = results_dir / "lights"
    process_dir = results_dir / "process"
    lights_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    # Copy subs into lights directory
    for f in info["fits_files"]:
        dest = lights_dir / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))

    log.info("Copied %d subs to %s", len(info["fits_files"]), lights_dir)

    script_path = Path(__file__).parent / "seestar_stack.ssf"
    siril_path = get_tool_path(config, "siril_cli")
    run_siril_script(siril_path, script_path, results_dir)

    if not output_file.exists():
        raise RuntimeError("Stacking failed: result.fit not created")

    log.info("Phase 1 complete: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phase 2 – Calibration
# ---------------------------------------------------------------------------

def phase_2_calibrate(
    results_dir: Path,
    current_file: Path,
    info: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Colour calibration: autocrop, remove green noise, PCC."""
    log.info("=" * 60)
    log.info("Phase 2: Color Calibration")
    log.info("=" * 60)

    output_file = results_dir / "result_calibrated.fit"
    if output_file.exists():
        log.info("Calibrated result already exists, skipping")
        return output_file

    # Ensure RA/DEC and optics info are in the FITS header so PCC can plate-solve
    seestar = config.get("seestar", {})
    with fits.open(str(current_file), mode="update") as hdul:
        hdr = hdul[0].header
        if info.get("ra") is not None:
            hdr["RA"] = info["ra"]
        if info.get("dec") is not None:
            hdr["DEC"] = info["dec"]
        if info.get("object_name"):
            hdr["OBJECT"] = info["object_name"]
        hdr["FOCALLEN"] = seestar.get("focal_length", 250)
        hdr["XPIXSZ"] = seestar.get("pixel_size", 2.9)
        hdr["YPIXSZ"] = seestar.get("pixel_size", 2.9)
        log.info("Wrote RA/DEC/optics metadata into FITS header for PCC")

    script_path = Path(__file__).parent / "seestar_calibrate.ssf"
    siril_path = get_tool_path(config, "siril_cli")
    run_siril_script(siril_path, script_path, results_dir)

    if not output_file.exists():
        raise RuntimeError("Calibration failed: result_calibrated.fit not created")

    log.info("Phase 2 complete: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phase 3 – Background Extraction
# ---------------------------------------------------------------------------

def phase_3_background(
    results_dir: Path,
    current_file: Path,
    config: dict[str, Any],
) -> Path:
    """AI background extraction using GraXpert."""
    log.info("=" * 60)
    log.info("Phase 3: Background Extraction")
    log.info("=" * 60)

    output_file = results_dir / "result_bgextracted.fit"
    if output_file.exists():
        log.info("Background-extracted result already exists, skipping")
        return output_file

    graxpert_path = get_tool_path(config, "graxpert")
    proc = config.get("processing", {}).get("graxpert", {})
    gpu = proc.get("gpu", True)
    correction = proc.get("background_correction", "Subtraction")

    cmd = [
        graxpert_path,
        str(current_file),
        "-cli",
        "-cmd", "background-extraction",
        "-correction", correction,
        "-output", str(output_file),
    ]
    if gpu:
        cmd.extend(["-gpu", "true"])

    run_command(cmd, "GraXpert background extraction")

    if not output_file.exists():
        raise RuntimeError("Background extraction failed: output not created")

    log.info("Phase 3 complete: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phase 4 – Deconvolution / Sharpening
# ---------------------------------------------------------------------------

def phase_4_sharpen(
    results_dir: Path,
    current_file: Path,
    info: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """AI deconvolution / sharpening using Cosmic Clarity (FITS->TIFF->FITS)."""
    log.info("=" * 60)
    log.info("Phase 4: Deconvolution (Cosmic Clarity)")
    log.info("=" * 60)

    output_file = results_dir / "result_sharpened.fit"
    if output_file.exists():
        log.info("Sharpened result already exists, skipping")
        return output_file

    profile = info["profile"]
    deconv_mode = profile.get("deconv_mode", "both")
    deconv_amount = profile.get("deconv_amount", 0.4)

    cosmic_clarity_path = get_tool_path(config, "cosmic_clarity")
    gpu = config.get("processing", {}).get("cosmic_clarity", {}).get("gpu", "cuda")

    # Convert FITS to 16-bit TIFF (Cosmic Clarity requires TIFF input)
    tiff_input = results_dir / "cc_input.tiff"
    original_header = fits_to_tiff(current_file, tiff_input)

    cmd = [
        cosmic_clarity_path,
        "--input", str(tiff_input),
        "--sharpen_mode", deconv_mode,
        "--sharpen_amount", str(deconv_amount),
        "--gpu", gpu,
    ]
    run_command(cmd, "Cosmic Clarity deconvolution")

    # Locate Cosmic Clarity output (check common naming patterns)
    possible_outputs = [
        tiff_input.with_name("cc_input_sharpened.tiff"),
        tiff_input.with_name("cc_input_output.tiff"),
        results_dir / "output" / "cc_input.tiff",
        results_dir / "output" / "cc_input_sharpened.tiff",
    ]
    # Also scan for any new .tiff in results_dir
    actual_output = None
    for p in possible_outputs:
        if p.exists():
            actual_output = p
            break

    if actual_output is None:
        # Broad search as last resort
        tiff_candidates = sorted(
            results_dir.rglob("*.tiff"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for t in tiff_candidates:
            if t != tiff_input:
                actual_output = t
                break

    if actual_output is None:
        raise RuntimeError("Cosmic Clarity produced no output TIFF")

    # Convert back to FITS preserving original header
    tiff_to_fits(actual_output, output_file, original_header)

    # Clean up temporary TIFFs
    tiff_input.unlink(missing_ok=True)
    if actual_output and actual_output.exists():
        actual_output.unlink(missing_ok=True)

    log.info("Phase 4 complete: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phase 5 – Denoising
# ---------------------------------------------------------------------------

def phase_5_denoise(
    results_dir: Path,
    current_file: Path,
    info: dict[str, Any],
    config: dict[str, Any],
    denoise_override: Optional[float] = None,
) -> Path:
    """AI denoising using GraXpert."""
    log.info("=" * 60)
    log.info("Phase 5: Denoising")
    log.info("=" * 60)

    output_file = results_dir / "result_denoised.fit"
    if output_file.exists():
        log.info("Denoised result already exists, skipping")
        return output_file

    profile = info["profile"]
    strength = denoise_override if denoise_override is not None else profile.get("denoise_strength", 0.4)

    graxpert_path = get_tool_path(config, "graxpert")
    gpu = config.get("processing", {}).get("graxpert", {}).get("gpu", True)

    cmd = [
        graxpert_path,
        str(current_file),
        "-cli",
        "-cmd", "denoising",
        "-strength", str(strength),
        "-output", str(output_file),
    ]
    if gpu:
        cmd.extend(["-gpu", "true"])

    run_command(cmd, "GraXpert denoising")

    if not output_file.exists():
        raise RuntimeError("Denoising failed: output not created")

    log.info("Phase 5 complete (strength=%.2f): %s", strength, output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phases 6-7 – Star Split, Stretch & Recomposition
# ---------------------------------------------------------------------------

def phase_6_7_stretch_recompose(
    results_dir: Path,
    current_file: Path,
    info: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Star split (StarNet), independent stretching, and PixelMath recomposition."""
    log.info("=" * 60)
    log.info("Phase 6-7: Star Split, Stretch & Recomposition")
    log.info("=" * 60)

    output_file = results_dir / "final_result.fit"
    if output_file.exists():
        log.info("Final result already exists, skipping")
        return output_file

    profile = info["profile"]
    star_split = profile.get("star_split", True)
    siril_path = get_tool_path(config, "siril_cli")
    jpg_quality = config.get("processing", {}).get("output", {}).get("jpg_quality", 95)

    # Ensure input file is named as expected by the static .ssf script
    expected_input = results_dir / "result_denoised.fit"
    if current_file.resolve() != expected_input.resolve():
        if not expected_input.exists():
            shutil.copy2(str(current_file), str(expected_input))
            log.info("Copied %s -> %s", current_file.name, expected_input.name)

    if star_split:
        log.info("Using star-split workflow (StarNet + separate stretch)")
        script_path = Path(__file__).parent / "seestar_stretch.ssf"
        run_siril_script(siril_path, script_path, results_dir)
    else:
        log.info("Star split disabled for this profile; stretching directly")
        commands = [
            "load result_denoised",
            "autostretch",
            "save final_result",
            "savetif final_result",
            f"savejpg final_result {jpg_quality}",
            "close",
        ]
        temp_script = results_dir / "temp_stretch.ssf"
        write_siril_script(temp_script, commands)
        run_siril_script(siril_path, temp_script, results_dir)
        temp_script.unlink(missing_ok=True)

    if not output_file.exists():
        raise RuntimeError("Stretch/recompose failed: final_result.fit not created")

    log.info("Phase 6-7 complete: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Phase 8 – Plate Solving
# ---------------------------------------------------------------------------

def phase_8_platesolve(
    results_dir: Path,
    current_file: Path,
    info: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Plate solve the final image and embed WCS headers."""
    log.info("=" * 60)
    log.info("Phase 8: Plate Solving")
    log.info("=" * 60)

    siril_path = get_tool_path(config, "siril_cli")
    seestar = config.get("seestar", {})
    focal_length = seestar.get("focal_length", 250)
    pixel_size = seestar.get("pixel_size", 2.9)

    log.info("  Focal length: %smm | Pixel size: %sum", focal_length, pixel_size)
    log.info("  Plate scale: %.2f arcsec/pixel", pixel_size / focal_length * 206.265)

    # Write RA/DEC hint into the FITS header to speed up the solve
    if info.get("ra") is not None and info.get("dec") is not None:
        with fits.open(str(current_file), mode="update") as hdul:
            hdul[0].header["RA"] = info["ra"]
            hdul[0].header["DEC"] = info["dec"]
            if info.get("object_name"):
                hdul[0].header["OBJECT"] = info["object_name"]

    script_path = Path(__file__).parent / "seestar_platesolve.ssf"
    try:
        run_siril_script(siril_path, script_path, results_dir)
        log.info("Phase 8 complete: plate solving successful")
    except RuntimeError:
        log.warning(
            "Plate solving failed (non-fatal). "
            "The image is still valid but won't have WCS astrometry headers. "
            "Ensure Siril star catalogs are installed for offline solving."
        )

    return current_file


# ---------------------------------------------------------------------------
# Phase 9 – AstroBin Upload Prep
# ---------------------------------------------------------------------------

def phase_9_astrobin(
    subs_dir: Path,
    results_dir: Path,
    info: dict[str, Any],
    config: dict[str, Any],
    bortle_override: Optional[int] = None,
    site_override: Optional[str] = None,
) -> None:
    """Generate AstroBin acquisition CSV and description text file."""
    log.info("=" * 60)
    log.info("Phase 9: AstroBin Upload Prep")
    log.info("=" * 60)

    csv_file = results_dir / "acquisition.csv"
    desc_file = results_dir / "astrobin_description.txt"

    if csv_file.exists() and desc_file.exists():
        log.info("AstroBin files already exist, skipping")
        return

    # Parse all sub-frame headers
    fits_files = info["fits_files"]
    headers: list[fits.Header] = []
    for f in fits_files:
        try:
            headers.append(fits.getheader(str(f)))
        except Exception as e:
            log.warning("Could not read header from %s: %s", f.name, e)

    if not headers:
        log.error("No valid FITS headers found for AstroBin prep")
        return

    # Site info
    site_config = config.get("site", {})
    bortle = bortle_override or site_config.get("bortle", "")
    sqm = site_config.get("sqm", "")
    site = site_override or site_config.get("name", "")

    # Group by observation night (DATE-OBS truncated to date)
    nights: dict[str, list] = defaultdict(list)
    for h in headers:
        date_obs = h.get("DATE-OBS", "")
        date = date_obs[:10] if date_obs else "unknown"
        nights[date].append(h)

    # Build one CSV row per observation night
    rows = []
    for date, night_headers in sorted(nights.items()):
        rows.append({
            "date": date,
            "filter": "",
            "number": len(night_headers),
            "duration": night_headers[0].get("EXPTIME", info.get("exptime", 10.0)),
            "gain": night_headers[0].get("GAIN", info.get("gain", 80)),
            "binning": night_headers[0].get("XBINNING", 1),
            "sensorCooling": night_headers[0].get("CCD-TEMP", ""),
            "bortle": bortle,
            "meanSqm": sqm,
        })

    # Write CSV
    fieldnames = [
        "date", "filter", "number", "duration", "gain",
        "binning", "sensorCooling", "bortle", "meanSqm",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Generated %s (%d rows)", csv_file.name, len(rows))

    # Totals
    total_subs = sum(r["number"] for r in rows)
    exposure = float(rows[0]["duration"]) if rows else 10.0
    integration_min = total_subs * exposure / 60.0
    dates = ", ".join(sorted(nights.keys()))
    gain_val = rows[0]["gain"] if rows else 80

    # Sensor temperature average
    temps = [h.get("CCD-TEMP") for h in headers if h.get("CCD-TEMP") is not None]
    sensor_temp = f"{sum(temps) / len(temps):.0f}" if temps else "N/A"

    # Build processing steps description
    profile = info["profile"]
    steps = [
        f"Siril 1.4: Debayer ({info.get('bayer', 'GBRG')}), registration, sigma-clip stacking",
        "Siril 1.4: Photometric Color Calibration",
        "GraXpert 3.x: AI background extraction",
    ]
    if profile.get("deconv_mode"):
        steps.append(
            f"Cosmic Clarity: AI deconvolution "
            f"({profile['deconv_mode']}, {profile.get('deconv_amount', 0.4)})"
        )
    steps.append(
        f"GraXpert 3.x: AI denoise (strength {profile.get('denoise_strength', 0.4)})"
    )
    if profile.get("star_split"):
        steps.extend([
            "Siril 1.4 + StarNet v2: Star/nebula split",
            "Siril 1.4: GHS stretch (starless aggressive, stars gentle)",
            "Siril 1.4: PixelMath recomposition",
        ])
    else:
        steps.append("Siril 1.4: Autostretch")
    steps.append("Siril 1.4: Plate solved (Gaia DR3)")

    # Build description from template
    astrobin_config = config.get("astrobin", {})
    default_template = (
        "{object_common_name}\n\n"
        "Captured with ZWO Seestar S50 from {site} (Bortle {bortle})\n\n"
        "Acquisition:\n"
        "  {num_subs} x {exposure}s subs ({integration_min:.1f} min total integration)\n"
        "  Dates: {dates}\n"
        "  Gain: {gain} | Sensor temp: ~{sensor_temp}C\n\n"
        "Processing (automated pipeline):\n"
        "  {processing_steps}"
    )
    template = astrobin_config.get("description_template", default_template)

    description = template.format(
        object_common_name=info.get("main_id", info["object_name"]),
        object_name=info["object_name"],
        site=site,
        bortle=bortle,
        num_subs=total_subs,
        exposure=exposure,
        integration_min=integration_min,
        dates=dates,
        gain=gain_val,
        sensor_temp=sensor_temp,
        processing_steps="\n  ".join(steps),
    )

    with open(desc_file, "w") as f:
        f.write(description)

    log.info("Generated %s", desc_file.name)

    title_template = astrobin_config.get(
        "title_template",
        "{object_name} | Seestar S50 | {integration_min:.0f} min",
    )
    title = title_template.format(
        object_name=info["object_name"],
        integration_min=integration_min,
    )
    log.info("  Title: \"%s\"", title)
    log.info("  %d nights, %d subs, %.1f min integration", len(nights), total_subs, integration_min)
    log.info("Phase 9 complete")


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full pipeline (Phases 0-9)."""
    start_time = time.time()

    subs_dir = Path(args.subs_directory).resolve()
    if not subs_dir.exists():
        log.error("Subs directory not found: %s", subs_dir)
        sys.exit(1)

    # Output directory
    results_dir = Path(args.output).resolve() if args.output else subs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Add file logging
    file_handler = logging.FileHandler(results_dir / "pipeline.log", mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(file_handler)

    log.info("Seestar S50 Astrophotography Pipeline")
    log.info("Subs directory: %s", subs_dir)
    log.info("Results directory: %s", results_dir)

    # Load config and profiles
    script_dir = Path(__file__).parent
    config = load_config(script_dir / "pipeline.yaml")
    profiles = load_profiles(script_dir / "profiles.yaml")

    from_phase = args.from_phase if args.from_phase is not None else 0

    # ------------------------------------------------------------------
    # Phase 0: Identify
    # ------------------------------------------------------------------
    if from_phase <= 0:
        info = phase_0_identify(
            subs_dir, config, profiles,
            target_override=args.target,
            profile_override=args.profile,
        )
    else:
        # Try to resume from cached identification
        info_cache = results_dir / "pipeline_info.json"
        if info_cache.exists():
            with open(info_cache) as f:
                info = json.load(f)
            info["profile"] = profiles.get(
                info.get("profile_name", "generic_dso"),
                profiles.get("generic_dso", {}),
            )
            lights_dir = subs_dir / "lights"
            search_dir = lights_dir if lights_dir.exists() else subs_dir
            info["fits_files"] = sorted(
                list(search_dir.glob("*.fits")) + list(search_dir.glob("*.fit"))
            )
            info["search_dir"] = search_dir
            log.info("Resumed from cached ID: %s (%s)", info["object_name"], info["profile_name"])
        else:
            info = phase_0_identify(
                subs_dir, config, profiles,
                target_override=args.target,
                profile_override=args.profile,
            )

    # Cache identification for --from-phase resume
    info_cache = results_dir / "pipeline_info.json"
    serializable = {
        k: v for k, v in info.items()
        if k not in ("profile", "fits_files", "search_dir")
    }
    serializable["fits_files"] = [str(f) for f in info.get("fits_files", [])]
    serializable["search_dir"] = str(info.get("search_dir", ""))
    with open(info_cache, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Phase 1: Stack
    # ------------------------------------------------------------------
    if from_phase <= 1:
        current_file = phase_1_stack(subs_dir, results_dir, info, config)
    else:
        current_file = results_dir / "result.fit"

    if args.stack_only:
        elapsed = time.time() - start_time
        log.info("Stack-only mode: pipeline complete in %dm %ds",
                 int(elapsed // 60), int(elapsed % 60))
        return

    # ------------------------------------------------------------------
    # Phase 2: Calibrate
    # ------------------------------------------------------------------
    if from_phase <= 2:
        current_file = phase_2_calibrate(results_dir, current_file, info, config)
    else:
        current_file = results_dir / "result_calibrated.fit"

    # ------------------------------------------------------------------
    # Phase 3: Background extraction
    # ------------------------------------------------------------------
    if from_phase <= 3:
        current_file = phase_3_background(results_dir, current_file, config)
    else:
        current_file = results_dir / "result_bgextracted.fit"

    # ------------------------------------------------------------------
    # Phase 4: Deconvolution (skippable)
    # ------------------------------------------------------------------
    if from_phase <= 4 and not args.skip_deconvolution:
        current_file = phase_4_sharpen(results_dir, current_file, info, config)
    elif args.skip_deconvolution:
        log.info("Skipping Phase 4: Deconvolution (--skip-deconvolution)")
    else:
        candidate = results_dir / "result_sharpened.fit"
        current_file = candidate if candidate.exists() else current_file

    # ------------------------------------------------------------------
    # Phase 5: Denoise
    # ------------------------------------------------------------------
    if from_phase <= 5:
        current_file = phase_5_denoise(
            results_dir, current_file, info, config,
            denoise_override=args.denoise_strength,
        )
    else:
        current_file = results_dir / "result_denoised.fit"

    # ------------------------------------------------------------------
    # Phases 6-7: Star split, stretch & recompose
    # ------------------------------------------------------------------
    if from_phase <= 7:
        current_file = phase_6_7_stretch_recompose(
            results_dir, current_file, info, config,
        )
    else:
        current_file = results_dir / "final_result.fit"

    # ------------------------------------------------------------------
    # Phase 8: Plate solve (skippable)
    # ------------------------------------------------------------------
    if from_phase <= 8 and not args.skip_platesolve:
        current_file = phase_8_platesolve(results_dir, current_file, info, config)
    elif args.skip_platesolve:
        log.info("Skipping Phase 8: Plate solving (--skip-platesolve)")

    # ------------------------------------------------------------------
    # Phase 9: AstroBin prep (skippable)
    # ------------------------------------------------------------------
    if not args.skip_astrobin:
        phase_9_astrobin(
            subs_dir, results_dir, info, config,
            bortle_override=args.bortle,
            site_override=args.site,
        )
    else:
        log.info("Skipping Phase 9: AstroBin prep (--skip-astrobin)")

    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("Pipeline complete! Total time: %dm %ds",
             int(elapsed // 60), int(elapsed % 60))
    log.info("Results in: %s", results_dir)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seestar S50 Astrophotography Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python astro_pipeline.py "D:\\Astro\\M42_subs"
  python astro_pipeline.py "D:\\Astro\\M42_subs" --target "M 42"
  python astro_pipeline.py "D:\\Astro\\M42_subs" --profile emission_nebula
  python astro_pipeline.py "D:\\Astro\\M42_subs" --denoise-strength 0.3
  python astro_pipeline.py "D:\\Astro\\M42_subs" --skip-deconvolution
  python astro_pipeline.py "D:\\Astro\\M42_subs" --stack-only
  python astro_pipeline.py "D:\\Astro\\M42_subs" --from-phase 3
""",
    )

    parser.add_argument(
        "subs_directory",
        help="Path to directory containing .fits/.fit sub-frames",
    )
    parser.add_argument(
        "--target",
        help="Force target name (skips FITS header / folder name detection)",
    )
    parser.add_argument(
        "--profile",
        help="Override auto-detected processing profile",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        help="Override profile's denoise strength (0.0-1.0)",
    )
    parser.add_argument(
        "--skip-deconvolution",
        action="store_true",
        help="Skip Cosmic Clarity sharpening",
    )
    parser.add_argument(
        "--skip-platesolve",
        action="store_true",
        help="Skip plate solving",
    )
    parser.add_argument(
        "--skip-astrobin",
        action="store_true",
        help="Skip AstroBin CSV/description generation",
    )
    parser.add_argument(
        "--stack-only",
        action="store_true",
        help="Only run stacking (phase 1), skip all post-processing",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        help="Resume from phase N (uses existing intermediate files)",
    )
    parser.add_argument(
        "--bortle",
        type=int,
        help="Override Bortle class for AstroBin CSV",
    )
    parser.add_argument(
        "--site",
        help="Override site name for AstroBin description",
    )
    parser.add_argument(
        "--output",
        help="Custom output directory (default: <subs_dir>/results)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        log.exception("Traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
