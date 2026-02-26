# Seestar S50 Astrophotography Pipeline

You are building a CLI-driven astrophotography pipeline that takes raw `.fits` sub-frames from a ZWO Seestar S50 smart telescope and produces a processed, plate-solved image plus AstroBin metadata files. One command, zero manual intervention.

## Specification

Read `seestar-astro-pipeline-plan.md` for the complete phase-by-phase specification before writing any code. It contains exact CLI commands, SIMBAD OTYPE mappings, processing profile parameters, Siril script syntax, FITS header field mappings, AstroBin CSV format, and config schemas. Treat it as the source of truth for all implementation decisions.

## Target Environment

Windows 11 with Python 3.12+, NVIDIA GPU with CUDA. All file paths use backslashes. The shell is `cmd.exe`. External tools are invoked via `subprocess.run()`.

## External Tools (all pre-installed by user)

|Tool          |Binary                                  |Invocation Pattern                                              |
|--------------|----------------------------------------|----------------------------------------------------------------|
|Siril 1.4.x   |`siril-cli`                             |Runs `.ssf` script files via `-s` flag with `-d` for working dir|
|GraXpert 3.x  |`graxpert` (pip) or `GraXpert-win64.exe`|`-cli` flag for headless, `-cmd` for operation, `-gpu true`     |
|Cosmic Clarity|`SetiAstroCosmicClarity.exe`            |`--input`, `--sharpen_mode`, `--sharpen_amount`, `--gpu cuda`   |
|StarNet v2    |Integrated in Siril 1.4                 |`starnet` command inside Siril scripts                          |

## Project Structure

```
seestar-pipeline/
  astro_pipeline.py           # Main orchestrator - the only file the user runs
  seestar_stack.ssf           # Siril stacking script (lights only, no darks/biases/flats)
  seestar_calibrate.ssf       # Siril color calibration script
  seestar_stretch.ssf         # Siril star split + GHS stretch + recomposition
  seestar_platesolve.ssf      # Siril plate solving script
  profiles.yaml               # Processing profiles keyed by object type
  pipeline.yaml               # User config: tool paths, site defaults, AstroBin settings
  setup.bat                   # Validates all tools are installed and accessible
  README.md                   # User-facing setup and usage guide
```

## Pipeline Phases (sequential)

1. **Identify** - Read FITS headers → SIMBAD query → select processing profile
1. **Stack** - Siril: debayer, register, sigma-clip stack → `result.fit`
1. **Calibrate** - Siril: autocrop, remove green, PCC → `result_calibrated.fit`
1. **Background** - GraXpert: AI background extraction → `result_bgextracted.fit`
1. **Sharpen** - Cosmic Clarity: AI deconvolution (requires FITS→TIFF→FITS conversion)
1. **Denoise** - GraXpert: AI denoise → `result_denoised.fit`
1. **Star split** - Siril + StarNet: separate stars from nebula, stretch each independently
1. **Recompose** - Siril PixelMath: merge stretched starless + stars → final outputs (.fit, .tif, .jpg)
1. **Plate solve** - Siril: match stars to Gaia DR3, embed WCS headers
1. **AstroBin prep** - Python: parse sub headers → write `acquisition.csv` + `astrobin_description.txt`

Each phase checks if its output already exists before running (idempotent). All intermediate files are preserved in a `results/` subdirectory.

## Deliverables

Build these files in order. Verify each works before moving to the next.

1. `pipeline.yaml` - Config template with documented defaults for tool paths, Seestar S50 specs, site info, AstroBin settings
1. `profiles.yaml` - All processing profiles from the plan’s OTYPE mapping table (emission_nebula, galaxy, star_cluster, planetary_nebula, dark_nebula, star_field, generic_dso)
1. `seestar_stack.ssf` - Siril stacking script (exact syntax in plan)
1. `seestar_calibrate.ssf` - Siril calibration script
1. `seestar_stretch.ssf` - Siril star split, stretch, and recomposition script
1. `seestar_platesolve.ssf` - Siril plate solving script
1. `astro_pipeline.py` - Main orchestrator covering all 10 phases
1. `setup.bat` - Checks for siril-cli, graxpert/GraXpert, CosmicClarity, Python deps, CUDA availability
1. `README.md` - Setup instructions and usage examples

## CLI Interface

```
python astro_pipeline.py <subs_directory> [options]

Required:
  subs_directory          Path to directory containing .fits/.fit sub-frames

Options:
  --target NAME           Force target name (skips FITS header / folder name detection)
  --profile NAME          Override auto-detected processing profile
  --denoise-strength N    Override profile's denoise strength (0.0-1.0)
  --skip-deconvolution    Skip Cosmic Clarity sharpening
  --skip-platesolve       Skip plate solving
  --skip-astrobin         Skip AstroBin CSV/description generation
  --stack-only            Only run stacking (phase 1), skip all post-processing
  --from-phase N          Resume from phase N (uses existing intermediate files)
  --bortle N              Override Bortle class for AstroBin CSV
  --site NAME             Override site name for AstroBin description
  --output DIR            Custom output directory (default: <subs_dir>/results)
```

Use `argparse`. No subcommands. Flat and simple.

## Seestar S50 Constants

These are fixed hardware specs that never change:

- Focal length: 250mm
- Pixel size: 2.9μm
- Plate scale: ~2.39 arcsec/pixel
- Field of view: ~0.72° x 1.28°
- Bayer pattern: GBRG (verify from FITS `BAYERPAT` header, fall back to GBRG)
- Typical gain: 80
- Typical exposure: 10s
- No filter wheel, no active cooling, no separate calibration frames

## Things You Will Get Wrong

Read this section carefully. These are the mistakes most likely to cause bugs.

**Siril scripts are NOT Python.** They use Siril’s own command language. No colons, no indentation, no quotes around string args. Commands are one-per-line. Comments start with `#`. See the exact syntax in the plan’s Phase 1/2/6/7/8 sections.

**Cosmic Clarity needs TIFF, not FITS.** Convert to 16-bit TIFF before calling it, convert back to FITS after. Use `astropy.io.fits` + `Pillow` for this. Preserve the original FITS header metadata through the round-trip.

**GraXpert’s `-cli` flag varies by version.** The pip-installed version uses `graxpert` as the command. The standalone `.exe` uses `GraXpert-win64.exe`. Support both via the `pipeline.yaml` config.

**SIMBAD rate limiting.** Cache query results locally (a simple JSON file keyed by object name). Don’t hit SIMBAD on every run of the same target.

**Multi-night sub aggregation in Phase 9.** Group subs by `DATE-OBS` truncated to date. One CSV row per observation night, not one row per sub.

**The `starnet` command in Siril strips stars in-place.** The workflow is: load image, run `starnet -stretch`, save as starless, then use PixelMath to compute stars = original - starless.

**Windows paths in subprocess calls.** Use raw strings or `pathlib.Path`. Siril’s `-d` flag needs the directory path without a trailing backslash.

**PCC needs coordinates.** Pass the RA/DEC from Phase 0’s FITS header parsing or SIMBAD lookup to the calibration script so PCC can plate-solve internally.

## Verification

After building, verify with these checks:

```cmd
python astro_pipeline.py --help
python -c "import yaml; import astropy; import astroquery; print('deps OK')"
python -c "import yaml; yaml.safe_load(open('pipeline.yaml')); print('config OK')"
python -c "import yaml; yaml.safe_load(open('profiles.yaml')); print('profiles OK')"
```

The setup.bat should exit non-zero if any tool is missing.

## Code Style

Python 3.12+, type hints on function signatures, `logging` module (not print statements), `pathlib.Path` for all file operations. Use `PyYAML` for config. No classes unless they genuinely simplify the code – top-level functions organized by phase are fine. Keep it readable for someone who knows Python but not astrophotography.
