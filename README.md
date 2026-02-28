# Seestar S50 Astrophotography Pipeline

Automated stacking and post-processing pipeline for the ZWO Seestar S50 smart telescope. One command takes raw `.fits` sub-frames and produces a processed, plate-solved image plus AstroBin-ready metadata files.

```cmd
python astro_pipeline.py "D:\Astro\M42_subs"
```

No `--target` flag needed. The pipeline reads FITS headers, identifies the object via SIMBAD, selects optimal processing parameters, processes everything, plate-solves the result, and generates AstroBin acquisition data.

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.12+** | Pipeline orchestration | [python.org](https://www.python.org/downloads/) (check "Add to PATH") |
| **Siril 1.4.x** | Stacking, calibration, stretching, plate solving | [siril.org](https://siril.org/download/) |
| **GraXpert 3.x** | AI background extraction + denoising | `pip install graxpert[cuda]` or [standalone](https://github.com/Steffenhir/GraXpert/releases) |
| **Cosmic Clarity** | AI deconvolution/sharpening | [setiastro.com](https://www.setiastro.com/cosmic-clarity) |
| **NVIDIA GPU** | CUDA acceleration (recommended) | [nvidia.com/drivers](https://www.nvidia.com/drivers/) |

StarNet v2 is integrated into Siril 1.4 and does not require a separate install.

## Installation

### 1. Install External Tools

Install Siril, GraXpert, and Cosmic Clarity following the links above. Ensure `siril-cli` is in your PATH:

```cmd
siril-cli --version
```

### 2. Install Python Dependencies

```cmd
pip install astropy astroquery pyyaml Pillow numpy
```

### 3. Download Siril Star Catalogs (for offline plate solving)

Open Siril and run the built-in Catalogue Installer script:

1. Go to **Scripts > Python Scripts > Siril_Catalog_Installer**
2. Select **Astrometric** catalog type and choose your sky coverage (or **All** for full sky)
3. Click **Install** (~1.5 GB download, one-time setup)

The script automatically configures the catalog path in Siril's preferences. To verify or change the path manually, open **☰ > Preferences (Ctrl+P) > Astrometry > Local star catalogues**.

### 4. Configure the Pipeline

Edit `pipeline.yaml` to set your tool paths and site defaults:

```yaml
tools:
  siril_cli: "C:\\Program Files\\Siril\\bin\\siril-cli.exe"
  graxpert: "graxpert"
  cosmic_clarity: "C:\\AstroTools\\CosmicClarity\\SetiAstroCosmicClarity.exe"

site:
  name: "My Observatory"
  bortle: 4
  sqm: 20.5
```

### 5. Validate Your Setup

```cmd
setup.bat
```

This checks all tools, Python dependencies, and config files.

## Usage

### Basic (auto-detects everything)

```cmd
python astro_pipeline.py "D:\Astro\M42_subs"
```

### Common Options

```cmd
# Force a target name
python astro_pipeline.py "D:\Astro\mystery_subs" --target "NGC 7000"

# Override processing profile
python astro_pipeline.py "D:\Astro\M31_subs" --profile galaxy

# Custom denoise strength
python astro_pipeline.py "D:\Astro\M42_subs" --denoise-strength 0.3

# Skip sharpening (if stars already look good)
python astro_pipeline.py "D:\Astro\M31_subs" --skip-deconvolution

# Stack only (no post-processing)
python astro_pipeline.py "D:\Astro\M42_subs" --stack-only

# Resume from a specific phase
python astro_pipeline.py "D:\Astro\M42_subs" --from-phase 3

# Override site info for AstroBin
python astro_pipeline.py "D:\Astro\M42_subs" --bortle 3 --site "Cherry Springs, PA"

# Custom output directory
python astro_pipeline.py "D:\Astro\M42_subs" --output "D:\Results\M42"
```

### Full CLI Reference

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

## Pipeline Phases

| Phase | Tool | Input | Output |
|-------|------|-------|--------|
| 0. Identify | astropy + SIMBAD | FITS headers | Processing profile |
| 1. Stack | Siril | Raw .fits subs | `result.fit` |
| 2. Calibrate | Siril | Stacked image | `result_calibrated.fit` |
| 3. Background | GraXpert | Calibrated image | `result_bgextracted.fit` |
| 4. Sharpen | Cosmic Clarity | BG-extracted image | `result_sharpened.fit` |
| 5. Denoise | GraXpert | Sharpened image | `result_denoised.fit` |
| 6-7. Stretch | Siril + StarNet | Denoised image | `final_result.fit/.tif/.jpg` |
| 8. Plate solve | Siril | Final image | WCS headers embedded |
| 9. AstroBin | Python | Sub headers | `acquisition.csv` + description |

Each phase is **idempotent** -- if the output already exists, the phase is skipped. This lets you re-run after tweaking parameters without re-stacking.

## Processing Profiles

Profiles are auto-selected based on SIMBAD object classification:

| Profile | Objects | Denoise | Deconv | Star Split |
|---------|---------|---------|--------|------------|
| `emission_nebula` | M42, NGC 7000, IC 1396 | 0.3 | non_stellar @ 0.4 | Yes |
| `galaxy` | M31, M51, NGC 891 | 0.5 | both @ 0.6 | No |
| `star_cluster` | M13, M45, NGC 884 | 0.5 | stellar @ 0.5 | No |
| `planetary_nebula` | M57, NGC 7293 | 0.4 | non_stellar @ 0.5 | Yes |
| `dark_nebula` | B33, LDN 1622 | 0.2 | non_stellar @ 0.3 | Yes |
| `star_field` | Double stars, variables | 0.5 | stellar @ 0.5 | No |
| `generic_dso` | Fallback | 0.4 | both @ 0.4 | Yes |

Override with `--profile <name>` or fine-tune with `--denoise-strength`.

## Output Files

After a complete run, `results/` contains:

```
results/
  final_result.fit            # Plate-solved FITS with WCS headers
  final_result.tif            # 16-bit TIFF (ASTRO-TIFF with WCS)
  final_result.jpg            # 95% quality JPEG, ready to upload
  result.fit                  # Stacked linear master (for reprocessing)
  acquisition.csv             # AstroBin "Import CSV" ready
  astrobin_description.txt    # Copy-paste into AstroBin description
  pipeline.log                # Full processing log
  pipeline_info.json          # Cached identification (for --from-phase)
```

## Seestar S50 Setup

In the Seestar app, enable saving individual sub-frames:

**Advanced Settings > "Save each frame in enhancing" = ON**

After a session, copy the `_subs` folder to your PC:

```cmd
xcopy /E "Seestar\MyWorks\M42_subs\*.fit" "D:\Astro\M42_subs\lights\"
```

The pipeline looks for subs in either a `lights/` subfolder or directly in the provided directory.

## Uploading to AstroBin

1. Upload `final_result.jpg` (or `.tif`) to AstroBin
2. Paste the title from `astrobin_description.txt`
3. Click **Import CSV** in acquisition details, select `acquisition.csv`
4. Paste the description from `astrobin_description.txt`
5. Select your saved Seestar S50 equipment profile
6. Publish

## Tuning Tips

**For faint nebulosity** (IFN, faint edges): `--denoise-strength 0.2`

**For bright nebulae** (Orion, Lagoon): `--denoise-strength 0.5`

**For galaxies**: `--profile galaxy` (uses both stellar + non-stellar deconv)

**More subs = better results.** 200+ subs at 10s each is ideal for faint structures.

## Troubleshooting

- **PCC fails**: Ensure Siril star catalogs are downloaded and internet is available for the first solve
- **Plate solving fails**: Install the Gaia DR3 astrometric catalog via Scripts > Python Scripts > Siril_Catalog_Installer
- **GraXpert not found**: Try `pip install graxpert[cuda]` or set the full path in `pipeline.yaml`
- **Cosmic Clarity fails**: Ensure TIFF output is being generated; check the tool's output directory
- **SIMBAD query fails**: The pipeline caches results in `simbad_cache.json`; works offline after first run

## Project Structure

```
astro_pipeline.py           # Main orchestrator
seestar_stack.ssf           # Siril stacking script
seestar_calibrate.ssf       # Siril calibration script
seestar_stretch.ssf         # Siril star split + stretch + recompose script
seestar_platesolve.ssf      # Siril plate solving script
profiles.yaml               # Processing profiles by object type
pipeline.yaml               # User configuration (tool paths, site defaults)
setup.bat                   # Setup validation script
README.md                   # This file
```
