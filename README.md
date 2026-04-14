# M2FS Echelle Reduction Pipeline

This pipeline consists of two main scripts, `image_processing.py` and `reduce_echelle.py`, plus three helper scripts. Most processing steps use [PyRAF](http://ascl.net/1207.011).

`image_processing.py` handles preprocessing (bias and dark correction), image stitching, and stacking.

`reduce_echelle.py` performs the echelle-specific processing steps:
1.) Automatically identifies apertures and opens a preview for manual verification.
2.) Scattered-light subtraction (interactive).
3.) Master-flat creation.
4.) Flat-field correction.
5.) Review of extracted apertures. An interactive plot opens where the user can edit aperture assignments by either reassigning an aperture to a different star (key `[e]`) or deleting apertures (key `[d]`). After editing, apertures can be reassigned automatically with key `[f]`. The process is finalized and aperture information is saved by pressing key `[q]` twice.
6.) Aperture extraction using the aperture IDs from the previous step.
7.) Cosmic-ray removal on extracted apertures using the `lineclean` task.
8.) Prompts the user to either identify lines manually or use a reference star for reidentification.
9.) Carries line IDs from step 8 to other stars. The drift relative to the reference star along the CCD is extrapolated and logged. Each reidentified spectrum is then opened again in `ecid` for manual review and wavelength-solution refitting; manual line reidentification is also possible.
10.) Creates a lamp reference for stellar images.
11.) Dispersion correction.

Both scripts attempt to automatically classify images by exposure type and assign labels (`dark`, `quartz`, `twilight`, `fibermap`, `object`), while prompting the user when the label is ambiguous.

Processing can start or stop at any step, which allows the workflow to resume after partial processing or during reprocessing (especially useful in line-identification stages).

## Run

Run commands from a night directory containing the raw FITS files (or from its `proc/` directory).

### one-command run

This runs preprocessing first and then starts echelle reduction:

```bash
python reduce_echelle.py \
	--input-dir . \
	--night 15Sep2014 \
	--shoe B \
	--run-preprocess \
	--preprocess-bias
```

Useful optional flags:
- `--start-step N --end-step M` to run only a subset of echelle steps.
- `--object-name "<substring>"` to restrict auto-discovery to one science target.

### Two-stage run

1.) Preprocess:

```bash
python image_processing.py \
	--infiles ./infiles \
	--bias
```

2.) Echelle reduction:

```bash
python reduce_echelle.py \
	--input-dir . \
	--night 15Sep2014 \
	--shoe B
```

### Resume examples

Resume only later echelle steps (for example, wavelength ID onward):

```bash
python reduce_echelle.py \
	--input-dir . \
	--night 15Sep2014 \
	--shoe B \
	--start-step 8 \
	--end-step 11
```

Run only preprocessing and stop before echelle steps:

```bash
python reduce_echelle.py \
	--input-dir . \
	--night 15Sep2014 \
	--shoe B \
	--run-preprocess \
	--preprocess-only
```