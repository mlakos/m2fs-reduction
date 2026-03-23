"""
reduce_echelle.py
=================
Post-notebook echelle reduction pipeline for McDonald 2.7 m / TS23.

Assumes the notebook has already produced one stacked mosaic per frame type
per shoe (bias-subtracted, dark-subtracted, CR-cleaned, assembled), e.g.:

    Quartz-r0650--ot-B-full-Dcrr.fits
    ThAr-r0651--ot-B-full-Dcrr.fits
    Object-...-sci_15Sep2014_B-b-d-cr.fits
    Twilight-...-ot-B-full-Dcrr.fits

Pipeline steps
--------------
  1.  apall on QUARTZ          – find / trace / NO extract
  2.  apscatter                – on quartz, thar, object, twilight
      – outputs new files with suffix -sl.fits
  3.  Normalised master flat
        a. fmedian  (xwindow=10, ywindow=1)  →  <quartz>_med.fits
        b. imarith  quartz / median_flat     →  <quartz>_nflat.fits
        c. imreplace  clip ≤0 and ≥5  →  1  (in-place on nflat)
  4.  ccdproc flat-field correction  – thar, object, twilight  → *-F.fits
  5.  Interactive aperture-trace preview  (aperture_preview.py)
      – runs on the traced QUARTZ reference image
        – shows all traces coloured by star assignment
        – user corrects assignments if needed, then presses q
  6.  Per-star apall extraction  (object + thar)
        – the pattern from step 5 is used to build one aperture list per star
        – for each star apall is called once for the object and once for the
          thar, extracting only that star's apertures
        – outputs are named  <stem>_star<N>_ec.fits
  7.  Second cosmic-ray removal  (lineclean on extracted object spectra)
        – fits spline3 order=6 along the dispersion axis of each order
        – low_rej=50 (preserves real flux), high_rej=3 (kills CRs)
        – ThAr spectra are left untouched
        – outputs: <stem>_star<N>_ec-crr2.fits
  8.  Manual wavelength identification  (ecidentify on reference star only)
        a. ecidentify  (interactive)  on the ThAr of the first (reference) star
        b. refspec     assign the solution to that star's CR-cleaned object
        – The solution is saved to the IRAF database for step 9 to propagate
  9.  Automatic wavelength propagation  (ecreidentify to remaining stars)
        a. ecreidentify (automatic)   propagates solution from ref star to others
        b. refspec     assign the propagated solution to each object
        c. dispcor     implant and linearise the wavelength solution
        – outputs: <stem>_star<N>_ec-crr2-wl.fits
        – workflow: run step 8 (interactive), then step 9 (automatic on all stars)

Usage
-----
    # Auto-discovery (metadata-driven):
    python reduce_echelle.py --input-dir . --night 15Sep2014 --shoe B

    # Manual overrides (can be mixed with auto-discovery):
    python reduce_echelle.py \
        --quartz   Quartz-r0650--ot-B-full-Dcrr.fits   \
        --thar     ThAr-r0651--ot-B-full-Dcrr.fits     \
        --object   jsimonScl_h3_sci_15Sep2014_B-b-d-cr.fits \
        --twilight Twilight-r0652--ot-B-full-Dcrr.fits  \
        [--rdnoise 3.06]  [--gain 0.584]  [--nstars 4]
        [--sep 5.0]  [--dispaxis 1]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import textwrap

import numpy as np
from astropy.io import fits
from pyraf import iraf


# ---------------------------------------------------------------------------
# IRAF package loading
# ---------------------------------------------------------------------------

def load_packages():
    iraf.noao()
    iraf.imred()
    iraf.ccdred()
    iraf.echelle()
    iraf.images()
    iraf.imutil()
    iraf.imfit()
    iraf.imfilter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stem(path):
    """Basename without .fits extension."""
    return os.path.splitext(os.path.basename(path))[0]


def write_list(listpath, items):
    with open(listpath, "w") as fh:
        for item in items:
            fh.write(str(item) + "\n")


def iraf_delete(path):
    """Remove a FITS file if it exists, silently."""
    if os.path.exists(path):
        iraf.imdelete(path, verify=iraf.no)


def section_banner(msg):
    bar = "─" * 72
    print(f"\n{bar}\n  {msg}\n{bar}")


def read_required_metadata(filepath):
    """Read required identity keys from a FITS header.

    This pipeline only supports header-rich files and fails fast when required
    keys are absent.
    """
    hdr = fits.getheader(filepath)
    required = ["OBJECT", "EXPTYPE", "NIGHT", "SHOE"]
    missing = [k for k in required if k not in hdr]
    if missing:
        raise RuntimeError(
            f"Missing required metadata {missing} in {filepath}. "
            "Re-run image_processing.py with metadata stamping enabled."
        )
    return {
        "path": filepath,
        "OBJECT": str(hdr["OBJECT"]),
        "EXPTYPE": str(hdr["EXPTYPE"]),
        "NIGHT": str(hdr["NIGHT"]),
        "SHOE": str(hdr["SHOE"]),
        "STACKED": bool(hdr.get("STACKED", False)),
        "STACKTYPE": str(hdr.get("STACKTYPE", "")),
        "STACKMOD": str(hdr.get("STACKMOD", "")),
    }


def is_stacked_product(meta):
    """Return True if metadata indicates a final stacked product."""
    if meta.get("STACKED", False):
        return True

    stype = str(meta.get("STACKTYPE", "")).strip().lower()
    smod = str(meta.get("STACKMOD", "")).strip().lower()
    if stype in {"sum", "median", "average"}:
        return True
    if smod in {"sum", "median", "average"}:
        return True

    exptype = str(meta.get("EXPTYPE", "")).strip().lower()
    if exptype.endswith("_stack"):
        return True

    path = os.path.basename(meta["path"]).lower()
    if any(tag in path for tag in ("_sstack", "_mstack", "_astack")):
        return True

    return False


def is_pipeline_intermediate(path):
    """Return True for files generated by reduce_echelle pipeline steps."""
    name = os.path.basename(path).lower()

    # Step 2/3/4 intermediates.
    if name.endswith(("-sl.fits", "-f.fits", "_med.fits", "_nflat.fits", "_ff.fits")):
        return True

    # Step 6 extracted outputs.
    if re.search(r"_star\d+_ec\.fits$", name):
        return True

    return False


def classify_role(meta):
    """Classify a frame role from FITS header metadata."""
    exptype = meta["EXPTYPE"].strip().lower()
    objname = meta["OBJECT"].strip().lower()
    text = f"{exptype} {objname}"

    if "twilight" in text:
        return "twilight"
    if "quartz" in text:
        return "quartz"
    if "thar" in text or "thne" in text or "lamp" in text or "arc" in text:
        return "thar"
    if "object" in text or "science" in text or "sci" in text:
        return "object"
    return None


def processing_rank(meta):
    """Score a candidate by reduction level using filename suffix heuristics.

    Higher is better; this helps auto-discovery prefer products from
    image_processing.py Step 9 (stacked) over intermediate or raw frames.
    """
    path = os.path.basename(meta["path"]).lower()
    exptype = meta["EXPTYPE"].lower()
    role = classify_role(meta)
    score = 0

    # Penalize per-chip raws strongly; reduction should use mosaics/stacks.
    if re.search(r"[br]\d{4}c[1-4]", path):
        score -= 500

    # Prefer stacked outputs from step 9.
    if "_stack" in exptype:
        score += 400
    if "_sstack" in path:
        score += 450
    if "_mstack" in path:
        score += 420

    # Next-best are assembled mosaics and CR-cleaned variants.
    if "-full" in path:
        score += 120
    if "-mcrr" in path:
        score += 80
    if "-ot" in path:
        score += 30

    # Slight role-specific preference: science object should be sum-stacked.
    if role == "object" and "_sstack" in path:
        score += 80

    return score


def _select_unique_candidate(role, candidates):
    """Return one candidate per role, preferring stacked products when present."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    scored = sorted(candidates, key=processing_rank, reverse=True)
    if len(scored) >= 1:
        top_rank = processing_rank(scored[0])
        tied = [c for c in scored if processing_rank(c) == top_rank]

        if len(tied) == 1:
            return scored[0]

        # If tied candidates differ only in SHOE, pick B shoe as default
        if len(tied) > 1 and len(set(c["SHOE"] for c in tied)) > 1:
            b_shoe = [c for c in tied if str(c["SHOE"]).upper() == "B"]
            if b_shoe:
                return b_shoe[0]
            r_shoe = [c for c in tied if str(c["SHOE"]).upper() == "R"]
            if r_shoe:
                return r_shoe[0]
            return tied[0]

        # Identical rank and same shoe: still ambiguous, fail
        if len(tied) > 1:
            pass

    msg = [f"Ambiguous candidates for role '{role}':"]
    for c in scored:
        rank = processing_rank(c)
        msg.append(
            f"  - {c['path']}  (rank={rank}, EXPTYPE={c['EXPTYPE']}, "
            f"OBJECT={c['OBJECT']}, NIGHT={c['NIGHT']}, SHOE={c['SHOE']})"
        )
    msg.append("Please pass an explicit --{role} file.")
    raise RuntimeError("\n".join(msg).replace("{role}", role))


def discover_inputs(input_dir, night=None, shoe=None, object_name=None, required_roles=None):
    """Auto-discover required role files from metadata-rich FITS products."""
    required_roles = set(required_roles or ("quartz", "thar", "object", "twilight"))
    fits_paths = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    by_role = {"quartz": [], "thar": [], "object": [], "twilight": []}
    skipped = 0

    for path in fits_paths:
        try:
            meta = read_required_metadata(path)
        except Exception:
            skipped += 1
            continue

        # Auto-discovery should only consider final stacked products.
        if not is_stacked_product(meta):
            continue

        # Exclude products generated by this script; role discovery should use
        # base stacked inputs from image_processing only.
        if is_pipeline_intermediate(meta["path"]):
            continue

        if night and str(meta["NIGHT"]) != str(night):
            continue
        if shoe and str(meta["SHOE"]).upper() != str(shoe).upper():
            continue

        role = classify_role(meta)
        if role is None:
            continue
        if object_name and role == "object":
            if object_name.lower() not in meta["OBJECT"].lower():
                continue
        by_role[role].append(meta)

    selected = {}

    # Resolve object first; if night/shoe not given, use its metadata to
    # constrain quartz/thar/twilight so we do not mix both shoes.
    object_meta = None
    if "object" in required_roles:
        object_meta = _select_unique_candidate("object", by_role["object"])
        if object_meta is not None:
            selected["object"] = object_meta["path"]

    inferred_night = night if night else (object_meta["NIGHT"] if object_meta else None)
    inferred_shoe = shoe if shoe else (object_meta["SHOE"] if object_meta else None)

    for role in ("quartz", "thar", "twilight"):
        if role not in required_roles:
            continue
        role_candidates = by_role[role]
        if inferred_night is not None:
            role_candidates = [c for c in role_candidates if str(c["NIGHT"]) == str(inferred_night)]
        if inferred_shoe is not None:
            role_candidates = [c for c in role_candidates if str(c["SHOE"]).upper() == str(inferred_shoe).upper()]
        selected_meta = _select_unique_candidate(role, role_candidates)
        if selected_meta is not None:
            selected[role] = selected_meta["path"]

    print(
        f"  Auto-discovery scanned {len(fits_paths)} FITS files; "
        f"{skipped} lacked required metadata keys."
    )
    return selected


def resolve_inputs(args, required_roles=None):
    """Resolve role paths from auto-discovery with manual override support."""
    if required_roles is None:
        required_roles = ("quartz", "thar", "object", "twilight")
    else:
        required_roles = tuple(required_roles)

    resolved = {}
    manual = {
        "quartz": args.quartz,
        "thar": args.thar,
        "object": args.object,
        "twilight": args.twilight,
    }

    if all(manual[r] for r in required_roles):
        resolved.update({r: manual[r] for r in required_roles})
        return resolved

    if required_roles:
        discovered = discover_inputs(
            args.input_dir,
            night=args.night,
            shoe=args.shoe,
            object_name=args.object_name,
            required_roles=required_roles,
        )
        resolved.update(discovered)

    for role, path in manual.items():
        if path:
            resolved[role] = path

    missing = [r for r in required_roles if r not in resolved]
    if missing:
        raise RuntimeError(
            "Could not resolve all required inputs. Missing roles: "
            f"{', '.join(missing)}. Provide explicit flags or refine "
            "--input-dir/--night/--shoe/--object-name."
        )
    return resolved


def validate_input_set(meta_by_role, args):
    """Validate metadata consistency and gate mismatches by confirmation."""
    warnings = []
    nights = {meta_by_role[r]["NIGHT"] for r in meta_by_role}
    shoes = {meta_by_role[r]["SHOE"] for r in meta_by_role}
    if len(nights) != 1:
        warnings.append(f"Mixed NIGHT values: {sorted(nights)}")
    if len(shoes) != 1:
        warnings.append(f"Mixed SHOE values: {sorted(shoes)}")

    expected = {
        "quartz": "quartz",
        "thar": "thar",
        "object": "object",
        "twilight": "twilight",
    }
    for role, meta in meta_by_role.items():
        observed = classify_role(meta)
        if role in expected and observed != expected[role]:
            warnings.append(
                f"Role mismatch for {role}: metadata looks like '{observed}' "
                f"(EXPTYPE={meta['EXPTYPE']}, OBJECT={meta['OBJECT']})"
            )

    if not warnings:
        return

    section_banner("Input metadata warnings")
    for w in warnings:
        print(f"  [warn] {w}")

    if args.yes:
        print("  --yes set: continuing despite metadata warnings.")
        return

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Metadata mismatch detected in non-interactive mode. "
            "Re-run with --yes to continue anyway."
        )

    answer = input("\nProceed anyway? Type 'yes' to continue: ").strip().lower()
    if answer != "yes":
        raise RuntimeError("Aborted by user after metadata mismatch warning.")


def write_output_metadata(path, metadata):
    """Attach key provenance metadata to an output FITS file."""
    with fits.open(path, mode="update") as hdul:
        hdr = hdul[0].header
        for key, value in metadata.items():
            hdr[key] = value


# ---------------------------------------------------------------------------
# Step 1 – apall on quartz: find + trace, NO extract
# ---------------------------------------------------------------------------

def apall_trace_quartz(quartz, rdnoise, gain, n_ap):
    """
    Interactively find and trace all apertures on the quartz image.
    Nothing is extracted here; we only want the aperture database written.
    """
    section_banner("Step 1 – apall on quartz (find & trace, no extraction)")
    iraf.echelle.apall.unlearn()
    iraf.echelle.apall(
        input       = quartz,
        output      = "",           # no spectrum output
        apertures   = "",
        format      = "echelle",
        references  = "",
        profiles    = "",

        interactive = iraf.yes,
        find        = iraf.yes,
        recenter    = iraf.yes,
        resize      = iraf.yes,
        edit        = iraf.yes,
        trace       = iraf.yes,
        fittrace    = iraf.yes,
        extract     = iraf.no,      # trace only
        extras      = iraf.no,
        review      = iraf.no,

        line        = "INDEF",
        nsum        = 10,

        lower       = -3.5,
        upper       =  3.5,

        b_function  = "chebyshev",
        b_order     = 1,
        b_sample    = "-10:-6,6:10",
        b_naverage  = -3,
        b_niterate  = 5,
        b_low_rejec = 3.0,
        b_high_reje = 3.0,
        b_grow      = 0.0,

        width       = 7.0,
        radius      = 10.0,
        threshold   = 0.0,

        nfind       = n_ap,
        minsep      = 3.0,
        maxsep      = 100000.0,
        order       = "increasing",

        llimit      = -5.0,
        ulimit      =  5.0,
        ylevel      =  0.1,
        peak        = iraf.yes,
        bkg         = iraf.no,
        r_grow      = 0.0,
        avglimits   = iraf.yes,

        t_nsum      = 5,
        t_step      = 5,
        t_nlost     = 3,
        t_function  = "legendre",
        t_order     = 3,
        t_sample    = "*",
        t_naverage  = 1,
        t_niterate  = 0,
        t_low_rejec = 3.0,
        t_high_reje = 3.0,
        t_grow      = 0.0,

        background  = "none",
        skybox      = 1,
        weights     = "variance",
        pfit        = "fit1d",
        clean       = iraf.yes,
        saturation  = "INDEF",
        readnoise   = "ENOISE",
        gain        = "EGAIN",
        lsigma      = 4.0,
        usigma      = 4.0,
        nsubaps     = 1,
        mode        = "ql",
    )
    print(f"  Aperture database written for: {quartz}")


# ---------------------------------------------------------------------------
# Step 2 – apscatter on all four images
# ---------------------------------------------------------------------------

def _apscatter_one(image, output, reference):
    iraf.echelle.apscatter.unlearn()
    iraf.echelle.apscatter(
        input       = image,
        output      = output,
        apertures   = "",
        scatter     = "",
        references  = reference,    # quartz aperture database
        interactive = iraf.yes,
        find        = iraf.no,
        recenter    = iraf.no,
        resize      = iraf.no,
        edit        = iraf.no,
        trace       = iraf.no,
        fittrace    = iraf.no,
        subtract    = iraf.yes,
        smooth      = iraf.yes,
        fitscatter  = iraf.yes,
        fitsmooth   = iraf.yes,
        line        = "INDEF",
        nsum        = 100,
        buffer      = 1.0,
        apscat1     = "",
        apscat2     = "",
        mode        = "ql",
    )
    print(f"  apscatter done: {image} -> {output}")


def apscatter_all(quartz, thar, obj, twilight):
    section_banner("Step 2 – apscatter (scattered-light subtraction)")
    inputs = [quartz, thar, obj, twilight]
    outputs = [stem(img) + "-sl.fits" for img in inputs]

    for out in outputs:
        iraf_delete(out)

    for img, out in zip(inputs, outputs):
        print(f"\n  -- {img} -> {out}")
        _apscatter_one(img, out, reference=quartz)

    return tuple(outputs)


# ---------------------------------------------------------------------------
# Step 3 – normalised master flat
# ---------------------------------------------------------------------------

def make_normalised_flat(quartz, xwindow=10):
    """
    a. fmedian (xwindow=10, ywindow=1) on the scatter-corrected quartz
    b. imarith: quartz / median_flat  -> raw normalised flat
    c. imreplace: set pixels <= 0 and >= 5 to 1

    Returns the path of the master flat.
    """
    section_banner("Step 3 – Normalised master flat")
    q_stem      = stem(quartz)
    median_flat = q_stem + "_med.fits"
    master_flat = q_stem + "_nflat.fits"

    # 3a – fmedian
    iraf_delete(median_flat)
    print(f"  fmedian  {quartz}  ->  {median_flat}  (xwindow=19, ywindow=1)")
    iraf.images.imfilter.fmedian.unlearn()
    iraf.images.imfilter.fmedian(
        input     = quartz,
        output    = median_flat,
        xwindow   = xwindow,
        ywindow   = 1,
        zloreject = "INDEF",
        zhireject = "INDEF",
        boundary  = "reflect",
        constant  = 0.0,
        verbose   = iraf.yes,
        mode      = "ql",
    )

    # 3b – divide
    iraf_delete(master_flat)
    print(f"  imarith  {quartz} / {median_flat}  ->  {master_flat}")
    iraf.images.imutil.imarith(
        operand1 = quartz,
        op       = "/",
        operand2 = median_flat,
        result   = master_flat,
        title    = "",
        divzero  = 1.0,
        hparams  = "",
        pixtype  = "",
        calctype = "",
        verbose  = iraf.yes,
        mode     = "ql",
    )

    # 3c – clip bad values to 1
    print(f"  imreplace {master_flat}: values <= 0  -> 1")
    iraf.images.imutil.imreplace(
        images   = master_flat,
        value    = 1.0,
        imagina  = 0.0,
        lower    = "INDEF",
        upper    = 0.0,
        radius   = 0.0,
        mode     = "ql",
    )
    print(f"  imreplace {master_flat}: values >= 5  -> 1")
    iraf.images.imutil.imreplace(
        images   = master_flat,
        value    = 1.0,
        imagina  = 0.0,
        lower    = 5.0,
        upper    = "INDEF",
        radius   = 0.0,
        mode     = "ql",
    )
    print(f"  Master flat ready: {master_flat}")
    return master_flat


# ---------------------------------------------------------------------------
# Step 4 – ccdproc flat-field correction
# ---------------------------------------------------------------------------

def flatcorrect_images(thar, obj, twilight, master_flat):
    section_banner("Step 4 – Flat-field correction (ccdproc)")

    targets  = [thar, obj, twilight]
    outputs  = [stem(t) + "-F.fits" for t in targets]

    in_list  = "_flatcorr_in.list"
    out_list = "_flatcorr_out.list"
    write_list(in_list,  targets)
    write_list(out_list, outputs)

    for out in outputs:
        iraf_delete(out)

    iraf.ccdred.ccdproc.unlearn()
    iraf.ccdred.ccdproc(
        images      = "@" + in_list,
        output      = "@" + out_list,
        ccdtype     = "",
        max_cache   = 0,
        noproc      = iraf.no,
        fixpix      = iraf.no,
        overscan    = iraf.no,
        trim        = iraf.no,
        zerocor     = iraf.no,
        darkcor     = iraf.no,
        flatcor     = iraf.yes,
        illumcor    = iraf.no,
        fringecor   = iraf.no,
        readcor     = iraf.no,
        scancor     = iraf.no,
        readaxis    = "line",
        fixfile     = "",
        biassec     = "image",
        trimsec     = "image",
        zero        = "",
        dark        = "",
        flat        = master_flat,
        illum       = "",
        fringe      = "",
        minreplace  = 1.0,
        scantype    = "shortscan",
        nscan       = 1,
        interactive = iraf.no,
        function    = "chebyshev",
        order       = 3,
        sample      = "*",
        naverage    = 1,
        niterate    = 1,
        low_reject  = 3.0,
        high_reject = 3.0,
        grow        = 0.0,
        mode        = "ql",
    )

    thar_ff, obj_ff, twi_ff = outputs
    for out in outputs:
        print(f"  flat-corrected: {out}")
    return thar_ff, obj_ff, twi_ff


# ---------------------------------------------------------------------------
# Step 5 – interactive aperture-trace preview (just before extraction)
# ---------------------------------------------------------------------------

def run_aperture_preview(preview_image, n_stars, sep):
    """
    Open the interactive matplotlib preview on a reference image for assigning
    aperture traces to stars. This does not require the final extraction image;
    the accepted star-assignment pattern is reused in step 6.

    Returns (centers, pattern) once the user presses q.
    """
    section_banner("Step 5 – Interactive aperture-trace preview")

    try:
        from aperture_preview import run_preview
    except ImportError:
        print(
            "  [warn] aperture_preview.py not importable.\n"
            "         Falling back to auto-generated pattern (no interactive preview)."
        )
        return _fallback_pattern(preview_image, n_stars, sep)

    print(
        f"  Image: {preview_image}\n\n"
        "  Controls inside the window:\n"
        "    e   hover over a trace then press e to reassign its star\n"
        "    d   press d to mark a trace as deleted (will not be extracted)\n"
        "    f   propagate the corrected pattern forward from that aperture\n"
        "    r   reset all assignments to the auto-generated pattern\n"
        "    q   accept and continue  <-- this moves the pipeline forward\n"
        "    h   print this reminder\n"
    )
    centers, pattern = run_preview(preview_image, n_stars=n_stars, sep=sep)

    # Print a summary for the terminal log
    unique_stars = sorted(np.unique(pattern))
    # Separate deleted (0) from valid stars
    deleted_count = len(np.where(pattern == 0)[0])
    valid_stars = [s for s in unique_stars if s != 0]
    
    active_count = len(pattern) - deleted_count
    print(f"\n  Accepted mapping  ({active_count} active apertures, {len(valid_stars)} stars):")
    for s in valid_stars:
        ap_indices = list(np.where(pattern == s)[0] + 1)   # 1-based
        preview    = ", ".join(str(a) for a in ap_indices[:12])
        suffix     = " ..." if len(ap_indices) > 12 else ""
        print(f"    Star {s:>2d}  ->  {len(ap_indices):>3} apertures  [{preview}{suffix}]")
    
    if deleted_count > 0:
        deleted_aps = list(np.where(pattern == 0)[0] + 1)
        preview    = ", ".join(str(a) for a in deleted_aps[:12])
        suffix     = " ..." if len(deleted_aps) > 12 else ""
        print(f"    Deleted  ->  {deleted_count:>3} apertures  [{preview}{suffix}]")

    return centers, pattern


def _fallback_pattern(image_path, n_stars, sep):
    """Auto-generate centers + pattern without any interactive window."""
    from astropy.io import fits as _fits
    with _fits.open(image_path) as hdul:
        data = hdul[0].data.astype(float)
    nrows = data.shape[0]

    try:
        from aperture_preview import find_aperture_centers
        centers = find_aperture_centers(data, col=data.shape[1] // 2,
                                        expected_sep=sep)
    except Exception:
        approx_n = int(nrows / max(1.0, sep))
        centers  = np.linspace(sep, nrows - sep, approx_n)

    # Keep non-interactive fallback identical to aperture_preview default mapping.
    try:
        from aperture_preview import default_pattern as _default_pattern
        pattern = _default_pattern(len(centers), n_stars=n_stars)
    except Exception:
        pattern = (np.arange(len(centers), dtype=int) // 4) + 1
    return centers, np.array(pattern, dtype=int)


# ---------------------------------------------------------------------------
# Step 6 – per-star apall extraction (object + thar)
# ---------------------------------------------------------------------------

def _aperture_range_string(ap_indices_1based):
    """
    Convert a list of 1-based aperture indices into a compact IRAF range
    string, e.g. [1,2,3,5,6,10] -> '1-3,5-6,10'.
    """
    if not ap_indices_1based:
        return ""
    indices = sorted(ap_indices_1based)
    parts   = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = end = idx
    parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def _apall_extract_star(image, reference_quartz, aperture_string,
                        star_number):
    """
    Extract one star's apertures from *image* using the trace database from
    *reference_quartz*.

    Parameters
    ----------
    image            : flat-corrected input image path
    reference_quartz : quartz image whose apall database holds all traces
    aperture_string  : IRAF aperture range string (e.g. '1-4,9-12')
    star_number      : integer label used to build the output filename

    Returns
    -------
    Path of the extracted multispec FITS file.
    """
    out_stem = f"{stem(image)}_star{star_number:02d}_ec"
    out_spec = out_stem + ".fits"
    iraf_delete(out_spec)

    print(f"      apall  apertures={aperture_string!r}  ->  {out_spec}")

    iraf.echelle.apall.unlearn()
    iraf.echelle.apall(
        input       = image,
        output      = out_stem,          # IRAF appends .fits
        apertures   = aperture_string,   # only this star's apertures
        format      = "echelle",
        references  = reference_quartz,  # all traces live in the quartz database
        profiles    = "",

        interactive = iraf.no,   # extraction only; traces already defined
        find        = iraf.no,
        recenter    = iraf.no,
        resize      = iraf.no,
        edit        = iraf.no,
        trace       = iraf.no,
        fittrace    = iraf.no,
        extract     = iraf.yes,
        extras      = iraf.yes,
        review      = iraf.yes,

        line        = "INDEF",
        nsum        = 10,

        lower       = -3,
        upper       =  3,

        b_function  = "chebyshev",
        b_order     = 1,
        b_sample    = "-10:-6,6:10",
        b_naverage  = -3,
        b_niterate  = 0,
        b_low_rejec = 3.0,
        b_high_reje = 3.0,
        b_grow      = 0.0,

        width       = 6.0,
        radius      = 10.0,
        threshold   = 0.0,

        nfind       = 0,             # irrelevant: find=no
        minsep      = 3.0,
        maxsep      = 100000.0,
        order       = "increasing",

        llimit      = -5.0,
        ulimit      =  5.0,
        ylevel      =  0.1,
        peak        = iraf.yes,
        bkg         = iraf.no,
        r_grow      = 0.0,
        avglimits   = iraf.yes,

        t_nsum      = 5,
        t_step      = 5,
        t_nlost     = 3,
        t_function  = "legendre",
        t_order     = 3,
        t_sample    = "*",
        t_naverage  = 1,
        t_niterate  = 0,
        t_low_rejec = 3.0,
        t_high_reje = 3.0,
        t_grow      = 0.0,

        background  = "none",
        skybox      = 1,
        weights     = "variance",
        pfit        = "fit1d",
        clean       = iraf.yes,
        saturation  = "INDEF",
        readnoise   = "ENOISE",
        gain        = "EGAIN",
        lsigma      = 4.0,
        usigma      = 4.0,
        nsubaps     = 1,
        mode        = "ql",
    )
    return out_spec


def _lineclean_one(input_spec, output_spec):
        """
        Run lineclean on a single extracted multispec file.

        Per the McDonald reduction notes:
            function  = spline3
            order     = 6
            low_rej   = 50   (very asymmetric: almost never rejects real flux)
            high_rej  = 3    (aggressive upward rejection to catch CRs)
            niterate  = 10
            interactive = no   (works very well non-interactively)

        lineclean operates along the dispersion axis (axis=1 for echelle multispec
        format) and fits each order independently.
        """
        iraf_delete(output_spec)
        try:
            iraf.onedspec.lineclean.unlearn()
        except AttributeError:
            print(f"      [warn] lineclean task not found in IRAF; skipping CR removal for {input_spec}")
            # Fall back to copying file without CR removal
            import shutil
            shutil.copy(input_spec, output_spec)
            return
    
        iraf.onedspec.lineclean(
                input     = input_spec,
                output    = output_spec,
                apertures = "",          # all apertures in the file
                crval     = "INDEF",
                cdelt     = "INDEF",
                function  = "spline3",
                order     = 6,
                low_rejec = 50.0,
                high_reje = 3.0,
                niterate  = 10,
                interacti = iraf.no,
                mode      = "ql",
        )
        print(f"      lineclean: {input_spec}  ->  {output_spec}")


def extract_all_stars(obj_ff, thar_ff, quartz, pattern):
    """
    For every unique star in *pattern*:
      1. Collect that star's 1-based aperture indices and build an IRAF range
         string.
      2. Run apall on the flat-corrected object  -> <obj_stem>_star<N>_ec.fits
      3. Run apall on the flat-corrected thar    -> <thar_stem>_star<N>_ec.fits
         using exactly the same aperture selection.

    Object and thar are always extracted with the same aperture set so that
    wavelength calibration can later be applied aperture-for-aperture.

    Returns
    -------
    obj_outputs  : {star_number: path, ...}
    thar_outputs : {star_number: path, ...}
    """
    section_banner("Step 6 – Per-star apall extraction (object + ThAr)")

    pattern      = np.asarray(pattern, dtype=int)
    unique_stars = sorted(np.unique(pattern))
    # Skip apertures marked as deleted (pattern == 0)
    unique_stars = [s for s in unique_stars if s != 0]
    obj_outputs  = {}
    thar_outputs = {}

    # Load IRAF packages for extraction.
    iraf.onedspec()
    iraf.noao()
    iraf.imred()
    iraf.echelle()

    for star in unique_stars:
        # All 1-based aperture indices that belong to this star
        ap_indices   = list(np.where(pattern == star)[0] + 1)
        aperture_str = _aperture_range_string(ap_indices)

        print(f"\n  -- Star {star:02d}  |  {len(ap_indices)} apertures  "
              f"|  IRAF range: {aperture_str}")

        print(f"    Extracting object ...")
        obj_ec = _apall_extract_star(
            obj_ff, quartz, aperture_str,
            star_number = star,
        )
        obj_outputs[star] = obj_ec

        print(f"    Extracting ThAr ...")
        thar_ec = _apall_extract_star(
            thar_ff, quartz, aperture_str,
            star_number = star,
        )
        thar_outputs[star] = thar_ec

    return obj_outputs, thar_outputs


# ---------------------------------------------------------------------------
# Step 7 – second cosmic-ray removal  (imfit/lineclean on extracted spectra)
# ---------------------------------------------------------------------------


def second_cosmic_removal(obj_outputs):
    """
    Apply lineclean to every per-star extracted object spectrum.

    The ThAr frames are left untouched – CR cleaning is only needed for the
    science spectra (arc lamps have no continuum for the spline to follow).

    Parameters
    ----------
    obj_outputs : {star_number: path, ...}
        Dict returned by extract_all_stars (step 6).

    Returns
    -------
    crr2_outputs : {star_number: path, ...}
        Same keys, new paths with suffix -crr2.fits.
    """
    section_banner("Step 7 – Second cosmic-ray removal (lineclean)")
    iraf.noao()
    iraf.imred()
    iraf.echelle()

    crr2_outputs = {}
    for star in sorted(obj_outputs):
        in_spec  = obj_outputs[star]
        out_spec = stem(in_spec) + "-crr2.fits"
        print(f"\n  -- Star {star:02d}  |  {in_spec}")
        _lineclean_one(in_spec, out_spec)
        crr2_outputs[star] = out_spec

    return crr2_outputs


def expected_step7_outputs(obj_outputs):
    """Return deterministic step-7 output paths."""
    return {star: stem(path) + "-crr2.fits" for star, path in obj_outputs.items()}


# ---------------------------------------------------------------------------
# Step 8 – wavelength scale  (ecidentify → refspec → dispcor)
# ---------------------------------------------------------------------------

def _ecidentify_thar(thar_ec, coordlist="linelists$thar.dat"):
    """
    Run ecidentify interactively on a ThAr multispec file to build the
    wavelength solution.

    Per the McDonald notes:
      - First run: maxfeatures=100, identify ~5 lines per order manually,
        fit with legendre xorder=4, yorder=4, then press 'l' to load more
        lines from the line list.
      - Second run (same call after 'q'+'f'+'l'): maxfeatures=1000,
        IRAF remembers the previously identified lines; press 'f' to refit,
        load more, inspect, quit and save.
    We set maxfeatures=100 here; the user is expected to do the
    high-maxfeature pass by re-running the task in the IRAF terminal if
    needed, or by calling this function a second time with maxfeatures=1000.
    """
    print(f"  ecidentify (interactive): {thar_ec}")
    iraf.noao.echelle.ecidentify.unlearn()
    iraf.noao.echelle.ecidentify(
        images    = thar_ec,
        database  = "database",
        coordlist = coordlist,
        units     = "",
        match     = 1.0,
        maxfeatur = 100,          # user increases to 1000 in the second pass
        zwidth    = 10.0,
        ftype     = "emission",
        fwidth    = 4.0,
        cradius   = 5.0,
        threshold = 10.0,
        minsep    = 2.0,
        function  = "legendre",
        xorder    = 4,
        yorder    = 4,
        niterate  = 5,
        lowreject = 3.0,
        highrejec = 3.0,
        autowrit  = iraf.no,
        graphic   = "stdgraph",
        cursor    = "",
        mode      = "ql",
    )
    print(f"  ecidentify done: {thar_ec}")


def _ecreidentify_thar(thar_ec, ref_thar_ec, coordlist="linelists$thar.dat"):
    """
    Run ecreidentify (automatic) on a ThAr multispec using a reference solution.

    Propagates the wavelength solution from ref_thar_ec to thar_ec based on
    spatial correlation of arc lines. This is much faster than manual identification.

    Parameters
    ----------
    thar_ec       : path to ThAr spectrum to be identified (uses reference)
    ref_thar_ec   : path to reference ThAr spectrum with existing solution
    coordlist      : IRAF line-list path (default: built-in ThAr list)
    """
    print(f"  ecreidentify (automatic): {thar_ec}  <--ref--  {ref_thar_ec}")
    iraf.noao.echelle.ecreidentify.unlearn()
    iraf.noao.echelle.ecreidentify(
        images     = thar_ec,
        reference  = ref_thar_ec,
        interactive = iraf.no,
        find       = iraf.no,
        recenter   = iraf.yes,
        database   = "database",
        coordlist  = coordlist,
        units      = "",
        match      = 1.0,
        maxfeatur  = 100,
        zwidth     = 10.0,
        ftype      = "emission",
        fwidth     = 4.0,
        cradius    = 5.0,
        threshold  = 10.0,
        minsep     = 2.0,
        function   = "legendre",
        xorder     = 4,
        yorder     = 4,
        niterate   = 5,
        lowreject  = 3.0,
        highrejec  = 3.0,
        refit      = iraf.yes,
        newaps     = iraf.no,
        override   = iraf.no,
        mode       = "ql",
    )
    print(f"  ecreidentify done: {thar_ec}")


def _refspec_one(obj_ec, thar_ec):
    """
    Assign the ThAr wavelength solution from *thar_ec* to the object
    spectrum *obj_ec* using refspec.

    select=match means IRAF pairs apertures by number (1→1, 2→2, …), which
    is correct because object and ThAr were extracted from the same aperture
    set in step 6.
    """
    print(f"  refspec: {obj_ec}  <--  {thar_ec}")
    iraf.noao.onedspec.refspec.unlearn()
    iraf.noao.onedspec.refspec(
        input    = obj_ec,
        referenc = thar_ec,
        aperture = "",
        refaps   = "",
        ignoreap = iraf.no,
        select   = "match",
        sort     = "",
        group    = "",
        time     = iraf.no,
        timewrap = 17.0,
        override = iraf.no,
        confirm  = iraf.no,
        assign   = iraf.yes,
        logfile  = "STDOUT,logfile",
        verbose  = iraf.no,
        mode     = "ql",
    )


def _dispcor_one(obj_ec, output_spec):
    """
    Apply (implant) the wavelength solution assigned by refspec onto *obj_ec*,
    writing the linearised spectrum to *output_spec*.

    flux=yes conserves total flux when resampling to a linear grid.
    """
    iraf_delete(output_spec)
    print(f"  dispcor: {obj_ec}  ->  {output_spec}")
    iraf.noao.onedspec.dispcor.unlearn()
    iraf.noao.onedspec.dispcor(
        input    = obj_ec,
        output   = output_spec,
        lineariz = iraf.yes,
        database = "database",
        table    = "",
        w1       = "INDEF",
        w2       = "INDEF",
        dw       = "INDEF",
        nw       = "INDEF",
        log      = iraf.no,
        flux     = iraf.yes,
        blank    = 0.0,
        samedisp = iraf.no,
        ignoreap = iraf.no,
        confirm  = iraf.no,
        listonl  = iraf.no,
        verbose  = iraf.yes,
        logfile  = "",
        mode     = "ql",
    )


def manual_wavelength_identification(crr2_outputs, thar_outputs,
                                     coordlist="linelists$thar.dat"):
    """
    Step 8: Manual wavelength identification on reference star only.

    Runs ecidentify (interactive) on one reference star's ThAr spectrum,
    then refspec to assign that solution to the corresponding object spectrum.
    Subsequent stars' solutions can be propagated in step 9 via ecreidentify.

    The reference star is the first star (minimum) in the star list.

    Parameters
    ----------
    crr2_outputs : {star_number: path, ...}
        CR-cleaned object spectra from step 7.
    thar_outputs : {star_number: path, ...}
        ThAr spectra from step 6 (same aperture selection as objects).
    coordlist    : str
        IRAF line-list path (default: built-in ThAr list).

    Returns
    -------
    ref_star : int
        The star number used as reference (minimum in sorted list).
    """
    section_banner("Step 8 – Manual wavelength identification (ecidentify on reference star)")
    iraf.noao()
    iraf.echelle()
    iraf.onedspec()

    # Use first star (minimum) as reference
    ref_star = min(crr2_outputs.keys())
    obj_crr2_ref = crr2_outputs[ref_star]
    thar_ec_ref  = thar_outputs[ref_star]

    print(f"\n  Reference star: {ref_star:02d}")
    print(f"    object : {obj_crr2_ref}")
    print(f"    thar   : {thar_ec_ref}")

    # 8a – ecidentify on reference ThAr (interactive)
    _ecidentify_thar(thar_ec_ref, coordlist=coordlist)

    # 8b – refspec: attach ThAr solution to the reference object
    _refspec_one(obj_crr2_ref, thar_ec_ref)

    return ref_star


def auto_wavelength_propagation(crr2_outputs, thar_outputs, ref_star,
                                coordlist="linelists$thar.dat"):
    """
    Step 9: Automatic wavelength propagation to remaining stars.

    Uses ecreidentify to propagate the solution from ref_star to all other
    stars, then refspec + dispcor to complete the wavelength calibration.

    Parameters
    ----------
    crr2_outputs : {star_number: path, ...}
        CR-cleaned object spectra from step 7.
    thar_outputs : {star_number: path, ...}
        ThAr spectra from step 6 (same aperture selection as objects).
    ref_star     : int
        Reference star number (should have been manually identified in step 8).
    coordlist    : str
        IRAF line-list path (default: built-in ThAr list).

    Returns
    -------
    wlcal_outputs : {star_number: path, ...}
        Wavelength-calibrated final spectra, suffix -wl.fits.
    """
    section_banner("Step 9 – Automatic wavelength propagation (ecreidentify + refspec + dispcor)")
    iraf.noao()
    iraf.echelle()
    iraf.onedspec()

    thar_ec_ref = thar_outputs[ref_star]
    wlcal_outputs = {}

    for star in sorted(crr2_outputs.keys()):
        obj_crr2 = crr2_outputs[star]
        thar_ec  = thar_outputs[star]
        out_wl   = stem(obj_crr2) + "-wl.fits"

        print(f"\n  -- Star {star:02d}")
        print(f"     object : {obj_crr2}")
        print(f"     thar   : {thar_ec}")
        print(f"     output : {out_wl}")

        # For reference star, use existing solution (already identified in step 8)
        if star == ref_star:
            print(f"     (reference star — solution from step 8)")
        else:
            # 9a – ecreidentify: propagate solution from reference star
            _ecreidentify_thar(thar_ec, thar_ec_ref, coordlist=coordlist)

        # 9b – refspec: attach ThAr solution to the CR-cleaned object
        _refspec_one(obj_crr2, thar_ec)

        # 9c – dispcor: implant the solution and linearise
        _dispcor_one(obj_crr2, out_wl)

        wlcal_outputs[star] = out_wl

    return wlcal_outputs


def expected_step8_outputs(crr2_outputs):
    """Return deterministic step-8 reference star output paths (just refspec, no dispcor)."""
    ref_star = min(crr2_outputs.keys())
    return {ref_star: stem(crr2_outputs[ref_star]) + "-wl.fits"}


def expected_step9_outputs(crr2_outputs):
    """Return deterministic step-9 output paths (all stars with dispcor)."""
    return {star: stem(path) + "-wl.fits" for star, path in crr2_outputs.items()}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            McDonald echelle reduction pipeline (post-notebook).

            Steps:
              1. apall trace on quartz  (find+trace, no extract)
              2. apscatter on quartz, thar, object, twilight
              3. Normalised master flat  (fmedian -> imarith -> imreplace)
              4. ccdproc flat-correction  ->  *-sl-F.fits
              5. Interactive aperture-trace preview on traced quartz reference
              6. Per-star apall extraction  ->  *_star<N>_ec.fits
              7. Second CR removal (lineclean)  ->  *_ec-crr2.fits
              8. Manual wavelength identification (ecidentify on reference star)
              9. Automatic wavelength propagation (ecreidentify + refspec + dispcor)  ->  *-wl.fits

                        Modular execution:
                            Use --start-step/--end-step to run a contiguous step range.
                            Strict mode: skipped prerequisite steps are not auto-run; their
                            expected output files must already exist on disk.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quartz",   help="Quartz mosaic FITS (manual override).")
    p.add_argument("--thar",     help="ThArThNe arc mosaic FITS (manual override).")
    p.add_argument("--object",   help="Stacked science mosaic FITS (manual override).")
    p.add_argument("--twilight", help="Twilight sky mosaic FITS (manual override).")
    p.add_argument("--input-dir", default=".",
                   help="Directory to scan for auto-discovery (default: .).")
    p.add_argument("--night", default=None,
                   help="Restrict auto-discovery to this NIGHT value.")
    p.add_argument("--shoe", default=None, choices=["B", "R", "b", "r"],
                   help="Restrict auto-discovery to this SHOE value.")
    p.add_argument("--object-name", default=None,
                   help="Substring filter applied to auto-discovered science OBJECT.")
    p.add_argument("--yes", action="store_true",
                   help="Continue despite metadata mismatch warnings.")
    p.add_argument("--rdnoise",  type=float, default=3.06,
                   help="Read noise in e-  (default: 3.06).")
    p.add_argument("--gain",     type=float, default=0.584,
                   help="CCD gain in e-/ADU  (default: 0.584).")
    p.add_argument("--nstars",   type=int,   default=4,
                   help="Number of unique stars in the pattern (default: 4).")
    p.add_argument("--sep",      type=float, default=5.0,
                   help="Approx. spatial separation between apertures in px (default: 5).")
    p.add_argument("--nap",      type=int,   default=None,
                   help="Total apertures expected (default: IRAF auto-detect).")
    p.add_argument("--dispaxis", type=int,   default=1, choices=[1, 2],
                   help="Dispersion axis: 1=columns, 2=rows (default: 1).")
    p.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help="First pipeline step to execute (default: 1).")
    p.add_argument("--end-step", type=int, default=9, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help="Last pipeline step to execute (default: 9).")
    p.add_argument("--use-auto-pattern", action="store_true",
                   help="Allow step 6 without step 5 by using fallback auto pattern.")
    p.add_argument("--affiliation-map", default=None,
                   help="Path to saved Step-5 affiliation map JSON.")
    p.add_argument("--coordlist", default="linelists$thar.dat",
                   help="IRAF line list for ecidentify steps 8-9 (default: linelists$thar.dat).")
    p.add_argument("--no-cr", action="store_true",
                   help="Skip step-7 CR removal (keep step-6 *_ec.fits as-is).")
    return p.parse_args()


def selected_steps_from_args(args):
    """Return the selected contiguous step list from CLI args."""
    if args.start_step > args.end_step:
        raise RuntimeError(
            f"Invalid step range: start-step ({args.start_step}) is greater "
            f"than end-step ({args.end_step})."
        )
    return list(range(args.start_step, args.end_step + 1))


def required_roles_for_steps(selected_steps):
    """Return base input roles required for the selected step range.
    
    Steps 5-9 operate on outputs of earlier steps, so they don't require
    base role discovery:
      - Step 5: requires step 2 output (quartz_sl), checked separately
      - Step 6: requires step 4 outputs (obj_ff, thar_ff), checked separately
      - Steps 7-9: require outputs from step 6 or earlier
    """
    steps = set(selected_steps)
    roles = set()

    if 1 in steps or 3 in steps:
        roles.add("quartz")
    if 2 in steps:
        roles.update(("quartz", "thar", "object", "twilight"))
    if 4 in steps:
        roles.update(("quartz", "thar", "object", "twilight"))

    return tuple(r for r in ("quartz", "thar", "object", "twilight") if r in roles)


def expected_step2_outputs(quartz, thar, obj, twilight):
    """Return deterministic step-2 output paths for all roles."""
    return {
        "quartz_sl": stem(quartz) + "-sl.fits",
        "thar_sl": stem(thar) + "-sl.fits",
        "obj_sl": stem(obj) + "-sl.fits",
        "twilight_sl": stem(twilight) + "-sl.fits",
    }


def expected_master_flat(quartz_sl):
    """Return deterministic step-3 master-flat path."""
    return stem(quartz_sl) + "_nflat.fits"


def expected_step4_outputs(thar_sl, obj_sl, twilight_sl=None):
    """Return deterministic step-4 output paths for all roles."""
    outputs = {
        "thar_ff": stem(thar_sl) + "-F.fits",
        "obj_ff": stem(obj_sl) + "-F.fits",
    }
    if twilight_sl:
        outputs["twi_ff"] = stem(twilight_sl) + "-F.fits"
    return outputs


def require_existing(path, requirement):
    """Fail with a clear message if a required file is missing."""
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required file for {requirement}: {path}")


def quartz_trace_db_candidates(quartz):
    """Return likely IRAF aperture database paths for a quartz reference."""
    from pathlib import Path
    if quartz is None:
        return []  # No candidates if no quartz provided
    base = stem(quartz)
    db_dir = Path("database")
    candidates = [
        str(db_dir / f"ap._{base}"),  # Primary: IRAF's standard convention
        str(db_dir / f"ap.{base}"),   # Alternative without underscore
        str(db_dir / f"ap{base}"),    # Alternative without dot
        base + ".db",                 # Legacy fallback
    ]
    return candidates


def require_quartz_trace_db(quartz, requirement):
    """Ensure step prerequisites include an aperture trace database."""
    if quartz is None:
        raise RuntimeError(
            f"Missing quartz reference file for {requirement}. "
            "Provide --quartz <quartz_file> or run steps 1-4 first."
        )
    candidates = quartz_trace_db_candidates(quartz)
    if any(os.path.exists(p) for p in candidates):
        return
    raise RuntimeError(
        f"Missing quartz aperture trace database for {requirement}. "
        f"Looked for: {', '.join(candidates)}. "
        "Run step 1 first for this quartz reference."
    )


def infer_night_shoe(meta_by_role, reference_path=None, fallback_night=None, fallback_shoe=None):
    """Infer NIGHT/SHOE from metadata, with safe fallbacks for late-step runs."""
    if meta_by_role:
        first = next(iter(meta_by_role.values()))
        return str(first["NIGHT"]), str(first["SHOE"])

    if reference_path:
        meta = read_required_metadata(reference_path)
        return str(meta["NIGHT"]), str(meta["SHOE"])

    if fallback_night is not None and fallback_shoe is not None:
        return str(fallback_night), str(fallback_shoe)

    raise RuntimeError(
        "Cannot infer NIGHT/SHOE. Provide --night/--shoe or inputs with metadata."
    )


def default_affiliation_map_path(meta_by_role, reference_path=None,
                                 fallback_night=None, fallback_shoe=None):
    """Return default affiliation-map filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    return f"affiliation_{night}_{shoe}.json"


def save_affiliation_map(path, pattern, quartz_path, meta_by_role):
    """Persist accepted aperture affiliation map from Step 5."""
    night, shoe = infer_night_shoe(meta_by_role, reference_path=quartz_path)
    quartz_db = ""
    for candidate in quartz_trace_db_candidates(quartz_path):
        if os.path.exists(candidate):
            quartz_db = candidate
            break

    payload = {
        "schema_version": 1,
        "night": night,
        "shoe": shoe,
        "quartz_path": os.path.basename(quartz_path),
        "quartz_db_path": quartz_db,
        "n_apertures": int(len(pattern)),
        "pattern": [int(x) for x in np.asarray(pattern, dtype=int).tolist()],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  Affiliation map saved: {path}")


def load_affiliation_map(path):
    """Load persisted Step-5 affiliation map."""
    with open(path, "r") as fh:
        return json.load(fh)


def validate_affiliation_map(mapping, quartz_path, meta_by_role):
    """Validate persisted mapping against current run metadata."""
    required = {"schema_version", "night", "shoe", "quartz_path", "n_apertures", "pattern"}
    missing = sorted(required.difference(mapping.keys()))
    if missing:
        raise RuntimeError(f"Affiliation map missing keys: {missing}")

    if int(mapping["schema_version"]) != 1:
        raise RuntimeError(f"Unsupported affiliation map schema_version={mapping['schema_version']}")

    night, shoe = infer_night_shoe(meta_by_role, reference_path=quartz_path)
    issues = []
    if str(mapping["night"]) != str(night):
        issues.append(f"night mismatch: map={mapping['night']} run={night}")
    if str(mapping["shoe"]).upper() != str(shoe).upper():
        issues.append(f"shoe mismatch: map={mapping['shoe']} run={shoe}")
    if not quartz_path:
        issues.append("missing quartz reference path for map validation")
    elif os.path.basename(str(mapping["quartz_path"])) != os.path.basename(quartz_path):
        issues.append(
            "quartz mismatch: "
            f"map={mapping['quartz_path']} run={os.path.basename(quartz_path)}"
        )

    raw_pattern = np.asarray(mapping["pattern"])
    if raw_pattern.ndim != 1:
        issues.append("pattern must be a 1D list of aperture assignments")
        raw_pattern = raw_pattern.reshape(-1)

    pattern = None
    try:
        if not np.all(np.equal(raw_pattern, np.round(raw_pattern))):
            issues.append("pattern contains non-integer values")
        pattern = raw_pattern.astype(int)
    except Exception:
        issues.append("pattern contains non-numeric values")
        pattern = np.array([], dtype=int)
    if int(mapping["n_apertures"]) != int(len(pattern)):
        issues.append(
            f"n_apertures mismatch: map={mapping['n_apertures']} pattern_len={len(pattern)}"
        )
    if np.any(pattern < 0):
        issues.append("pattern contains values < 0 (allowed: 0=deleted, 1..N=stars)")

    if issues:
        raise RuntimeError("Invalid affiliation map:\n  - " + "\n  - ".join(issues))
    return pattern


def write_extraction_pairs_index(path, obj_outputs, thar_outputs, pattern):
    """Write star-to-output pairing summary for object and atlas extractions."""
    pattern = np.asarray(pattern, dtype=int)
    stars = sorted(set(obj_outputs).intersection(set(thar_outputs)))
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["star", "object_file", "atlas_file", "apertures"])
        for star in stars:
            ap_indices = list(np.where(pattern == star)[0] + 1)
            aperture_str = _aperture_range_string(ap_indices)
            writer.writerow([star, obj_outputs[star], thar_outputs[star], aperture_str])
    print(f"  Extraction pairing index saved: {path}")


def _extract_star_number(path):
    """Extract star number from a per-star extracted filename."""
    m = re.search(r"_star(\d+)_ec(?:-crr2)?(?:-wl)?\.fits$", os.path.basename(path), re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _resolve_existing_path(path_value, search_dirs):
    """Resolve relative file paths against likely base directories."""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value

    candidates = [path_value]
    for base in search_dirs:
        if not base:
            continue
        candidates.append(os.path.join(base, path_value))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path_value


def _prefer_crr2(path):
    """Prefer the CR-cleaned counterpart when available."""
    if not path:
        return path
    if path.endswith("_ec-crr2.fits"):
        return path
    if path.endswith("_ec.fits"):
        crr2 = stem(path) + "-crr2.fits"
        if os.path.exists(crr2):
            return crr2
    return path


def reconstruct_star_outputs_from_disk(args):
    """Reconstruct per-star object/ThAr outputs for standalone late-step runs."""
    search_dirs = [args.input_dir, os.getcwd()]
    pair_index_candidates = []
    if args.night and args.shoe:
        shoe = str(args.shoe).upper()
        pair_name = f"extraction_pairs_{args.night}_{shoe}.csv"
        pair_index_candidates.extend([
            pair_name,
            os.path.join(args.input_dir, pair_name),
        ])

    for pair_index in pair_index_candidates:
        if not os.path.exists(pair_index):
            continue

        obj_outputs = {}
        thar_outputs = {}
        pair_dir = os.path.dirname(pair_index) or "."
        with open(pair_index, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            required_cols = {"star", "object_file", "atlas_file"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise RuntimeError(
                    f"Invalid extraction index format in {pair_index}. "
                    "Expected columns: star, object_file, atlas_file."
                )

            for row in reader:
                star = int(row["star"])
                obj_path = _resolve_existing_path(row["object_file"], [pair_dir] + search_dirs)
                thar_path = _resolve_existing_path(row["atlas_file"], [pair_dir] + search_dirs)

                require_existing(obj_path, f"standalone late-step input star {star:02d} object")
                require_existing(thar_path, f"standalone late-step input star {star:02d} thar")

                obj_outputs[star] = _prefer_crr2(obj_path)
                thar_outputs[star] = thar_path

        if obj_outputs and thar_outputs:
            print(
                f"  Reconstructed {len(obj_outputs)} stars from extraction index: {pair_index}"
            )
            return obj_outputs, thar_outputs

    # Fallback: derive pairs directly from per-star extracted files.
    obj_candidates = {}
    thar_candidates = {}
    fits_paths = sorted(glob.glob(os.path.join(args.input_dir, "*_star*_ec*.fits")))
    for path in fits_paths:
        name = os.path.basename(path).lower()
        if name.endswith("-wl.fits"):
            continue

        star = _extract_star_number(path)
        if star is None:
            continue

        role = None
        try:
            role = classify_role(read_required_metadata(path))
        except Exception:
            role = None

        if role not in ("object", "thar"):
            if any(tag in name for tag in ("thar", "lamp", "arc")):
                role = "thar"
            else:
                role = "object"

        if role == "thar":
            if star not in thar_candidates:
                thar_candidates[star] = path
            continue

        rank = 2 if name.endswith("_ec-crr2.fits") else 1
        prev = obj_candidates.get(star)
        if prev is None or rank > prev[0]:
            obj_candidates[star] = (rank, path)

    common_stars = sorted(set(obj_candidates).intersection(set(thar_candidates)))
    if not common_stars:
        raise RuntimeError(
            "Could not reconstruct per-star extraction pairs for standalone late-step run. "
            "Expected extraction_pairs_<NIGHT>_<SHOE>.csv or matching *_starNN_ec*.fits pairs."
        )

    obj_outputs = {s: obj_candidates[s][1] for s in common_stars}
    thar_outputs = {s: thar_candidates[s] for s in common_stars}
    print(
        f"  Reconstructed {len(common_stars)} stars by scanning per-star extracted files in {args.input_dir}"
    )
    return obj_outputs, thar_outputs


def find_step2_outputs(input_dir, night, shoe):
    """
    Scan input_dir for step 2 outputs (quartz_sl, thar_sl, obj_sl, twilight_sl)
    or step 4 flat-corrected outputs (*-sl-F.fits).
    
    Used when running steps 5+ without base roles to find the flat-corrected intermediates.
    
    Returns
    -------
    dict with keys 'quartz_sl', 'thar_sl', 'obj_sl', 'twilight_sl' (or a subset if not all found).
    """
    import glob
    step2_outputs = {}
    
    # Search patterns: prefer step 4 flat-corrected files, then step 2 outputs.
    # Object is discovered from metadata role, not filename-only heuristics.
    patterns = {
        "quartz_sl": ["*quartz*-sl-F.fits", "*quartz*-sl.fits"],
        "thar_sl": ["*thar*-sl-F.fits", "*thar*-sl.fits"],
        "obj_sl": ["*-sl-F.fits", "*-sl.fits"],
        "twilight_sl": ["*twilight*-sl-F.fits", "*twilight*-sl.fits"],
    }

    def candidate_score(key, path, meta):
        name = os.path.basename(path).lower()
        score = 0
        if name.endswith("-sl-F.fits"):
            score += 100
        if meta is not None:
            score += processing_rank(meta)
        if key == "obj_sl" and "sstack" in name:
            score += 25
        return score

    def role_matches_key(key, path, meta):
        name = os.path.basename(path).lower()
        if key == "obj_sl":
            if meta is not None:
                return classify_role(meta) == "object"
            return not any(tag in name for tag in ("quartz", "thar", "arc", "lamp", "twilight"))
        if meta is not None:
            return classify_role(meta) == key.replace("_sl", "")
        return key.replace("_sl", "") in name
    
    for key, pattern_list in patterns.items():
        candidates = []
        for pattern in pattern_list:
            matches = glob.glob(os.path.join(input_dir, pattern))
            candidates.extend(matches)
        
        if candidates:
            # Filter by night/shoe if provided
            filtered = []
            for match in candidates:
                meta = None
                try:
                    meta = read_required_metadata(match)
                    match_night = meta.get("NIGHT")
                    match_shoe = meta.get("SHOE")
                    if (night is not None and str(match_night) != str(night)):
                        continue
                    if (shoe is not None and str(match_shoe).upper() != str(shoe).upper()):
                        continue
                except Exception:
                    meta = None

                if role_matches_key(key, match, meta):
                    filtered.append((match, meta))
            
            if filtered:
                filtered.sort(key=lambda item: candidate_score(key, item[0], item[1]), reverse=True)
                step2_outputs[key] = filtered[0][0]
    
    return step2_outputs

def main():
    args = parse_args()

    try:
        selected_steps = selected_steps_from_args(args)
        required_roles = required_roles_for_steps(selected_steps)
    except Exception as exc:
        sys.exit(f"ERROR parsing step range: {exc}")

    try:
        resolved = resolve_inputs(args, required_roles=required_roles)
    except Exception as exc:
        sys.exit(f"ERROR resolving inputs: {exc}")

    for role, path in resolved.items():
        if not os.path.exists(path):
            sys.exit(f"ERROR: --{role} file not found: {path}")

    meta_by_role = {}
    if resolved:
        try:
            meta_by_role = {role: read_required_metadata(path) for role, path in resolved.items()}
            validate_input_set(meta_by_role, args)
        except Exception as exc:
            sys.exit(f"ERROR validating input metadata: {exc}")

    quartz = resolved.get("quartz")
    thar = resolved.get("thar")
    obj = resolved.get("object")
    twilight = resolved.get("twilight")
    step2_expected = {}
    if quartz:
        step2_expected["quartz_sl"] = stem(quartz) + "-sl.fits"
    if thar:
        step2_expected["thar_sl"] = stem(thar) + "-sl.fits"
    if obj:
        step2_expected["obj_sl"] = stem(obj) + "-sl.fits"
    if twilight:
        step2_expected["twilight_sl"] = stem(twilight) + "-sl.fits"

    print("\n" + "="*72)
    print("  McDonald echelle reduction pipeline")
    print("="*72)
    for label, val in [
            ("quartz",   quartz if quartz else "<not required>"),
            ("thar",     thar if thar else "<not required>"),
            ("object",   obj if obj else "<not required>"),
            ("twilight", twilight if twilight else "<not required>"),
            ("rdnoise",  f"{args.rdnoise} e-"),
            ("gain",     f"{args.gain} e-/ADU"),
            ("n_stars",  args.nstars),
            ("sep",      f"{args.sep} px"),
            ("dispaxis", args.dispaxis),
            ("steps",    f"{args.start_step}..{args.end_step}"),
            ("aff_map",  args.affiliation_map if args.affiliation_map else "<auto>"),
    ]:
        print(f"  {label:<10}: {val}")
    print("\n  Metadata summary:")
    for role in ("quartz", "thar", "object", "twilight"):
        if role not in meta_by_role:
            continue
        meta = meta_by_role[role]
        print(
            f"    {role:<8} EXPTYPE={meta['EXPTYPE']:<14} "
            f"OBJECT={meta['OBJECT']:<24} NIGHT={meta['NIGHT']} SHOE={meta['SHOE']}"
        )
    print("="*72)

    load_packages()
    iraf.echelle.dispaxis = args.dispaxis

    state = {}
    obj_outputs = {}
    thar_outputs = {}
    
    # ── 1. Trace apertures on quartz ─────────────────────────────────────────
    if 1 in selected_steps:
        # If --nap is given use it, otherwise pass a large number so IRAF finds
        # all peaks it can; the user will refine interactively.
        n_ap = args.nap if args.nap else 120
        apall_trace_quartz(
            quartz,
            rdnoise = args.rdnoise,
            gain    = args.gain,
            n_ap    = n_ap,
        )
    elif any(s in selected_steps for s in (2, 6)):
        try:
            require_quartz_trace_db(quartz, "steps 2/6")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 2. apscatter on all four images ──────────────────────────────────────
    if 2 in selected_steps:
        quartz_sl, thar_sl, obj_sl, twilight_sl = apscatter_all(
            quartz, thar, obj, twilight,
        )
        state.update({
            "quartz_sl": quartz_sl,
            "thar_sl": thar_sl,
            "obj_sl": obj_sl,
            "twilight_sl": twilight_sl,
        })
    elif any(s in selected_steps for s in (3, 4, 5, 6)):
        try:
            required_step2_keys = set()
            if 3 in selected_steps or 4 in selected_steps:
                required_step2_keys.add("quartz_sl")
            if 4 in selected_steps:
                required_step2_keys.update(["thar_sl", "obj_sl", "twilight_sl"])
            elif 6 in selected_steps:
                required_step2_keys.update(["thar_sl", "obj_sl"])
            elif 5 in selected_steps:
                # Step 5 needs quartz_sl for the aperture preview
                required_step2_keys.add("quartz_sl")

            # Step 6 also needs the quartz trace database reference
            if 6 in selected_steps and not quartz:
                raise RuntimeError(
                    "Step 6 requires the quartz reference file (with trace database). "
                    "Provide --quartz <quartz_file> or run steps 1-4 first."
                )

            # If step2_expected is empty but we need step 2 outputs, scan for them
            if not step2_expected and required_step2_keys:
                step2_expected = find_step2_outputs(args.input_dir, args.night, args.shoe)
                if not step2_expected:
                    print(
                        "  [warn] Could not auto-discover step 2 outputs (quartz_sl, etc.).\n"
                        "         Provide explicit base file flags (--quartz, --thar, --object) for steps 5+."
                    )

            for key in required_step2_keys:
                if key not in step2_expected:
                    raise RuntimeError(
                        f"Could not find required step 2 output for {key}. "
                        "Provide explicit base file flags for selected steps (e.g., --quartz Quartz-...-sl.fits)."
                    )
                if step2_expected[key] is None:
                    raise RuntimeError(
                        f"Step 2 output for {key} is None (internal error). "
                        "Provide explicit base file flags (e.g., --quartz Quartz-...-sl.fits)."
                    )
                require_existing(step2_expected[key], f"step 2 output ({key})")
                state[key] = step2_expected[key]
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 3. Normalised master flat ─────────────────────────────────────────────
    if 3 in selected_steps:
        try:
            require_existing(state["quartz_sl"], "step 3 input quartz-sl")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")
        state["master_flat"] = make_normalised_flat(state["quartz_sl"])
    elif 4 in selected_steps:
        try:
            state["master_flat"] = expected_master_flat(state["quartz_sl"])
            require_existing(state["master_flat"], "step 3 output master flat")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 4. Flat-field correction ──────────────────────────────────────────────
    if 4 in selected_steps:
        try:
            require_existing(state["thar_sl"], "step 4 input thar-sl")
            require_existing(state["obj_sl"], "step 4 input object-sl")
            require_existing(state["twilight_sl"], "step 4 input twilight-sl")
            require_existing(state["master_flat"], "step 4 input master flat")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

        thar_ff, obj_ff, twi_ff = flatcorrect_images(
            state["thar_sl"], state["obj_sl"], state["twilight_sl"], state["master_flat"],
        )
        state.update({"thar_ff": thar_ff, "obj_ff": obj_ff, "twi_ff": twi_ff})

        write_output_metadata(thar_ff, {
            "OBJECT": meta_by_role["thar"]["OBJECT"],
            "EXPTYPE": meta_by_role["thar"]["EXPTYPE"],
            "NIGHT": meta_by_role["thar"]["NIGHT"],
            "SHOE": meta_by_role["thar"]["SHOE"],
            "PROCSTEP": "reduce_step4_flatcorr",
        })
        write_output_metadata(obj_ff, {
            "OBJECT": meta_by_role["object"]["OBJECT"],
            "EXPTYPE": meta_by_role["object"]["EXPTYPE"],
            "NIGHT": meta_by_role["object"]["NIGHT"],
            "SHOE": meta_by_role["object"]["SHOE"],
            "PROCSTEP": "reduce_step4_flatcorr",
        })
        write_output_metadata(twi_ff, {
            "OBJECT": meta_by_role["twilight"]["OBJECT"],
            "EXPTYPE": meta_by_role["twilight"]["EXPTYPE"],
            "NIGHT": meta_by_role["twilight"]["NIGHT"],
            "SHOE": meta_by_role["twilight"]["SHOE"],
            "PROCSTEP": "reduce_step4_flatcorr",
        })
    elif 6 in selected_steps:
        try:
            state.update(expected_step4_outputs(
                state["thar_sl"], state["obj_sl"], state.get("twilight_sl"),
            ))
            require_existing(state["thar_ff"], "step 4 output thar-ff")
            require_existing(state["obj_ff"], "step 4 output object-ff")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 5. Interactive aperture preview on traced quartz reference ───────────
    if 5 in selected_steps:
        preview_quartz = quartz if quartz else state.get("quartz_sl")
        if not preview_quartz:
            sys.exit(
                "ERROR dependency check: step 5 requires a quartz reference image. "
                "Provide --quartz or ensure quartz_sl can be auto-discovered."
            )
        _centers, pattern = run_aperture_preview(
            preview_quartz,
            n_stars = args.nstars,
            sep     = args.sep,
        )
        state["pattern"] = pattern
        map_path = args.affiliation_map or default_affiliation_map_path(
            meta_by_role,
            reference_path=preview_quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
        )
        try:
            save_affiliation_map(map_path, pattern, preview_quartz, meta_by_role)
        except Exception as exc:
            sys.exit(f"ERROR saving affiliation map: {exc}")
        print(f"\n  Total apertures in accepted mapping: {len(pattern)}")
    elif 6 in selected_steps:
        map_path = args.affiliation_map or default_affiliation_map_path(
            meta_by_role,
            reference_path=quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
        )
        if os.path.exists(map_path):
            try:
                mapping = load_affiliation_map(map_path)
                state["pattern"] = validate_affiliation_map(mapping, quartz, meta_by_role)
                print(f"  Loaded affiliation map: {map_path}")
                print(f"  Using saved pattern with {len(state['pattern'])} apertures")
            except Exception as exc:
                sys.exit(f"ERROR loading affiliation map '{map_path}': {exc}")
        else:
            if not args.use_auto_pattern:
                sys.exit(
                    "ERROR dependency check: step 6 selected without step 5 and no saved "
                    f"affiliation map found at {map_path}. Run step 5 first, pass "
                    "--affiliation-map, or use --use-auto-pattern."
                )
            section_banner("Step 5 (skipped) – Using fallback aperture pattern")
            _centers, pattern = _fallback_pattern(state["obj_ff"], args.nstars, args.sep)
            state["pattern"] = pattern
            print(f"  Auto pattern generated: {len(pattern)} apertures")

    # ── 6. Per-star extraction ────────────────────────────────────────────────
    if 6 in selected_steps:
        try:
            require_quartz_trace_db(quartz, "step 6")
            require_existing(state["obj_ff"], "step 6 input object flat-corrected")
            require_existing(state["thar_ff"], "step 6 input thar flat-corrected")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

        try:
            obj_meta = meta_by_role.get("object") or read_required_metadata(state["obj_ff"])
            thar_meta = meta_by_role.get("thar") or read_required_metadata(state["thar_ff"])
        except Exception as exc:
            sys.exit(f"ERROR dependency check: could not read step-6 metadata: {exc}")

        obj_outputs, thar_outputs = extract_all_stars(
            state["obj_ff"], state["thar_ff"],
            quartz  = quartz,
            pattern = state["pattern"],
        )
        for star_num in sorted(obj_outputs):
            write_output_metadata(obj_outputs[star_num], {
                "OBJECT": obj_meta["OBJECT"],
                "EXPTYPE": "Object_extract",
                "NIGHT": obj_meta["NIGHT"],
                "SHOE": obj_meta["SHOE"],
                "STARNUM": int(star_num),
                "PROCSTEP": "reduce_step6_extract",
            })

        for star_num in sorted(thar_outputs):
            write_output_metadata(thar_outputs[star_num], {
                "OBJECT": thar_meta["OBJECT"],
                "EXPTYPE": "ThAr_extract",
                "NIGHT": thar_meta["NIGHT"],
                "SHOE": thar_meta["SHOE"],
                "STARNUM": int(star_num),
                "PROCSTEP": "reduce_step6_extract",
            })

        try:
            night, shoe = infer_night_shoe(
                meta_by_role,
                reference_path=quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            pair_index = f"extraction_pairs_{night}_{shoe}.csv"
            write_extraction_pairs_index(pair_index, obj_outputs, thar_outputs, state["pattern"])
        except Exception as exc:
            sys.exit(f"ERROR writing extraction pairing index: {exc}")

    # ── 7. Second cosmic-ray removal ─────────────────────────────────────────
    if 7 in selected_steps:
        if not obj_outputs:
            # Running step 7 standalone: reconstruct obj_outputs from step-6
            # expected paths so the user can run --start-step 7 --end-step 8.
            try:
                if state.get("obj_ff"):
                    require_existing(state.get("obj_ff", ""), "step 7 input (obj_ff needed to infer step-6 outputs)")
                    _ap_str_map = {}
                    if "pattern" in state:
                        _pat = np.asarray(state["pattern"], dtype=int)
                        for _s in sorted(np.unique(_pat)):
                            _ap_str_map[_s] = _aperture_range_string(
                                list(np.where(_pat == _s)[0] + 1)
                            )
                    _obj_ff = state.get("obj_ff", stem(obj or "") + "-sl-F.fits")
                    obj_outputs = {
                        _s: f"{stem(_obj_ff)}_star{_s:02d}_ec.fits"
                        for _s in (_ap_str_map or {1: ""})
                    }
                    for _s, _p in obj_outputs.items():
                        require_existing(_p, f"step 7 input star {_s:02d} object ec")
                else:
                    reconstructed_obj, reconstructed_thar = reconstruct_star_outputs_from_disk(args)
                    obj_outputs = {}
                    for _s, _p in reconstructed_obj.items():
                        ec_path = _p
                        if ec_path.endswith("_ec-crr2.fits"):
                            candidate_ec = ec_path.replace("_ec-crr2.fits", "_ec.fits")
                            if not os.path.exists(candidate_ec):
                                raise RuntimeError(
                                    f"Step 7 requires extracted *_ec.fits inputs, but star {_s:02d} "
                                    f"has only CR-cleaned output: {ec_path}. "
                                    "Re-run step 6 with current extraction-only behavior, "
                                    "or run step 8/9 directly if CR-cleaned spectra are already final."
                                )
                            ec_path = candidate_ec
                        require_existing(ec_path, f"step 7 input star {_s:02d} object ec")
                        obj_outputs[_s] = ec_path
                    thar_outputs = reconstructed_thar
            except Exception as exc:
                sys.exit(f"ERROR dependency check: {exc}")

        if args.no_cr:
            print("  Step 7 CR removal skipped due to --no-cr; using step-6 outputs as-is.")
            crr2_outputs = dict(obj_outputs)
        else:
            crr2_outputs = second_cosmic_removal(obj_outputs)
        state["crr2_outputs"] = crr2_outputs

    elif 8 in selected_steps or 9 in selected_steps:
        # Step 8/9 without step 7: expect step-7 outputs already on disk.
        try:
            if not obj_outputs or not thar_outputs:
                obj_outputs, thar_outputs = reconstruct_star_outputs_from_disk(args)

            crr2_outputs = {star: _prefer_crr2(path) for star, path in obj_outputs.items()}
            for _s, _p in crr2_outputs.items():
                require_existing(_p, f"step 8/9 input star {_s:02d} object extracted")
            for _s, _p in thar_outputs.items():
                require_existing(_p, f"step 8/9 input star {_s:02d} thar extracted")
            state["crr2_outputs"] = crr2_outputs
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 8. Manual wavelength identification (interactive on reference star) ───
    if 8 in selected_steps:
        crr2_outputs = state.get("crr2_outputs", {})
        if not crr2_outputs:
            sys.exit("ERROR: step 8 has no CR-cleaned spectra to calibrate.")
        if not thar_outputs:
            sys.exit(
                "ERROR dependency check: step 8 requires ThAr extractions. "
                "Run step 6 first or include it in the step range."
            )
        ref_star = manual_wavelength_identification(
            crr2_outputs, thar_outputs,
            coordlist = args.coordlist,
        )
        state["ref_star"] = ref_star

    # ── 9. Automatic wavelength propagation (ecreidentify to remaining stars) ─
    if 9 in selected_steps:
        crr2_outputs = state.get("crr2_outputs", {})
        if not crr2_outputs:
            sys.exit("ERROR: step 9 has no CR-cleaned spectra to calibrate.")
        if not thar_outputs:
            sys.exit(
                "ERROR dependency check: step 9 requires ThAr extractions. "
                "Run step 6 first or include it in the step range."
            )
        # Determine reference star: from step 8 if it ran, otherwise first star
        ref_star = state.get("ref_star", min(crr2_outputs.keys()))
        
        # If step 8 didn't run but we're running step 9, verify ref star has solution
        if 8 not in selected_steps:
            ref_thar = thar_outputs[ref_star]
            db_stem_cand = stem(ref_thar)
            db_file = f"./database/ec.{db_stem_cand}"
            if not os.path.exists(db_file):
                sys.exit(
                    f"ERROR: step 9 requires wavelength solution from step 8. "
                    f"Run step 8 first on reference star {ref_star:02d}."
                )
        
        wlcal_outputs = auto_wavelength_propagation(
            crr2_outputs, thar_outputs, ref_star,
            coordlist = args.coordlist,
        )
        state["wlcal_outputs"] = wlcal_outputs

    # ── Summary ───────────────────────────────────────────────────────────────
    section_banner("Pipeline complete")
    print(f"  Executed steps       : {args.start_step}..{args.end_step}")
    if "master_flat" in state:
        print(f"  Master flat          : {state['master_flat']}")
    if "thar_ff" in state:
        print(f"  Flat-corrected ThAr  : {state['thar_ff']}")
    if "obj_ff" in state:
        print(f"  Flat-corrected object: {state['obj_ff']}")
    if "twi_ff" in state:
        print(f"  Flat-corrected twi   : {state['twi_ff']}")
    if obj_outputs or thar_outputs or state.get("crr2_outputs") or state.get("wlcal_outputs"):
        print("")
        for s in sorted(set(list(obj_outputs) + list(thar_outputs))):
            if s in obj_outputs:
                print(f"  Star {s:02d}  object    : {obj_outputs[s]}")
            if s in thar_outputs:
                print(f"         thar      : {thar_outputs[s]}")
            if s in state.get("crr2_outputs", {}):
                print(f"         crr2      : {state['crr2_outputs'][s]}")
            if s in state.get("wlcal_outputs", {}):
                print(f"         wlcal     : {state['wlcal_outputs'][s]}")
        if state.get("wlcal_outputs"):
            print(
                "\n  Wavelength-calibrated spectra ready for continuum normalisation."
            )
        elif obj_outputs:
            print(
                "\n  Next steps for each star:\n"
                "    Step 7  lineclean         ->  *_star<N>_ec-crr2.fits\n"
                "    Step 8  ecidentify on ThAr, refspec + dispcor  ->  *-wl.fits\n"
            )


if __name__ == "__main__":
    main()