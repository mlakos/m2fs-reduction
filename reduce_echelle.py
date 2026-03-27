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
        – user corrects assignments if needed, then presses q or g
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
  8.  Reference-star wavelength setup  (manual or reuse existing reference)
      a. choose mode for first (reference) star:
        - manual: ecidentify (interactive) on that star's ThAr
        - reuse : ecreidentify from an already identified ThAr reference
      b. refspec     assign the solution to that star's CR-cleaned object
      – The solution is saved/used in the IRAF database for step 9
  9.  Automatic line-ID propagation + review  (ecreidentify to remaining stars)
      a. ecreidentify (automatic)   propagates line IDs from ref star to others
      b. review      inspect reidentified ThAr line IDs (interactive when TTY)
      c. refspec     assign the reviewed IDs to each object spectrum
      – outputs: no new FITS files in this section (refspec updates assignments)
      – workflow: run step 8 (interactive), then step 9 (automatic + review)

# NOTE: "ENOISE" and "EGAIN" keyowrds should always be used in iraf values for readnoise and gain

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
        [--nstars 4]  [--sep 5.0]  [--dispaxis 1]
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
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


def iraf_spec_token(path):
    """Return canonical IRAF spectroscopy/database image token (./<root>)."""
    # Filesystem names stay as *.fits; IRAF spectroscopy tasks use ./<root>.
    return stem(path)


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
        resolved.update({role: path for role, path in manual.items() if path})
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

def apall_trace_quartz(quartz, n_ap):
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

def run_aperture_preview(preview_image, n_stars, sep, preview_out=None):
    """
    Open the interactive matplotlib preview on a reference image for assigning
    aperture traces to stars. This does not require the final extraction image;
    the accepted star-assignment pattern is reused in step 6.

    Returns (centers, pattern) once the user presses q or g.
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
        "    q/g accept and continue  <-- this moves the pipeline forward\n"
        "    h   print this reminder\n"
    )
    centers, pattern = run_preview(
        preview_image,
        n_stars=n_stars,
        sep=sep,
        out=preview_out,
    )

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

    iraf_reference = f"./{stem(reference_quartz)}"

    iraf.echelle.apall.unlearn()
    iraf.echelle.apall(
        input       = image,
        output      = out_stem,          # IRAF appends .fits
        apertures   = aperture_string,   # only this star's apertures
        format      = "echelle",
        references  = iraf_reference,    # IRAF DB lookup is basename-oriented
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
        b_niterate  = 5,
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

    if not os.path.exists(out_spec):
        db_candidates = quartz_trace_db_candidates(reference_quartz)
        raise RuntimeError(
            "apall produced no output for "
            f"{os.path.basename(image)} (star {star_number:02d}, apertures={aperture_string}). "
            "IRAF reported no apertures defined. "
            f"Reference DB candidates checked: {', '.join(db_candidates)}"
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
            iraf.images.imfit.lineclean.unlearn()
        except AttributeError:
            print(f"      [warn] lineclean task not found in IRAF; skipping CR removal for {input_spec}")
            # Fall back to copying file without CR removal
            import shutil
            shutil.copy(input_spec, output_spec)
            return
    
        iraf.images.imfit.lineclean(
                input     = input_spec,
                output    = output_spec,
                sample    = "*",
                naverage  = 1,
                function  = "spline3",
                order     = 6,
                low_reject = 50.0,
                high_reject = 3.0,
                niterate  = 10,
                grow      = 1.0,
                interactive = iraf.no,
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
    iraf.images()
    iraf.imfit()
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
# Step 8/9 – line identification propagation (ecidentify/ecreidentify + refspec)
# ---------------------------------------------------------------------------

def _ecidentify_thar(thar_ec, coordlist="linelists$thar.dat"):
    """
    Run ecidentify interactively on a ThAr multispec file to build the
    wavelength solution.

    Per the McDonald notes:
      - First run: maxfeatures=100, identify ~5 lines per order manually,
        fit with legendre xorder=4, yorder=4, then press 'l' to load more
        lines from the line list.
      - Second run (same call after 'q'+'f'+'l'): maxfeatures=100,
        IRAF remembers the previously identified lines; press 'f' to refit,
        load more, inspect, quit and save.
    We set maxfeatures=100 here; the user is expected to do the
    high-maxfeature pass by re-running the task in the IRAF terminal if
    needed, or by calling this function a second time with maxfeatures=100.
    """
    iraf_image = iraf_spec_token(thar_ec)
    print(f"  ecidentify (interactive): file={thar_ec}, iraf={iraf_image}")
    # Get the task object
    ecid = iraf.noao.imred.echelle.ecidentify

    # Reset learned/cached task parameters
    iraf.unlearn(ecid)

    # Set parameters first
    ecid.images     = iraf_image
    ecid.database   = "database"
    ecid.coordlist  = coordlist
    ecid.units      = ""
    ecid.match      = 1.0
    ecid.maxfeatures = 100
    ecid.zwidth     = 10.0
    ecid.ftype      = "emission"
    ecid.fwidth     = 4.0
    ecid.cradius    = 5.0
    ecid.threshold  = 10.0
    ecid.minsep     = 2.0
    ecid.function   = "legendre"
    ecid.xorder     = 4
    ecid.yorder     = 4
    ecid.niterate   = 5
    ecid.lowreject  = 3.0
    ecid.highreject = 3.0
    ecid.autowrite  = iraf.no
    ecid.graphics   = "stdgraph"
    ecid.cursor     = ""
    ecid.mode       = "ql"

    # Inspect before execution
    ecid.lParam()

    # Then execute
    ecid()
    print(f"  ecidentify done: {thar_ec}")


def _ecreidentify_thar(thar_ec, ref_thar_ec, drift_log_path=None, drift_stage="step9",
                       predicted_shift=None, search_radius=None,
                       allow_indef_fallback=True,
                       reference_override=None):
    """
    Run ecreidentify (automatic) on a ThAr multispec using a reference solution.

    Propagates the wavelength solution from ref_thar_ec to thar_ec based on
    spatial correlation of arc lines. This is much faster than manual identification.

    Parameters
    ----------
    thar_ec       : path to ThAr spectrum to be identified (uses reference)
    ref_thar_ec   : path to reference ThAr spectrum with existing solution
    predicted_shift : float or None
        Model-driven shift hint for ecreidentify. If None, uses INDEF.
    search_radius : float or None
        Correlation search radius in pixels. If None, defaults to 8.0.
    allow_indef_fallback : bool
        If True, retry once with shift=INDEF when hinted call fails.
    """
    reference_input = reference_override if reference_override else ref_thar_ec
    iraf_target_token = iraf_spec_token(thar_ec)
    iraf_reference_token = iraf_spec_token(reference_input)

    found_db = None
    for probe in (reference_input, os.path.basename(reference_input), stem(reference_input), iraf_reference_token):
        found_db, _db_candidates = resolve_existing_wavelength_db(probe)
        if found_db:
            break

    if found_db:
        # Keep DB lookup permissive, but normalize actual IRAF task identity.
        for alias_probe in (reference_input, os.path.basename(reference_input), stem(reference_input), iraf_reference_token):
            ensure_wavelength_db_aliases(alias_probe, found_db)

    shift_value = "INDEF" if predicted_shift is None else float(predicted_shift)
    cradius_value = 8.0 if search_radius is None else float(search_radius)
    fallback_used = False

    print(
        "  ecreidentify (automatic): "
        f"target_file={thar_ec}, target_iraf={iraf_target_token}, "
        f"ref_file={reference_input}, ref_iraf={iraf_reference_token}  "
        f"[shift={shift_value}, cradius={cradius_value}]"
    )
    task_log = f".ecreidentify_{stem(thar_ec)}_{os.getpid()}.log"
    iraf.noao.imred.echelle.ecreidentify.unlearn()

    def _run_once(shift_arg):
        iraf.noao.imred.echelle.ecreidentify(
            images     = iraf_target_token,
            reference  = iraf_reference_token,
            shift      = shift_arg,
            cradius    = cradius_value,
            threshold  = 10.0,
            refit      = iraf.yes,
            database   = "database",
            logfiles   = f"STDOUT,{task_log}",
        )

    try:
        _run_once(shift_value)
    except Exception as exc:
        should_fallback = (
            allow_indef_fallback and
            shift_value != "INDEF"
        )
        if not should_fallback:
            raise
        print(
            "  [warn] hinted ecreidentify failed; retrying with shift=INDEF "
            f"({exc})"
        )
        fallback_used = True
        _run_once("INDEF")
        shift_value = "INDEF"

    metrics = _parse_ecreidentify_metrics(task_log, thar_ec)
    if metrics is None:
        metrics = {}
    metrics.update(
        {
            "requested_shift": predicted_shift,
            "used_shift": shift_value,
            "search_radius": cradius_value,
            "fallback_indef": bool(fallback_used),
        }
    )
    append_reidentify_drift(
        drift_log_path,
        stage=drift_stage,
        target_thar=os.path.basename(thar_ec),
        reference_thar=os.path.basename(reference_input),
        metrics=metrics,
    )
    parsed_metrics = (
        metrics.get("found_num") is not None and
        metrics.get("found_den") is not None and
        metrics.get("fit_num") is not None and
        metrics.get("fit_den") is not None
    )
    if parsed_metrics:
        found_num = metrics.get("found_num")
        found_den = metrics.get("found_den")
        fit_num = metrics.get("fit_num")
        fit_den = metrics.get("fit_den")
        pix_shift = metrics.get("pix_shift")
        rms = metrics.get("rms")
        print(
            "  drift: "
            f"found={found_num}/{found_den}, fit={fit_num}/{fit_den}, "
            f"pix_shift={pix_shift}, rms={rms}"
        )
    else:
        print(f"  [warn] could not parse ecreidentify drift metrics for {thar_ec}")

    if os.path.exists(task_log):
        os.remove(task_log)
    print(f"  ecreidentify done: {thar_ec}")
    return metrics


def _as_float_or_none(value):
    """Return float(value) when possible, otherwise None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "INDEF":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_ecreidentify_metrics(log_path, target_image):
    """Parse IRAF ecreidentify summary row from a logfile."""
    if not os.path.exists(log_path):
        return None

    target_stem = stem(target_image)
    pattern = re.compile(
        r"^\s*(\S+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+([-+\d\.Ee]+)\s+([-+\d\.Ee]+)\s+([-+\d\.Ee]+)\s+(\S+)\s*$"
    )

    with open(log_path, "r") as fh:
        for line in fh:
            m = pattern.match(line)
            if not m:
                continue
            image_token = m.group(1)
            if stem(image_token) != target_stem:
                continue

            found_num = int(m.group(2))
            found_den = int(m.group(3))
            fit_num = int(m.group(4))
            fit_den = int(m.group(5))
            pix_shift = _as_float_or_none(m.group(6))
            user_shift = _as_float_or_none(m.group(7))
            z_shift = _as_float_or_none(m.group(8))
            rms = _as_float_or_none(m.group(9))
            found_frac = (found_num / found_den) if found_den else None
            fit_frac = (fit_num / fit_den) if fit_den else None
            return {
                "image": image_token,
                "found_num": found_num,
                "found_den": found_den,
                "fit_num": fit_num,
                "fit_den": fit_den,
                "found_frac": found_frac,
                "fit_frac": fit_frac,
                "pix_shift": pix_shift,
                "user_shift": user_shift,
                "z_shift": z_shift,
                "rms": rms,
            }
    return None


def append_reidentify_drift(csv_path, stage, target_thar, reference_thar, metrics):
    """Append one ecreidentify drift/quality row to CSV."""
    if not csv_path:
        return

    fieldnames = [
        "stage", "target_thar", "reference_thar", "image",
        "found_num", "found_den", "fit_num", "fit_den",
        "found_frac", "fit_frac", "pix_shift", "user_shift", "z_shift", "rms",
    ]

    row = {
        "stage": stage,
        "target_thar": target_thar,
        "reference_thar": reference_thar,
        "image": "",
        "found_num": "",
        "found_den": "",
        "fit_num": "",
        "fit_den": "",
        "found_frac": "",
        "fit_frac": "",
        "pix_shift": "",
        "user_shift": "",
        "z_shift": "",
        "rms": "",
    }
    if metrics:
        row.update({k: metrics.get(k, "") for k in fieldnames if k in metrics})

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_reidentify_quality(metrics, min_found_frac, min_fit_frac, max_rms):
    """Evaluate ecreidentify summary metrics against quality thresholds."""
    if not metrics:
        return False, ["missing ecreidentify summary metrics"]

    failures = []

    found_frac = metrics.get("found_frac")
    if found_frac is None:
        failures.append("found fraction unavailable")
    elif found_frac < min_found_frac:
        failures.append(f"found_frac={found_frac:.3f} < {min_found_frac:.3f}")

    fit_frac = metrics.get("fit_frac")
    if fit_frac is None:
        failures.append("fit fraction unavailable")
    elif fit_frac < min_fit_frac:
        failures.append(f"fit_frac={fit_frac:.3f} < {min_fit_frac:.3f}")

    rms = metrics.get("rms")
    if rms is None:
        failures.append("rms unavailable")
    elif rms > max_rms:
        failures.append(f"rms={rms:.3f} > {max_rms:.3f}")

    return (len(failures) == 0), failures


def ensure_wavelength_db_aliases(thar_path, source_db):
    """Populate expected DB alias names for a resolved reference ThAr solution."""
    if not source_db or not os.path.exists(source_db):
        return

    src_abs = os.path.abspath(source_db)
    for candidate in wavelength_db_candidates(thar_path):
        if os.path.exists(candidate):
            continue
        try:
            os.symlink(src_abs, candidate)
        except OSError:
            shutil.copy2(source_db, candidate)


def _review_reidentified_lines(thar_ec, coordlist):
    """Offer an interactive review pass for reidentified ThAr line IDs."""
    iraf_image = iraf_spec_token(thar_ec)
    fully_interactive = (
        sys.stdin.isatty() and
        sys.stdout.isatty() and
        sys.stderr.isatty()
    )
    if not fully_interactive:
        raise RuntimeError(
            "Step 9 review is required after successful reidentify, "
            f"but no interactive TTY is available for {thar_ec}."
        )

    print(
        "  Opening ecidentify review for reidentified spectrum.\n"
        "  Inspect line IDs/residuals, then quit back to continue Step 9.\n"
        f"  review target: file={thar_ec}, iraf={iraf_image}"
    )
    iraf.noao.echelle.ecidentify.unlearn()
    iraf.noao.echelle.ecidentify(
        images    = iraf_image,
        database  = "database",
        coordlist = coordlist,
        units     = "",
        match     = 1.0,
        maxfeatur = 100,
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
    print(f"  review complete: {thar_ec}")


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


def _choose_step8_mode(step8_mode):
    """Resolve Step-8 mode, prompting only for interactive TTY runs."""
    if step8_mode in {"manual", "reuse"}:
        return step8_mode

    if not sys.stdin.isatty():
        print("  step 8 mode: non-interactive run, defaulting to manual ecidentify")
        return "manual"

    print("\n  Step 8 mode for reference star:")
    print("    [m] manual ecidentify")
    print("    [r] reuse existing identified ThAr (run ecreidentify)")

    while True:
        answer = input("  Choose mode [m/r] (default: m): ").strip().lower()
        if answer in {"", "m", "manual"}:
            return "manual"
        if answer in {"r", "reuse"}:
            return "reuse"
        print("  Please type 'm' for manual or 'r' for reuse.")


def _resolve_reuse_reference_thar(reference_thar):
    """Resolve and validate external reference ThAr path for Step-8 reuse mode."""
    ref_path = reference_thar

    if not ref_path and sys.stdin.isatty():
        ref_path = input(
            "  Path to already identified reference ThAr FITS: "
        ).strip()

    if not ref_path:
        raise RuntimeError(
            "Step 8 reuse mode requires --step8-reference-thar <identified_thar.fits>."
        )

    require_existing(ref_path, "step 8 reuse reference thar")

    found_db, db_candidates = resolve_existing_wavelength_db(ref_path)
    if not found_db:
        raise RuntimeError(
            "Step 8 reuse mode requires an existing IRAF wavelength database entry "
            f"for reference ThAr '{ref_path}'. Checked: {', '.join(db_candidates)}"
        )

    ensure_wavelength_db_aliases(ref_path, found_db)
    ref_token = os.path.basename(ref_path)
    ensure_wavelength_db_aliases(ref_token, found_db)

    print(f"  reuse reference ThAr : {ref_token}")
    print(f"  reuse wavelength DB  : {found_db}")
    return ref_token


def manual_wavelength_identification(crr2_outputs, thar_outputs,
                                     coordlist="linelists$thar.dat",
                                     step8_mode="ask",
                                     step8_reference_thar=None,
                                     drift_log_path=None):
    """
    Step 8: Reference-star wavelength setup (manual or reuse mode).

    Manual mode runs ecidentify on one reference-star ThAr spectrum.
    Reuse mode runs ecreidentify on that reference star, using an already
    identified external ThAr reference and its existing IRAF DB entry.
    Both modes then run refspec on the reference-star object spectrum.

    The reference star is the first star (minimum) in the star list.

    Parameters
    ----------
    crr2_outputs : {star_number: path, ...}
        CR-cleaned object spectra from step 7.
    thar_outputs : {star_number: path, ...}
        ThAr spectra from step 6 (same aperture selection as objects).
    coordlist    : str
        IRAF line-list path (default: built-in ThAr list).
    step8_mode   : {'ask', 'manual', 'reuse'}
        Step-8 mode selector.
    step8_reference_thar : str or None
        Path to identified reference ThAr for reuse mode.

    Returns
    -------
    ref_star : int
        The star number used as reference (minimum in sorted list).
    """
    section_banner("Step 8 – Reference-star wavelength setup (manual/reuse)")
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

    mode = _choose_step8_mode(step8_mode)

    if mode == "manual":
        # 8a – ecidentify on reference ThAr (interactive)
        _ecidentify_thar(thar_ec_ref, coordlist=coordlist)
    else:
        # 8a-alt – reuse external identified reference via ecreidentify.
        ref_thar_external = _resolve_reuse_reference_thar(step8_reference_thar)
        print(
            "  reuse mode: ecreidentify on reference star using existing identified ThAr"
        )
        _ecreidentify_thar(
            thar_ec_ref,
            ref_thar_external,
            drift_log_path=drift_log_path,
            drift_stage="step8_reuse",
        )

    # 8b – refspec: attach ThAr solution to the reference object
    _refspec_one(obj_crr2_ref, thar_ec_ref)

    return ref_star


def auto_wavelength_propagation(crr2_outputs, thar_outputs, ref_star,
                                coordlist="linelists$thar.dat",
                                drift_log_path=None,
                                step9_gate_mode="warn",
                                step9_min_found_frac=0.05,
                                step9_min_fit_frac=0.05,
                                step9_max_rms=0.30,
                                star_geometry=None,
                                step5_geometry_path=None,
                                extraction_pairs_path=None):
    """Step 9: automatic line-ID propagation and required review for non-reference stars."""
    section_banner("Step 9 – Automatic line-ID propagation (ecreidentify + review + refspec)")
    iraf.noao()
    iraf.echelle()
    iraf.onedspec()

    thar_ec_ref = thar_outputs[ref_star]
    master_ref_apertures = _read_apertures_from_apnum_cards(thar_ec_ref)
    if not master_ref_apertures:
        raise RuntimeError(
            f"Could not read APNUM apertures from master reference ThAr: {thar_ec_ref}"
        )

    id_assigned_outputs = {}
    failed_gate_stars = []

    if step9_gate_mode != "off":
        print(
            "  Step 9 gate: "
            f"mode={step9_gate_mode}, "
            f"min_found_frac={step9_min_found_frac:.3f}, "
            f"min_fit_frac={step9_min_fit_frac:.3f}, "
            f"max_rms={step9_max_rms:.3f}"
        )

    star_order = star_order_by_distance_from_reference(
        star_geometry,
        ref_star,
        list(crr2_outputs.keys()),
    )
    print(
        "  Step 9 order: "
        + ", ".join(f"{int(s):02d}" for s in star_order)
        + " (nearest in y to reference when geometry is available)"
    )

    for star in star_order:
        obj_crr2 = crr2_outputs[star]
        thar_ec = thar_outputs[star]

        print(f"\n  -- Star {star:02d}")
        print(f"     object : {obj_crr2}")
        print(f"     thar   : {thar_ec}")
        print(f"     output : {obj_crr2}  (line IDs assigned in-place)")

        if star == ref_star:
            print("     (reference star — solution from step 8)")
        else:
            target_apertures, aperture_source = get_target_aperture_numbers(
                star,
                thar_ec,
                geometry_path=step5_geometry_path,
                extraction_pairs_path=extraction_pairs_path,
            )
            print(f"     real target file : {os.path.basename(thar_ec)}")
            print(f"     target apertures : {target_apertures} (source={aperture_source})")
            print(f"     master apertures : {master_ref_apertures}")

            prep = None
            gate_failed = False
            try:
                prep = _prepare_temp_target_for_reidentify(
                    thar_ec_ref,
                    thar_ec,
                    star,
                    target_apertures,
                )
                print(f"     temp target file : {prep['temp_target_path']}")
                print(
                    "     target->master   : "
                    f"{_format_aperture_mapping(prep['target_to_master'])}"
                )

                metrics = _ecreidentify_thar(
                    prep["temp_target_path"],
                    thar_ec_ref,
                    drift_log_path=drift_log_path,
                    drift_stage="step9_propagation",
                )

                if step9_gate_mode != "off":
                    gate_ok, failures = evaluate_reidentify_quality(
                        metrics,
                        min_found_frac=step9_min_found_frac,
                        min_fit_frac=step9_min_fit_frac,
                        max_rms=step9_max_rms,
                    )
                    if gate_ok:
                        print("     gate: PASS")
                    else:
                        failure_text = "; ".join(failures)
                        print(f"     gate: FAIL ({failure_text})")
                        print("     gate action: skipping review/refspec for this star")
                        failed_gate_stars.append(star)
                        gate_failed = True

                if not gate_failed:
                    print("     review start     : ecidentify on temporary target")
                    _review_reidentified_lines(prep["temp_target_path"], coordlist=coordlist)
                    print("     review end       : ecidentify complete")

                    transfer = _transfer_reviewed_temp_target_solution_to_real_target(
                        prep["temp_target_path"],
                        thar_ec,
                        prep["master_to_target"],
                    )
                    print(
                        "     db transfer      : "
                        f"{os.path.basename(transfer['temp_db_path'])} -> "
                        f"{os.path.basename(transfer['real_db_path'])}"
                    )
            finally:
                if prep is not None:
                    removed, failed = _cleanup_temporary_reference_artifacts(
                        prep["temp_target_path"],
                        prep["cleanup_db_candidates"],
                    )
                    if failed:
                        print(f"     cleanup status   : WARN ({'; '.join(failed)})")
                    else:
                        print(f"     cleanup status   : OK ({len(removed)} temporary files removed)")

            if gate_failed:
                continue

        print("     step9 sequence: applying refspec")
        _refspec_one(obj_crr2, thar_ec)
        id_assigned_outputs[star] = obj_crr2

    if failed_gate_stars:
        failed_text = ", ".join(f"{s:02d}" for s in failed_gate_stars)
        print(f"\n  Step 9 gate skipped stars: {failed_text}")
        print("  These stars need manual fallback (step 8-style identification path).")

    return id_assigned_outputs


def expected_step8_outputs(crr2_outputs):
    """Return deterministic step-8 reference-star outputs (refspec in-place)."""
    ref_star = min(crr2_outputs.keys())
    return {ref_star: crr2_outputs[ref_star]}


def expected_step9_outputs(crr2_outputs):
    """Return deterministic step-9 outputs (refspec assignment in-place)."""
    return {star: path for star, path in crr2_outputs.items()}


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
              6. Per-star extraction with accepted pattern  ->  *_starNN_ec.fits
              7. Second CR removal (lineclean)  ->  *_ec-crr2.fits
              8. Reference-star wavelength setup (manual/reuse)
              9. Automatic wavelength propagation + review + refspec
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--quartz", help="Quartz mosaic FITS (manual override).")
    p.add_argument(
        "--quartz-reference",
        default=None,
        help="Use this traced quartz reference FITS for steps 5-6 instead of auto alias.",
    )
    p.add_argument("--thar", help="ThAr mosaic FITS (manual override).")
    p.add_argument("--object", help="Science object mosaic FITS (manual override).")
    p.add_argument("--twilight", help="Twilight sky mosaic FITS (manual override).")
    p.add_argument("--input-dir", default=".",
                   help="Directory to scan for auto-discovery (default: .).")
    p.add_argument("--run-preprocess", action="store_true",
                   help="Run image_processing.py on --input-dir before echelle steps.")
    p.add_argument("--preprocess-infiles", default=None,
                   help="Output path for generated infiles list (default: <input-dir>/infiles).")
    p.add_argument("--preprocess-object", default=None,
                   help="Pass object name through to image_processing.py --object.")
    p.add_argument("--preprocess-bias", action="store_true",
                   help="Pass --bias to image_processing.py.")
    p.add_argument("--preprocess-flat", default=None,
                   help="Pass --flat <file> to image_processing.py.")
    p.add_argument("--night", default=None,
                   help="Restrict auto-discovery to this NIGHT value.")
    p.add_argument("--shoe", default=None, choices=["B", "R", "b", "r"],
                   help="Restrict auto-discovery to this SHOE value.")
    p.add_argument("--object-name", default=None,
                   help="Substring filter applied to auto-discovered science OBJECT.")
    p.add_argument("--yes", action="store_true",
                   help="Continue despite metadata mismatch warnings.")
    p.add_argument("--nstars", type=int, default=24,
                   help="Number of unique stars in the pattern (default: 24).")
    p.add_argument("--sep", type=float, default=8.0,
                   help="Approx. spatial separation between apertures in px (default: 8).")
    p.add_argument("--nap", type=int, default=None,
                   help="Total apertures expected (default: IRAF auto-detect).")
    p.add_argument("--dispaxis", type=int, default=1, choices=[1, 2],
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
    p.add_argument(
        "--step8-mode",
        default="ask",
        choices=["ask", "manual", "reuse"],
        help=(
            "Step-8 reference-star mode: ask (TTY prompt), manual (ecidentify), "
            "or reuse (ecreidentify from --step8-reference-thar)."
        ),
    )
    p.add_argument(
        "--step8-reference-thar",
        default=None,
        help=(
            "Path to already identified reference ThAr FITS for --step8-mode=reuse. "
            "A matching IRAF DB entry in ./database must exist."
        ),
    )
    p.add_argument(
        "--drift-log",
        default=None,
        help=(
            "CSV path for ecreidentify drift metrics (steps 8/9). "
            "Default: reidentify_drift_<NIGHT>_<SHOE>.csv when night/shoe are set, "
            "otherwise reidentify_drift.csv."
        ),
    )
    p.add_argument(
        "--step9-gate-mode",
        default="warn",
        choices=["off", "warn", "strict"],
        help=(
            "Step-9 quality gate behavior: off (disabled), warn (report failures), "
            "strict (skip refspec assignment for failed stars)."
        ),
    )
    p.add_argument(
        "--step9-min-found-frac",
        type=float,
        default=0.05,
        help="Step-9 minimum accepted ecreidentify found fraction (default: 0.05).",
    )
    p.add_argument(
        "--step9-min-fit-frac",
        type=float,
        default=0.05,
        help="Step-9 minimum accepted ecreidentify fit fraction (default: 0.05).",
    )
    p.add_argument(
        "--step9-max-rms",
        type=float,
        default=0.30,
        help="Step-9 maximum accepted ecreidentify RMS (default: 0.30).",
    )
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

        Late-step runs still need enough base context to reconstruct deterministic
        intermediate filenames from existing artifacts on disk.
    """
    steps = set(selected_steps)
    roles = set()

    if 1 in steps or 3 in steps:
        roles.add("quartz")
    if 2 in steps:
        roles.update(("quartz", "thar", "object", "twilight"))
    if 4 in steps:
        roles.update(("quartz", "thar", "object", "twilight"))
    if 5 in steps:
        roles.add("quartz")
    if 6 in steps:
        roles.update(("thar", "object"))

    return tuple(r for r in ("quartz", "thar", "object", "twilight") if r in roles)


def _normalize_step2_like_input(path):
    """Normalize a path to the corresponding step-2 '*-sl.fits' artifact."""
    if not path:
        return None
    lower = path.lower()
    if lower.endswith("-sl.fits"):
        return path
    if lower.endswith("-sl-f.fits"):
        return path[:-7] + ".fits"
    return stem(path) + "-sl.fits"


def expected_step2_outputs(quartz, thar, obj, twilight=None):
    """Return deterministic step-2 output paths for all roles."""
    outputs = {}
    if quartz:
        outputs["quartz_sl"] = _normalize_step2_like_input(quartz)
    if thar:
        outputs["thar_sl"] = _normalize_step2_like_input(thar)
    if obj:
        outputs["obj_sl"] = _normalize_step2_like_input(obj)
    if twilight:
        outputs["twilight_sl"] = _normalize_step2_like_input(twilight)
    return outputs


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


def write_infiles_from_directory(input_dir, infiles_path=None):
    """Create an image_processing-style infiles list from input_dir/*.fits."""
    fits_paths = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if not fits_paths:
        raise RuntimeError(f"No FITS files found in input directory: {input_dir}")

    out_path = infiles_path or os.path.join(input_dir, "infiles")
    with open(out_path, "w") as fh:
        for path in fits_paths:
            fh.write(os.path.basename(path) + "\n")
    print(f"  Preprocess infiles written: {out_path} ({len(fits_paths)} files)")
    return out_path


def run_image_preprocessing(args):
    """Run image_processing.py in args.input_dir before echelle reduction."""
    infiles_path = write_infiles_from_directory(args.input_dir, args.preprocess_infiles)
    script_path = os.path.join(os.path.dirname(__file__), "image_processing.py")
    cmd = [sys.executable, script_path, "--infiles", os.path.abspath(infiles_path)]

    if args.preprocess_object:
        cmd.extend(["--object", args.preprocess_object])
    if args.preprocess_bias:
        cmd.append("--bias")
    if args.preprocess_flat:
        cmd.extend(["--flat", args.preprocess_flat])

    print("  Launching preprocessing:")
    print("    " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=args.input_dir)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"image_processing.py failed with exit code {exc.returncode}"
        ) from exc


def prepare_quartz_reference_alias(quartz_path, meta_by_role,
                                   fallback_night=None, fallback_shoe=None):
    """Create/update a stable quartz alias used for aperture-sensitive steps."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=quartz_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    alias_name = f"quartz_trace_ref_{night}_{shoe}.fits"
    alias_path = os.path.join(os.path.dirname(quartz_path) or ".", alias_name)

    src_abs = os.path.abspath(quartz_path)
    dst_abs = os.path.abspath(alias_path)
    if src_abs == dst_abs:
        return alias_path

    needs_copy = (not os.path.exists(alias_path))
    if not needs_copy:
        needs_copy = os.path.getmtime(quartz_path) > os.path.getmtime(alias_path)
    if needs_copy:
        shutil.copy2(quartz_path, alias_path)
        print(f"  Quartz reference alias updated: {alias_path}")
    else:
        print(f"  Quartz reference alias reused: {alias_path}")
    return alias_path


def rewrite_quartz_db_image_identity(db_path, alias_quartz):
    """Rewrite IRAF aperture DB image tags so alias references resolve cleanly."""
    if not os.path.exists(db_path):
        return

    alias_base = stem(alias_quartz)

    with open(db_path, "r") as fh:
        lines = fh.readlines()

    changed = False
    rewritten = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("begin") and "aperture" in stripped:
            m = re.match(r"^(\s*begin\s+aperture\s+)\S+(\s+.*)$", line)
            if m:
                replacement = f"{m.group(1)}./{alias_base}{m.group(2)}\n"
                if line != replacement:
                    line = replacement
                    changed = True
        if stripped.startswith("image"):
            indent = line[:len(line) - len(stripped)]
            replacement = f"{indent}image\t./{alias_base}\n"
            if line != replacement:
                line = replacement
                changed = True
        rewritten.append(line)

    if changed:
        with open(db_path, "w") as fh:
            fh.writelines(rewritten)


def _trace_source_from_step2_like(quartz_path):
    """Return likely traced-quartz path when input is a step-2/4 quartz product."""
    if not quartz_path:
        return None
    lower = quartz_path.lower()
    if lower.endswith("-sl-f.fits"):
        return quartz_path[:-10] + ".fits"
    if lower.endswith("-sl.fits"):
        return quartz_path[:-7] + ".fits"
    return None


def ensure_quartz_trace_db_alias(source_quartz, alias_quartz):
    """Ensure alias quartz has aperture DB with alias-consistent image identity."""
    alias_candidates = quartz_trace_db_candidates(alias_quartz)
    existing_alias = [p for p in alias_candidates if os.path.exists(p)]

    source_db = None
    for candidate in quartz_trace_db_candidates(source_quartz):
        if os.path.exists(candidate):
            source_db = candidate
            break

    if source_db is None:
        traced_source = _trace_source_from_step2_like(source_quartz)
        if traced_source:
            for candidate in quartz_trace_db_candidates(traced_source):
                if os.path.exists(candidate):
                    source_db = candidate
                    break

    # Resume runs often pass --quartz as '*-sl.fits' (which has no trace DB).
    # Fall back to any existing alias/source trace DB and normalize identity.
    if source_db is None:
        for candidate in quartz_trace_db_candidates(alias_quartz):
            if os.path.exists(candidate):
                source_db = candidate
                break
    if source_db is None and existing_alias:
        source_db = existing_alias[0]
    if source_db is None:
        return None

    template_db = source_db
    if os.path.abspath(source_db) in {os.path.abspath(p) for p in alias_candidates if os.path.lexists(p)}:
        template_db = os.path.join("database", f"._ap_source_{stem(alias_quartz)}.tmp")
        shutil.copy2(source_db, template_db)

    os.makedirs("database", exist_ok=True)

    # Ensure all common IRAF DB naming variants exist for the alias and point
    # to alias image identity (IRAF checks DB image tags during extraction).
    for target in alias_candidates:
        if os.path.exists(target) and os.path.islink(target):
            os.unlink(target)

        needs_refresh = (not os.path.exists(target))
        if not needs_refresh:
            try:
                needs_refresh = os.path.getmtime(source_db) > os.path.getmtime(target)
            except OSError:
                needs_refresh = True

        if needs_refresh:
            shutil.copy2(template_db, target)

        rewrite_quartz_db_image_identity(target, alias_quartz)

    if template_db != source_db and os.path.exists(template_db):
        os.remove(template_db)

    # Return the preferred candidate if present, otherwise any existing alias.
    for target in alias_candidates:
        if os.path.exists(target):
            return target
    for target in existing_alias:
        if os.path.exists(target):
            return target
    return None


def wavelength_db_candidates(thar_path):
    """Return likely IRAF wavelength-database paths for a ThAr spectrum."""
    db_dir = "./database"
    base_stem = stem(thar_path)
    base_name = os.path.basename(thar_path)
    candidates = [
        f"{db_dir}/ec{base_stem}",
        f"{db_dir}/ec.{base_stem}",
        f"{db_dir}/ec_{base_stem}",
        f"{db_dir}/ec{base_name}",
        f"{db_dir}/ec.{base_name}",
        f"{db_dir}/ec_{base_name}",
        # Compatibility: some historical runs produce ec.ec* style names.
        f"{db_dir}/ec.ec{base_stem}",
        f"{db_dir}/ec.ec{base_name}",
    ]

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def resolve_existing_wavelength_db(thar_path):
    """Return the first existing wavelength DB path and full candidate list."""
    candidates = wavelength_db_candidates(thar_path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, candidates
    return None, candidates


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


def default_geometry_path(meta_by_role, reference_path=None,
                          fallback_night=None, fallback_shoe=None):
    """Return default Step-5 geometry filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    return f"geometry_{night}_{shoe}.json"


def _evaluate_iraf_curve(curve_values, ncols):
    """Evaluate IRAF aperture curve metadata onto detector x-pixel coordinates."""
    if len(curve_values) < 4:
        return None

    fit_type = int(round(curve_values[0]))
    ncoeff = int(round(curve_values[1]))
    xmin = float(curve_values[2])
    xmax = float(curve_values[3])
    if len(curve_values) < 4 + ncoeff:
        return None
    coeffs = np.array(curve_values[4: 4 + ncoeff], dtype=float)
    if len(coeffs) == 0 or xmax == xmin:
        return None

    lo = max(0.0, min(xmin, xmax))
    hi = min(float(ncols - 1), max(xmin, xmax))
    if hi - lo < 1.0:
        return None

    x_pixels = np.linspace(lo, hi, 220)
    xnorm = 2.0 * (x_pixels - xmin) / (xmax - xmin) - 1.0

    if fit_type == 2:
        delta = np.polynomial.chebyshev.chebval(xnorm, coeffs)
    elif fit_type == 1:
        delta = np.polynomial.legendre.legval(xnorm, coeffs)
    else:
        return None
    return x_pixels, delta


def _load_quartz_trace_details(quartz_path):
    """Load per-aperture geometry details from IRAF aperture database."""
    try:
        from aperture_preview import find_iraf_aperture_db
    except ImportError:
        return {"db_path": None, "entries": []}

    if not quartz_path or not os.path.exists(quartz_path):
        return {"db_path": None, "entries": []}

    data = fits.getdata(quartz_path)
    ncols = int(data.shape[1])
    db_path = find_iraf_aperture_db(quartz_path)
    if not db_path or not os.path.exists(db_path):
        return {"db_path": None, "entries": []}

    with open(db_path, "r") as fh:
        lines = fh.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("begin"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 4 or parts[1] != "aperture":
            i += 1
            continue

        try:
            ap_num = int(parts[3])
        except ValueError:
            i += 1
            continue

        center_y = None
        lower = None
        upper = None
        curve_values = None

        j = i + 1
        while j < len(lines):
            inner = lines[j].strip()
            if inner.startswith("begin"):
                break

            if inner.startswith("center"):
                toks = inner.split()
                if len(toks) >= 3:
                    try:
                        center_y = float(toks[2])
                    except ValueError:
                        center_y = None
            elif inner.startswith("low"):
                toks = inner.split()
                try:
                    lower = float(toks[-1])
                except (ValueError, IndexError):
                    lower = None
            elif inner.startswith("high"):
                toks = inner.split()
                try:
                    upper = float(toks[-1])
                except (ValueError, IndexError):
                    upper = None
            elif inner.startswith("curve"):
                toks = inner.split()
                try:
                    nvals = int(toks[1])
                except (ValueError, IndexError):
                    nvals = 0

                vals = []
                k = j + 1
                while k < len(lines) and len(vals) < nvals:
                    probe = lines[k].strip()
                    if not probe:
                        k += 1
                        continue
                    if probe.startswith("begin"):
                        break
                    try:
                        vals.append(float(probe))
                    except ValueError:
                        pass
                    k += 1
                if len(vals) == nvals:
                    curve_values = vals
            j += 1

        trace_xy = None
        if curve_values is not None and center_y is not None:
            evaluated = _evaluate_iraf_curve(curve_values, ncols)
            if evaluated is not None:
                x_pixels, delta = evaluated
                trace_xy = (x_pixels, center_y + delta)

        entries.append(
            {
                "aperture": ap_num,
                "center_y": center_y,
                "lower": lower,
                "upper": upper,
                "trace_coeffs": curve_values,
                "trace_xy": trace_xy,
            }
        )
        i = j

    entries.sort(key=lambda x: x["aperture"])
    return {"db_path": db_path, "entries": entries}


def _fit_bundle_center_parabolas(aperture_records, ncols):
    """Fit quadratic bundle-center loci y_b(x) from per-aperture traces."""
    bundles = {}
    x_eval = np.linspace(0.0, float(ncols - 1), 50)

    for rec in aperture_records:
        bundle = rec.get("bundle")
        trace_xy = rec.get("trace_xy")
        if bundle is None or trace_xy is None:
            continue
        x_trace, y_trace = trace_xy
        if len(x_trace) < 2:
            continue

        interp = np.full_like(x_eval, np.nan, dtype=float)
        mask = (x_eval >= float(np.min(x_trace))) & (x_eval <= float(np.max(x_trace)))
        if np.any(mask):
            interp[mask] = np.interp(x_eval[mask], x_trace, y_trace)
        bundles.setdefault(int(bundle), []).append(interp)

    center_models = []
    mean_curves = {}
    for bundle in sorted(bundles):
        stack = np.array(bundles[bundle], dtype=float)
        mean_curve = np.nanmean(stack, axis=0)
        good = np.isfinite(mean_curve)
        if np.count_nonzero(good) < 3:
            continue
        coeff = np.polyfit(x_eval[good], mean_curve[good], deg=2)
        center_models.append(
            {
                "bundle": bundle,
                "coefficients": [float(coeff[0]), float(coeff[1]), float(coeff[2])],
                "n_samples": int(np.count_nonzero(good)),
            }
        )
        mean_curves[bundle] = mean_curve

    spacing_models = []
    sorted_bundles = sorted(mean_curves)
    for left, right in zip(sorted_bundles[:-1], sorted_bundles[1:]):
        delta = mean_curves[right] - mean_curves[left]
        good = np.isfinite(delta)
        if np.count_nonzero(good) < 3:
            continue
        coeff = np.polyfit(x_eval[good], delta[good], deg=2)
        spacing_models.append(
            {
                "bundle_left": int(left),
                "bundle_right": int(right),
                "coefficients": [float(coeff[0]), float(coeff[1]), float(coeff[2])],
                "n_samples": int(np.count_nonzero(good)),
            }
        )

    return center_models, spacing_models


def save_step5_geometry(path, centers, pattern, quartz_path, meta_by_role):
    """Persist full Step-5 geometry metadata for downstream modeling."""
    night, shoe = infer_night_shoe(meta_by_role, reference_path=quartz_path)
    centers = np.asarray(centers, dtype=float)
    pattern = np.asarray(pattern, dtype=int)

    trace_details = _load_quartz_trace_details(quartz_path)
    by_ap = {entry["aperture"]: entry for entry in trace_details["entries"]}

    order_index_by_ap = {}
    for star in sorted(set(int(s) for s in pattern if int(s) > 0)):
        ap_indices = list(np.where(pattern == star)[0] + 1)
        ap_sorted = sorted(ap_indices, key=lambda ap: float(centers[ap - 1]))
        for idx, ap in enumerate(ap_sorted, start=1):
            order_index_by_ap[ap] = idx

    sort_idx = np.argsort(centers)
    gap_prev = np.full(len(centers), np.nan, dtype=float)
    gap_next = np.full(len(centers), np.nan, dtype=float)
    sorted_centers = centers[sort_idx]
    if len(sorted_centers) >= 2:
        deltas = np.diff(sorted_centers)
        for pos, ap_i in enumerate(sort_idx):
            if pos > 0:
                gap_prev[ap_i] = float(deltas[pos - 1])
            if pos < len(sorted_centers) - 1:
                gap_next[ap_i] = float(deltas[pos])

    aperture_records = []
    for ap_idx in range(len(pattern)):
        aperture = ap_idx + 1
        star = int(pattern[ap_idx])
        bundle = ((star - 1) // 4 + 1) if star > 0 else None
        star_in_bundle = ((star - 1) % 4 + 1) if star > 0 else None
        trace = by_ap.get(aperture, {})

        aperture_records.append(
            {
                "aperture": aperture,
                "bundle": bundle,
                "star": star if star > 0 else None,
                "star_in_bundle": star_in_bundle,
                "order": order_index_by_ap.get(aperture),
                "y_center_ref": float(centers[ap_idx]),
                "trace_coeffs": trace.get("trace_coeffs"),
                "lower": trace.get("lower"),
                "upper": trace.get("upper"),
                "gap_prev": (None if not np.isfinite(gap_prev[ap_idx]) else float(gap_prev[ap_idx])),
                "gap_next": (None if not np.isfinite(gap_next[ap_idx]) else float(gap_next[ap_idx])),
                "status": "active" if star > 0 else "deleted",
                "trace_xy": trace.get("trace_xy"),
            }
        )

    ncols = int(fits.getdata(quartz_path).shape[1])
    bundle_center_models, bundle_spacing_models = _fit_bundle_center_parabolas(
        aperture_records,
        ncols,
    )

    serializable_apertures = []
    for rec in aperture_records:
        trace_xy = rec.pop("trace_xy")
        if trace_xy is not None:
            x_trace, y_trace = trace_xy
            rec["trace_x"] = [float(x) for x in x_trace.tolist()]
            rec["trace_y"] = [float(y) for y in y_trace.tolist()]
        else:
            rec["trace_x"] = None
            rec["trace_y"] = None
        serializable_apertures.append(rec)

    payload = {
        "schema_version": 1,
        "night": str(night),
        "shoe": str(shoe),
        "quartz_path": os.path.basename(quartz_path) if quartz_path else "",
        "quartz_db_path": trace_details.get("db_path") or "",
        "n_apertures": int(len(pattern)),
        "n_active_apertures": int(np.count_nonzero(pattern > 0)),
        "apertures": serializable_apertures,
        "bundle_center_parabolas": bundle_center_models,
        "bundle_spacing_parabolas": bundle_spacing_models,
    }

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  Geometry file saved   : {path}")


def plot_step5_geometry_png(geometry_path, output_png=None):
    """Render a static Step-5 geometry PNG for CCD/star inspection."""
    if not geometry_path or not os.path.exists(geometry_path):
        return None

    with open(geometry_path, "r") as fh:
        payload = json.load(fh)

    apertures = payload.get("apertures") or []
    bundle_center_parabolas = payload.get("bundle_center_parabolas") or []
    if not apertures:
        return None

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if output_png is None:
        output_png = os.path.splitext(geometry_path)[0] + "_ccd_star_geometry.png"

    xmins = []
    xmaxs = []
    for rec in apertures:
        x_vals = rec.get("trace_x")
        y_vals = rec.get("trace_y")
        if x_vals is None or y_vals is None:
            continue
        if len(x_vals) >= 2 and len(y_vals) >= 2 and len(x_vals) == len(y_vals):
            xmins.append(float(np.min(x_vals)))
            xmaxs.append(float(np.max(x_vals)))

    if xmins and xmaxs:
        x_min = float(np.min(xmins))
        x_max = float(np.max(xmaxs))
    else:
        x_min = 0.0
        x_max = 2047.0

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = 0.0, 2047.0

    x_mid = 0.5 * (x_min + x_max)
    x_plot = np.linspace(x_min, x_max, 512)
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(12, 7))
    star_y_values = {}

    for rec in apertures:
        star = rec.get("star")
        y_center = rec.get("y_center_ref")
        trace_x = rec.get("trace_x")
        trace_y = rec.get("trace_y")

        is_active = isinstance(star, int) and star > 0
        if is_active:
            color = cmap((int(star) - 1) % cmap.N)
            linestyle = "-"
            if y_center is not None:
                star_y_values.setdefault(int(star), []).append(float(y_center))
        else:
            color = "0.6"
            linestyle = "--"

        if (
            trace_x is not None
            and trace_y is not None
            and len(trace_x) >= 2
            and len(trace_y) >= 2
            and len(trace_x) == len(trace_y)
        ):
            ax.plot(trace_x, trace_y, color=color, linestyle=linestyle, linewidth=1.4, alpha=0.9)
        elif y_center is not None:
            y = float(y_center)
            ax.plot([x_min, x_max], [y, y], color=color, linestyle=linestyle, linewidth=1.2, alpha=0.85)

    for model in bundle_center_parabolas:
        coeff = model.get("coefficients")
        if not coeff or len(coeff) != 3:
            continue
        a, b, c = [float(v) for v in coeff]
        y_model = a * x_plot * x_plot + b * x_plot + c
        ax.plot(x_plot, y_model, color="k", linestyle=":", linewidth=1.0, alpha=0.8)

    for star in sorted(star_y_values):
        y_vals = np.asarray(star_y_values[star], dtype=float)
        y_med = float(np.median(y_vals))
        color = cmap((int(star) - 1) % cmap.N)
        ax.plot([x_mid], [y_med], marker="o", markersize=6, color=color,
                markeredgecolor="k", markeredgewidth=0.4)

    ax.set_xlabel("CCD X [pix]")
    ax.set_ylabel("CCD Y [pix]")
    ax.set_title("Step-5 Geometry: Aperture Traces and Bundle Centers")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"  Step-5 geometry PNG : {output_png}")
    return output_png


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


def _existing_quartz_db_realpaths(quartz_path, extra_db_path=None):
    """Return canonical realpaths for existing quartz aperture DB candidates."""
    realpaths = set()
    candidates = []
    if quartz_path:
        candidates.extend(quartz_trace_db_candidates(quartz_path))
    if extra_db_path:
        candidates.append(extra_db_path)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            realpaths.add(os.path.realpath(candidate))
    return realpaths


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
        map_quartz_name = os.path.basename(str(mapping.get("quartz_path", "")))
        map_db_realpaths = _existing_quartz_db_realpaths(
            mapping.get("quartz_path"),
            extra_db_path=mapping.get("quartz_db_path"),
        )
        run_db_realpaths = _existing_quartz_db_realpaths(quartz_path)
        map_is_alias_name = map_quartz_name.startswith("quartz_trace_ref_")
        if map_is_alias_name and run_db_realpaths:
            pass
        elif not map_db_realpaths.intersection(run_db_realpaths):
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


def default_star_geometry_path(meta_by_role, reference_path=None,
                               fallback_night=None, fallback_shoe=None):
    """Return default star-geometry filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    return f"star_geometry_{night}_{shoe}.json"


def save_star_geometry_table(path, pattern, quartz_path, obj_outputs, thar_outputs, meta_by_role):
    """Persist per-star geometry table with y_star and linked extraction outputs."""
    night, shoe = infer_night_shoe(meta_by_role, reference_path=quartz_path)
    pattern = np.asarray(pattern, dtype=int)

    trace_details = _load_quartz_trace_details(quartz_path)
    centers_by_ap = {}
    for entry in trace_details.get("entries", []):
        ap = int(entry.get("aperture", 0))
        center_y = entry.get("center_y")
        if ap > 0 and center_y is not None:
            centers_by_ap[ap] = float(center_y)

    stars = sorted(set(obj_outputs).intersection(set(thar_outputs)))
    records = []
    for star in stars:
        ap_indices = list(np.where(pattern == star)[0] + 1)
        order_centers = [centers_by_ap[ap] for ap in ap_indices if ap in centers_by_ap]
        if not order_centers:
            continue
        order_centers = sorted(float(v) for v in order_centers)
        y_star = float(np.median(order_centers))
        bundle = ((int(star) - 1) // 4) + 1
        records.append(
            {
                "star_id": int(star),
                "bundle": int(bundle),
                "y_star": y_star,
                "order_centers": order_centers,
                "thar_ec": thar_outputs[star],
                "object_ec": obj_outputs[star],
            }
        )

    payload = {
        "schema_version": 1,
        "night": str(night),
        "shoe": str(shoe),
        "quartz_path": os.path.basename(quartz_path) if quartz_path else "",
        "n_stars": len(records),
        "stars": records,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  Star geometry saved  : {path}")


def load_star_geometry_table(path):
    """Load star-geometry table from disk."""
    with open(path, "r") as fh:
        return json.load(fh)


def star_order_by_distance_from_reference(star_geometry, ref_star, available_stars):
    """Return stars ordered by increasing |y_star - y_ref| when geometry exists."""
    if not star_geometry:
        return sorted(available_stars)

    stars = star_geometry.get("stars", [])
    y_by_star = {}
    for rec in stars:
        try:
            y_by_star[int(rec["star_id"])] = float(rec["y_star"])
        except (KeyError, TypeError, ValueError):
            continue

    if ref_star not in y_by_star:
        return sorted(available_stars)

    y_ref = y_by_star[ref_star]
    with_y = [s for s in available_stars if s in y_by_star]
    without_y = [s for s in available_stars if s not in y_by_star]
    with_y_sorted = sorted(with_y, key=lambda s: (abs(y_by_star[s] - y_ref), s))
    return with_y_sorted + sorted(without_y)


def _expand_aperture_range_string(aperture_text):
    """Expand compact aperture ranges like '1-3,8' to sorted integer list."""
    result = []
    text = str(aperture_text or "").strip()
    if not text:
        return result

    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            try:
                left, right = token.split("-", 1)
                lo = int(left.strip())
                hi = int(right.strip())
            except ValueError:
                continue
            if lo <= hi:
                result.extend(range(lo, hi + 1))
            else:
                result.extend(range(hi, lo + 1))
        else:
            try:
                result.append(int(token))
            except ValueError:
                continue
    return sorted(set(result))


def _first_int_in_text(value):
    """Return first integer token found in value, otherwise None."""
    m = re.search(r"[-+]?\d+", str(value))
    if not m:
        return None
    return int(m.group(0))


def _replace_first_integer(value, new_int):
    """Replace the first integer token in value with new_int."""
    text = str(value)
    m = re.search(r"[-+]?\d+", text)
    if not m:
        return text
    return text[:m.start()] + str(int(new_int)) + text[m.end():]


def _read_apertures_from_geometry_file(geometry_path, target_star):
    """Read target star aperture numbers from persisted step-5 geometry."""
    if not geometry_path or not os.path.exists(geometry_path):
        return []

    try:
        with open(geometry_path, "r") as fh:
            payload = json.load(fh)
    except Exception:
        return []

    apertures = []
    for rec in payload.get("apertures", []):
        try:
            if int(rec.get("star")) != int(target_star):
                continue
            apertures.append(int(rec.get("aperture")))
        except (TypeError, ValueError):
            continue
    return sorted(set(apertures))


def _read_apertures_from_extraction_pairs(extraction_pairs_path, target_star):
    """Read target star apertures from extraction_pairs_<NIGHT>_<SHOE>.csv."""
    if not extraction_pairs_path or not os.path.exists(extraction_pairs_path):
        return []

    try:
        with open(extraction_pairs_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    if int(row.get("star")) != int(target_star):
                        continue
                except (TypeError, ValueError):
                    continue
                apertures = _expand_aperture_range_string(row.get("apertures", ""))
                if apertures:
                    return apertures
    except Exception:
        return []

    return []


def _read_apertures_from_apnum_cards(fits_path):
    """Read aperture numbers from APNUM* cards in an extracted multispec FITS."""
    if not fits_path or not os.path.exists(fits_path):
        return []

    try:
        hdr = fits.getheader(fits_path)
    except Exception:
        return []

    apertures = []
    for key in hdr.keys():
        if not re.match(r"APNUM\d+$", str(key)):
            continue
        raw = hdr[key]
        ap = _first_int_in_text(raw)
        if ap is None:
            ap = _first_int_in_text(str(key)[5:])
        if ap is not None:
            apertures.append(int(ap))

    return sorted(set(apertures))


def get_target_aperture_numbers(target_star, target_thar_path,
                                geometry_path=None, extraction_pairs_path=None):
    """Resolve target aperture numbers, preferring geometry mapping when available."""
    apertures = _read_apertures_from_geometry_file(geometry_path, target_star)
    source = "step5-geometry"
    if not apertures:
        apertures = _read_apertures_from_extraction_pairs(extraction_pairs_path, target_star)
        source = "extraction-pairs"
    if not apertures:
        apertures = _read_apertures_from_apnum_cards(target_thar_path)
        source = "APNUM"

    if not apertures:
        raise RuntimeError(
            f"Could not resolve aperture numbers for target star {int(target_star):02d}. "
            f"Tried geometry={geometry_path}, extraction_pairs={extraction_pairs_path}, "
            f"APNUM from {target_thar_path}."
        )
    return sorted(apertures), source


def _build_aperture_mapping(master_apertures, target_apertures, expected_count=4):
    """Build one-to-one mapping from master reference apertures to target apertures."""
    master = sorted(int(v) for v in master_apertures)
    target = sorted(int(v) for v in target_apertures)

    if len(master) != len(target):
        raise RuntimeError(
            "Master/target aperture count mismatch for temporary reference remapping: "
            f"master={master}, target={target}."
        )
    if expected_count is not None and len(target) != int(expected_count):
        raise RuntimeError(
            f"Expected {int(expected_count)} apertures for target remapping, "
            f"got {len(target)} for target={target}."
        )
    if len(set(master)) != len(master) or len(set(target)) != len(target):
        raise RuntimeError(
            f"Aperture mapping requires unique aperture IDs. master={master}, target={target}."
        )

    return {int(src): int(dst) for src, dst in zip(master, target)}


def _format_aperture_mapping(aperture_mapping):
    """Return compact human-readable mapping string (e.g. '1->5, 2->6')."""
    return ", ".join(
        f"{int(src)}->{int(dst)}"
        for src, dst in sorted(aperture_mapping.items())
    )


def _create_temporary_reference_fits(master_ref_thar, target_star):
    """Create a fresh per-target temporary FITS copy from master reference."""
    nonce = os.urandom(3).hex()
    tmp_name = f".tmp_ref_star{int(target_star):02d}_{os.getpid()}_{nonce}.fits"
    tmp_path = os.path.join(".", tmp_name)
    shutil.copy2(master_ref_thar, tmp_path)
    return tmp_path


def _create_temporary_target_fits(target_thar, target_star):
    """Create a fresh per-target temporary FITS copy from real target ThAr."""
    nonce = os.urandom(3).hex()
    tmp_name = f".tmp_tgt_star{int(target_star):02d}_{os.getpid()}_{nonce}.fits"
    tmp_path = os.path.join(".", tmp_name)
    shutil.copy2(target_thar, tmp_path)
    return tmp_path


def _resolve_master_reference_db(master_ref_thar):
    """Resolve the existing master wavelength DB record for a reference ThAr."""
    probes = [master_ref_thar, os.path.basename(master_ref_thar), stem(master_ref_thar)]
    for probe in probes:
        found_db, _candidates = resolve_existing_wavelength_db(probe)
        if found_db:
            return found_db
    raise RuntimeError(
        f"Could not resolve master wavelength DB record for reference ThAr: {master_ref_thar}"
    )


def _clone_temporary_reference_db(master_ref_thar, temp_ref_path):
    """Clone master wavelength DB record to temp reference DB entry."""
    master_db = _resolve_master_reference_db(master_ref_thar)
    temp_candidates = wavelength_db_candidates(temp_ref_path)
    if not temp_candidates:
        raise RuntimeError(f"No DB candidates available for temporary reference: {temp_ref_path}")

    temp_db = temp_candidates[0]
    os.makedirs(os.path.dirname(temp_db) or ".", exist_ok=True)
    shutil.copy2(master_db, temp_db)
    return temp_db, master_db, temp_candidates


def _wat2_keys_sorted(header):
    """Return WAT2_* header keys sorted by numeric suffix."""
    keys = [k for k in header.keys() if re.match(r"WAT2_\d+$", str(k))]
    return sorted(keys, key=lambda k: int(str(k).split("_")[1]))


def _renumber_wat2_payload(payload, aperture_mapping):
    """Renumber multispec WAT2 spec aperture IDs using aperture mapping."""
    def _replace_spec(match):
        content = match.group(2)
        old_ap = _first_int_in_text(content)
        if old_ap is None or old_ap not in aperture_mapping:
            return match.group(0)
        new_content = _replace_first_integer(content, aperture_mapping[old_ap])
        return f"{match.group(1)}{new_content}{match.group(3)}"

    return re.sub(r'(spec\d+\s*=\s*")([^\"]+)(")', _replace_spec, payload)


def _renumber_temporary_reference_fits(temp_ref_path, aperture_mapping):
    """Apply aperture remapping to APNUM* and WAT2 metadata in temp FITS copy."""
    with fits.open(temp_ref_path, mode="update") as hdul:
        hdr = hdul[0].header

        for key in list(hdr.keys()):
            if not re.match(r"APNUM\d+$", str(key)):
                continue
            raw_value = str(hdr[key])
            old_ap = _first_int_in_text(raw_value)
            if old_ap is None:
                old_ap = _first_int_in_text(str(key)[5:])
            if old_ap in aperture_mapping:
                hdr[key] = _replace_first_integer(raw_value, aperture_mapping[old_ap])

        wat_keys = _wat2_keys_sorted(hdr)
        if wat_keys:
            old_payload = "".join(str(hdr[k]) for k in wat_keys)
            new_payload = _renumber_wat2_payload(old_payload, aperture_mapping)
            if new_payload != old_payload:
                chunk_size = 68
                chunks = [new_payload[i:i + chunk_size] for i in range(0, len(new_payload), chunk_size)]
                if not chunks:
                    chunks = [""]
                for idx, chunk in enumerate(chunks, start=1):
                    hdr[f"WAT2_{idx:03d}"] = chunk
                for key in wat_keys:
                    if int(str(key).split("_")[1]) > len(chunks):
                        del hdr[key]


def _renumber_temporary_reference_db(temp_db_path, temp_ref_path, aperture_mapping):
    """Apply identical aperture remapping to temporary wavelength DB record."""
    tmp_image_token = iraf_spec_token(temp_ref_path)
    with open(temp_db_path, "r") as fh:
        lines = fh.readlines()

    rewritten = []
    for line in lines:
        raw = line.rstrip("\n")
        newline = "\n" if line.endswith("\n") else ""
        stripped = raw.strip()

        if stripped.startswith("image"):
            indent = raw[:len(raw) - len(raw.lstrip())]
            rewritten.append(f"{indent}image\t{tmp_image_token}{newline}")
            continue

        m_begin_ap = re.match(r"^(\s*begin\s+\S+\s+)(\S+)(\s+)(-?\d+)(\s*)$", raw)
        if m_begin_ap:
            old_ap = int(m_begin_ap.group(4))
            new_ap = aperture_mapping.get(old_ap, old_ap)
            rewritten.append(
                f"{m_begin_ap.group(1)}{tmp_image_token}{m_begin_ap.group(3)}{new_ap}{m_begin_ap.group(5)}{newline}"
            )
            continue

        m_begin = re.match(r"^(\s*begin\s+\S+\s+)(\S+)(\s*)$", raw)
        if m_begin:
            rewritten.append(
                f"{m_begin.group(1)}{tmp_image_token}{m_begin.group(3)}{newline}"
            )
            continue

        m_ap = re.match(r"^(\s*aperture\s+)(-?\d+)(\s*)$", raw)
        if m_ap:
            old_ap = int(m_ap.group(2))
            new_ap = aperture_mapping.get(old_ap, old_ap)
            rewritten.append(f"{m_ap.group(1)}{new_ap}{m_ap.group(3)}{newline}")
            continue

        rewritten.append(line)

    with open(temp_db_path, "w") as fh:
        fh.writelines(rewritten)


def _cleanup_temporary_reference_artifacts(temp_ref_path, temp_db_candidates):
    """Best-effort cleanup for per-target temporary reference artifacts."""
    removed = []
    failed = []
    cleanup_paths = [temp_ref_path] + list(dict.fromkeys(temp_db_candidates or []))
    for path in cleanup_paths:
        if not path or not os.path.lexists(path):
            continue
        try:
            os.remove(path)
            removed.append(path)
        except OSError as exc:
            failed.append(f"{path} ({exc})")
    return removed, failed


def _prepare_temp_reference_for_target(master_ref_thar, target_star, target_apertures):
    """Create per-target temporary reference FITS+DB pair and apply remapping."""
    master_apertures = _read_apertures_from_apnum_cards(master_ref_thar)
    if not master_apertures:
        raise RuntimeError(
            f"Could not read APNUM apertures from master reference ThAr: {master_ref_thar}"
        )

    aperture_mapping = _build_aperture_mapping(
        master_apertures,
        target_apertures,
        expected_count=4,
    )

    temp_ref_path = _create_temporary_reference_fits(master_ref_thar, target_star)
    temp_db_path = None
    temp_db_candidates = wavelength_db_candidates(temp_ref_path)
    master_db_path = None

    try:
        temp_db_path, master_db_path, temp_db_candidates = _clone_temporary_reference_db(
            master_ref_thar,
            temp_ref_path,
        )
        _renumber_temporary_reference_fits(temp_ref_path, aperture_mapping)
        _renumber_temporary_reference_db(temp_db_path, temp_ref_path, aperture_mapping)
    except Exception:
        _cleanup_temporary_reference_artifacts(temp_ref_path, temp_db_candidates)
        raise

    return {
        "target_star": int(target_star),
        "master_apertures": sorted(master_apertures),
        "target_apertures": sorted(int(v) for v in target_apertures),
        "aperture_mapping": aperture_mapping,
        "temp_ref_path": temp_ref_path,
        "temp_db_path": temp_db_path,
        "master_db_path": master_db_path,
        "cleanup_db_candidates": temp_db_candidates,
    }


def _prepare_temp_target_for_reidentify(master_ref_thar, target_thar, target_star, target_apertures):
    """Create per-target temporary target FITS and remap target apertures to master apertures."""
    master_apertures = _read_apertures_from_apnum_cards(master_ref_thar)
    if not master_apertures:
        raise RuntimeError(
            f"Could not read APNUM apertures from master reference ThAr: {master_ref_thar}"
        )

    target_to_master = _build_aperture_mapping(
        target_apertures,
        master_apertures,
        expected_count=4,
    )
    master_to_target = {int(dst): int(src) for src, dst in target_to_master.items()}

    temp_target_path = _create_temporary_target_fits(target_thar, target_star)
    cleanup_db_candidates = wavelength_db_candidates(temp_target_path)

    try:
        _renumber_temporary_reference_fits(temp_target_path, target_to_master)
    except Exception:
        _cleanup_temporary_reference_artifacts(temp_target_path, cleanup_db_candidates)
        raise

    return {
        "target_star": int(target_star),
        "target_apertures": sorted(int(v) for v in target_apertures),
        "master_apertures": sorted(int(v) for v in master_apertures),
        "target_to_master": target_to_master,
        "master_to_target": master_to_target,
        "temp_target_path": temp_target_path,
        "cleanup_db_candidates": cleanup_db_candidates,
    }


def _transfer_reviewed_temp_target_solution_to_real_target(
    temp_target_path,
    real_target_thar,
    master_to_target,
):
    """Transfer reviewed temp-target wavelength DB content back to real target identity."""
    temp_db_path, temp_candidates = resolve_existing_wavelength_db(temp_target_path)
    if not temp_db_path:
        raise RuntimeError(
            "Could not locate temporary target wavelength DB record after review. "
            f"Checked: {', '.join(temp_candidates)}"
        )

    real_db_candidates = wavelength_db_candidates(real_target_thar)
    if not real_db_candidates:
        raise RuntimeError(f"No DB candidates available for real target: {real_target_thar}")

    real_db_path = real_db_candidates[0]
    os.makedirs(os.path.dirname(real_db_path) or ".", exist_ok=True)
    shutil.copy2(temp_db_path, real_db_path)

    # Rewrite to real target identity and restore real target aperture numbering.
    _renumber_temporary_reference_db(real_db_path, real_target_thar, master_to_target)

    for alias_probe in (
        real_target_thar,
        os.path.basename(real_target_thar),
        stem(real_target_thar),
        iraf_spec_token(real_target_thar),
    ):
        ensure_wavelength_db_aliases(alias_probe, real_db_path)

    return {
        "temp_db_path": temp_db_path,
        "real_db_path": real_db_path,
        "real_db_candidates": real_db_candidates,
    }


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

    if args.step9_min_found_frac < 0.0 or args.step9_min_found_frac > 1.0:
        sys.exit("ERROR: --step9-min-found-frac must be within [0, 1].")
    if args.step9_min_fit_frac < 0.0 or args.step9_min_fit_frac > 1.0:
        sys.exit("ERROR: --step9-min-fit-frac must be within [0, 1].")
    if args.step9_max_rms < 0.0:
        sys.exit("ERROR: --step9-max-rms must be non-negative.")

    try:
        selected_steps = selected_steps_from_args(args)
        required_roles = required_roles_for_steps(selected_steps)
    except Exception as exc:
        sys.exit(f"ERROR parsing step range: {exc}")

    if args.run_preprocess:
        if 1 in selected_steps:
            section_banner("Preprocessing prelude (image_processing.py)")
            try:
                run_image_preprocessing(args)
            except Exception as exc:
                sys.exit(f"ERROR preprocessing: {exc}")
        else:
            print("  [info] --run-preprocess requested, but start-step > 1; skipping preprocessing.")

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
    quartz_ref = None
    if args.quartz_reference:
        try:
            quartz_ref = args.quartz_reference
            require_existing(quartz_ref, "--quartz-reference")
            if quartz and os.path.abspath(quartz) != os.path.abspath(quartz_ref):
                ensure_quartz_trace_db_alias(quartz, quartz_ref)
        except Exception as exc:
            sys.exit(f"ERROR preparing explicit quartz reference: {exc}")
    elif quartz and any(s in selected_steps for s in (1, 5, 6)):
        # Default behavior: use quartz directly as the trace reference.
        quartz_ref = quartz

    step2_expected = expected_step2_outputs(quartz, thar, obj, twilight)

    print("\n" + "="*72)
    print("  McDonald echelle reduction pipeline")
    print("="*72)
    for label, val in [
            ("quartz",   quartz if quartz else "<not required>"),
            ("quartz_ref", quartz_ref if quartz_ref else "<not required>"),
            ("thar",     thar if thar else "<not required>"),
            ("object",   obj if obj else "<not required>"),
            ("twilight", twilight if twilight else "<not required>"),
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

    drift_log_path = None
    if any(s in selected_steps for s in (8, 9)):
        if args.drift_log:
            drift_log_path = args.drift_log
        elif args.night and args.shoe:
            drift_log_path = f"reidentify_drift_{args.night}_{str(args.shoe).upper()}.csv"
        else:
            drift_log_path = "reidentify_drift.csv"
        if os.path.exists(drift_log_path):
            os.remove(drift_log_path)
        print(f"  Reidentify drift log : {drift_log_path}")

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
            n_ap    = n_ap,
        )
        if quartz_ref:
            ensure_quartz_trace_db_alias(quartz, quartz_ref)
    elif any(s in selected_steps for s in (2, 6)):
        try:
            if 2 in selected_steps:
                require_quartz_trace_db(quartz, "step 2")
            if 6 in selected_steps:
                if not quartz_ref:
                    raise RuntimeError(
                        "Step 6 requires a quartz trace reference. Provide --quartz "
                        "or --quartz-reference, or run steps 1-4 first."
                    )
                if quartz and os.path.abspath(quartz) != os.path.abspath(quartz_ref):
                    ensure_quartz_trace_db_alias(quartz, quartz_ref)
                require_quartz_trace_db(quartz_ref, "step 6")
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
            if 6 in selected_steps and not quartz_ref:
                raise RuntimeError(
                    "Step 6 requires a quartz reference file with trace database. "
                    "Provide --quartz <quartz_file> or --quartz-reference <reference_fits>, "
                    "or run steps 1-4 first."
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
        preview_quartz = quartz_ref
        if not preview_quartz:
            sys.exit(
                "ERROR dependency check: step 5 requires a quartz reference image. "
                "Provide --quartz or --quartz-reference, or run steps 1-4 first."
            )
        try:
            if quartz and os.path.abspath(quartz) != os.path.abspath(preview_quartz):
                ensure_quartz_trace_db_alias(quartz, preview_quartz)
            require_quartz_trace_db(preview_quartz, "step 5")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

        map_path = args.affiliation_map or default_affiliation_map_path(
            meta_by_role,
            reference_path=preview_quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
        )
        geometry_path = default_geometry_path(
            meta_by_role,
            reference_path=preview_quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
        )
        preview_out = os.path.splitext(map_path)[0] + "_preview_map.txt"

        centers, pattern = run_aperture_preview(
            preview_quartz,
            n_stars = args.nstars,
            sep     = args.sep,
            preview_out=preview_out,
        )
        state["pattern"] = pattern
        try:
            save_affiliation_map(map_path, pattern, preview_quartz, meta_by_role)
            save_step5_geometry(geometry_path, centers, pattern, preview_quartz, meta_by_role)
            try:
                plot_step5_geometry_png(geometry_path)
            except Exception as exc:
                print(f"  WARNING step-5 geometry PNG failed: {exc}")
        except Exception as exc:
            sys.exit(f"ERROR saving step-5 products: {exc}")

        print(f"  Preview mapping saved: {preview_out}")
        print(f"  Step-5 PNG preview   : {os.path.splitext(preview_out)[0]}_aperture_preview.png")
        print(f"  Step-5 PNG overlay   : {os.path.splitext(preview_out)[0]}_aperture_overlay.png")
        print(f"\n  Total apertures in accepted mapping: {len(pattern)}")
    elif 6 in selected_steps:
        map_path = args.affiliation_map or default_affiliation_map_path(
            meta_by_role,
            reference_path=quartz_ref or quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
        )
        if os.path.exists(map_path):
            try:
                mapping = load_affiliation_map(map_path)
                state["pattern"] = validate_affiliation_map(
                    mapping,
                    quartz_ref or quartz,
                    meta_by_role,
                )
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
            require_quartz_trace_db(quartz_ref, "step 6")
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
            quartz  = quartz_ref,
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
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            pair_index = f"extraction_pairs_{night}_{shoe}.csv"
            write_extraction_pairs_index(pair_index, obj_outputs, thar_outputs, state["pattern"])

            star_geometry_path = default_star_geometry_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            save_star_geometry_table(
                star_geometry_path,
                state["pattern"],
                quartz_ref or quartz,
                obj_outputs,
                thar_outputs,
                meta_by_role,
            )
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

    # ── 8. Reference-star wavelength setup (manual/reuse) ───────────────────
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
            coordlist=args.coordlist,
            step8_mode=args.step8_mode,
            step8_reference_thar=args.step8_reference_thar,
            drift_log_path=drift_log_path,
        )
        state["ref_star"] = ref_star

    # ── 9. Automatic line-ID propagation (ecreidentify + review) ─
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
        step5_geometry_path = None
        extraction_pairs_path = None

        star_geometry = None
        try:
            night_for_step9, shoe_for_step9 = infer_night_shoe(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            step5_geometry_path = default_geometry_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            candidate_pair_index = f"extraction_pairs_{night_for_step9}_{shoe_for_step9}.csv"
            if os.path.exists(candidate_pair_index):
                extraction_pairs_path = candidate_pair_index

            star_geometry_path = default_star_geometry_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
            )
            if os.path.exists(star_geometry_path):
                try:
                    star_geometry = load_star_geometry_table(star_geometry_path)
                    print(f"  Loaded star geometry: {star_geometry_path}")
                except Exception as exc:
                    print(f"  [warn] Could not load star geometry '{star_geometry_path}': {exc}")
            else:
                print(
                    f"  [warn] Star geometry not found ({star_geometry_path}); "
                    "falling back to numeric star order."
                )
        except Exception as exc:
            print(f"  [warn] Could not resolve star geometry path: {exc}")
        
        # If step 8 didn't run but we're running step 9, verify ref star has solution
        if 8 not in selected_steps:
            ref_thar = thar_outputs[ref_star]
            found_db, db_candidates = resolve_existing_wavelength_db(ref_thar)
            if not found_db:
                sys.exit(
                    f"ERROR: step 9 requires wavelength solution from step 8. "
                    f"Run step 8 first on reference star {ref_star:02d}. "
                    f"(Checked: {', '.join(db_candidates)})"
                )
        
        id_assigned_outputs = auto_wavelength_propagation(
            crr2_outputs, thar_outputs, ref_star,
            coordlist=args.coordlist,
            drift_log_path=drift_log_path,
            step9_gate_mode=args.step9_gate_mode,
            step9_min_found_frac=args.step9_min_found_frac,
            step9_min_fit_frac=args.step9_min_fit_frac,
            step9_max_rms=args.step9_max_rms,
            star_geometry=star_geometry,
            step5_geometry_path=step5_geometry_path,
            extraction_pairs_path=extraction_pairs_path,
        )
        state["id_assigned_outputs"] = id_assigned_outputs

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
    if obj_outputs or thar_outputs or state.get("crr2_outputs") or state.get("id_assigned_outputs"):
        print("")
        for s in sorted(set(list(obj_outputs) + list(thar_outputs))):
            if s in obj_outputs:
                print(f"  Star {s:02d}  object    : {obj_outputs[s]}")
            if s in thar_outputs:
                print(f"         thar      : {thar_outputs[s]}")
            if s in state.get("crr2_outputs", {}):
                print(f"         crr2      : {state['crr2_outputs'][s]}")
            if s in state.get("id_assigned_outputs", {}):
                print(f"         line IDs  : {state['id_assigned_outputs'][s]}")
        if state.get("id_assigned_outputs"):
            print(
                "\n  Step 9 complete: line IDs propagated/reviewed and assigned with refspec."
            )
        elif obj_outputs:
            print(
                "\n  Next steps for each star:\n"
                "    Step 7  lineclean         ->  *_star<N>_ec-crr2.fits\n"
                "    Step 8  ecidentify on ThAr, refspec assignment (in-place)\n"
                "    Step 9  ecreidentify + review + refspec assignment (in-place)\n"
            )


if __name__ == "__main__":
    main()