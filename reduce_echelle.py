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
                – each extracted object+ThAr pair is immediately renumbered to local
                    apertures 1..N (normally 1..4)
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
            – The solution is built on the local apertures from step 6 and saved/used
                in the IRAF database for steps 9-11
  9.  Automatic line-ID propagation + required review  (ecreidentify to remaining stars)
            a. ecreidentify (automatic)   propagates line IDs from ref star to others
                 directly on each real target ThAr
            b. review      inspect reidentified ThAr line IDs (interactive when TTY)
      – outputs: no new FITS files in this section
  10. refspec assignment to CR-cleaned object spectra
      – runs on *_ec-crr2.fits using reviewed real-target ThAr identities
      – skips stars without a valid identified/reviewed ThAr solution
      – workflow: run step 8 (interactive), then step 9 (automatic + review), then step 10
  11. dispcor wavelength linearization of refspec-assigned object spectra
      – runs only for stars that successfully passed step 10
      – outputs: *_ec-crr2-dc.fits

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
import hashlib
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

from file_handler import (
    load_image_type_overrides,
    normalize_header_value,
    save_image_type_overrides,
)
from naming_helper import (
    fits_stem,
    normalize_step2_like_input,
    step11_dispcor,
    step2_scattered,
    step3_master_flat,
    step3_median,
    step4_flat_corrected,
    step6_star_extract,
    step7_crr2,
)


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
    return fits_stem(path)


def _safe_token(value):
    """Return a filename-safe token from arbitrary metadata text."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "-", text)
    return text or "na"


def _first_nonblank(values):
    """Return first non-empty string from values, otherwise empty string."""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_token_text(value):
    """Normalize free-form text for stable tokenization."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_wavelength_identity(path_or_token):
    """Return canonical IRAF wavelength identity token (bare root, no .fits)."""
    text = str(path_or_token or "").strip().strip("\"'")
    if not text:
        return ""

    # Strip IRAF section/selection syntax, then canonicalize to basename root.
    text = text.split("[", 1)[0].strip()
    text = os.path.basename(text)
    if text.startswith("./"):
        text = text[2:]
    if text.lower().endswith(".fits"):
        text = text[:-5]
    return text


def iraf_spec_token(path):
    """Return canonical IRAF spectroscopy/database image token (bare root)."""
    return canonical_wavelength_identity(path)


def quartz_reference_token(reference_quartz):
    """Return canonical IRAF apall references token for quartz traces.

    The leading dot keeps IRAF DB naming aligned with database/ap.<stem>.
    """
    return f".{stem(reference_quartz)}"


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


def resolve_processing_dirs(input_dir):
    """Resolve raw night directory and proc working directory."""
    abs_input = os.path.abspath(input_dir)
    if os.path.basename(abs_input) == 'proc':
        proc_dir = abs_input
        raw_dir = os.path.dirname(proc_dir)
    else:
        raw_dir = abs_input
        proc_dir = os.path.join(raw_dir, 'proc')
    return raw_dir, proc_dir


def anchor_proc_workdir(proc_dir):
    """Anchor process and IRAF working directory to proc_dir."""
    proc_abs = os.path.abspath(proc_dir)
    os.makedirs(proc_abs, exist_ok=True)
    os.chdir(proc_abs)
    iraf.cd(proc_abs)
    return proc_abs


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
        "IMAGE_TYPE": str(hdr.get("IMAGE_TYPE", "")).strip().upper(),
        "NIGHT": str(hdr["NIGHT"]),
        "SHOE": str(hdr["SHOE"]),
        "PLATE": str(hdr.get("PLATE", "")).strip(),
        "STACKED": bool(hdr.get("STACKED", False)),
        "STACKTYP": str(hdr.get("STACKTYP", hdr.get("STACKTYPE", ""))),
        "STACKMOD": str(hdr.get("STACKMOD", "")),
    }


def is_stacked_product(meta):
    """Return True if metadata indicates a final stacked product."""
    if meta.get("STACKED", False):
        return True

    stype = str(meta.get("STACKTYP", "")).strip().lower()
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

    # Step 6+ extracted outputs and downstream derivatives.
    # Matches: *_starNN_ec.fits, *_starNN_ec-crr2.fits, *_starNN_ec-crr2-dc.fits
    if re.search(r"_star\d+_ec(?:-crr2(?:-dc)?)?\.fits$", name):
        return True

    return False


def classify_role(meta):
    """Classify a frame role from FITS header metadata."""
    image_type = str(meta.get("IMAGE_TYPE", "")).strip().upper()
    if image_type == "SCIENCE":
        return "object"
    if image_type == "QUARTZ":
        return "quartz"
    if image_type == "LAMP":
        return "thar"
    if image_type == "TWILIGHT":
        return "twilight"
    if image_type in {"FIBERMAP", "UNKNOWN", "BIAS", "MASTER_BIAS", "DARK", "DARK_MASTER"}:
        return None

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


ROLE_TO_IMAGE_TYPE = {
    "object": "SCIENCE",
    "quartz": "QUARTZ",
    "thar": "LAMP",
    "twilight": "TWILIGHT",
    "dark": "DARK",
}

_MANUAL_MENU_SELECTION = "__MANUAL_MENU_SELECTION__"


def _normalized_signature(exptype_norm, object_norm):
    return f"exptype={exptype_norm}|object={object_norm}"


def _role_from_image_type(image_type):
    value = str(image_type or "").strip().upper()
    mapping = {
        "SCIENCE": "object",
        "QUARTZ": "quartz",
        "LAMP": "thar",
        "TWILIGHT": "twilight",
        "DARK": "dark",
        "DARK_MASTER": "dark",
    }
    return mapping.get(value)


def _candidate_roles_for_meta(meta):
    roles = set()

    typed = _role_from_image_type(meta.get("IMAGE_TYPE", ""))
    if typed:
        roles.add(typed)

    inferred = classify_role(meta)
    if inferred:
        roles.add(inferred)

    exptype_norm = str(meta.get("EXPTYPE_NORM", ""))
    object_norm = str(meta.get("OBJECT_NORM", ""))
    text = f"{exptype_norm} {object_norm}".strip()

    if "dark" in text:
        roles.add("dark")
    if any(token in text for token in ("twilight", "dawn sky")):
        roles.add("twilight")
    if any(token in text for token in ("quartz", "flat", "domeflat", "dome flat")):
        roles.add("quartz")
    if any(token in text for token in ("thar", "thne", "tharne", "lamp", "arc", "comp")):
        roles.add("thar")
    if any(token in text for token in ("object", "science", " sci", "target", "star")):
        roles.add("object")

    return roles


def _is_dark_master_candidate(meta):
    """Return True when metadata represents a DARK_MASTER product."""
    image_type = str(meta.get("IMAGE_TYPE", "")).strip().upper()
    if image_type == "DARK_MASTER":
        return True

    exptype_norm = str(meta.get("EXPTYPE_NORM", "")).lower()
    object_norm = str(meta.get("OBJECT_NORM", "")).lower()
    text = f"{exptype_norm} {object_norm}".strip()
    return "dark master" in text or "master dark" in text


def _candidate_roots(args):
    roots = []
    for attr in ("raw_input_dir", "proc_dir", "input_dir"):
        value = getattr(args, attr, None)
        if not value:
            continue
        abspath = os.path.abspath(value)
        if os.path.isdir(abspath) and abspath not in roots:
            roots.append(abspath)
    return roots


def iter_candidate_fits_paths(args):
    """Yield unique FITS paths from raw/proc run roots."""
    seen = set()
    for root in _candidate_roots(args):
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if not fname.lower().endswith(".fits"):
                    continue
                path = os.path.abspath(os.path.join(dirpath, fname))
                if path in seen:
                    continue
                seen.add(path)
                yield path


def _infer_scope_from_resolved(args, resolved):
    allow_mixed_nights = bool(getattr(args, "allow_mixed_nights", False))
    night = None if allow_mixed_nights else (str(args.night) if args.night else None)
    shoe = str(args.shoe).upper() if args.shoe else None
    plate = str(args.plate) if args.plate else None

    if night and shoe and plate:
        return night, shoe, plate

    nights = set()
    shoes = set()
    plates = set()

    for path in resolved.values():
        try:
            meta = read_required_metadata(path)
        except Exception:
            continue
        nights.add(str(meta.get("NIGHT", "")))
        shoes.add(str(meta.get("SHOE", "")).upper())
        plate_val = str(meta.get("PLATE", "")).strip()
        if plate_val:
            plates.add(plate_val)

    if (not allow_mixed_nights) and night is None and len(nights) == 1:
        night = next(iter(nights))
    if shoe is None and len(shoes) == 1:
        shoe = next(iter(shoes))
    if plate is None and len(plates) == 1:
        plate = next(iter(plates))

    return night, shoe, plate


def collect_header_inventory(args, resolved, required_roles=None):
    """Collect run-local FITS header combinations for unresolved-role prompts."""
    required_roles = set(required_roles or ())
    allow_mixed_nights = bool(getattr(args, "allow_mixed_nights", False))
    override_path_arg = getattr(args, "image_type_override_path", None)
    overrides, override_path = load_image_type_overrides(override_path_arg)

    scope_night, scope_shoe, scope_plate = _infer_scope_from_resolved(args, resolved)
    candidates = []
    skipped_missing = 0

    for path in iter_candidate_fits_paths(args):
        try:
            hdr = fits.getheader(path)
        except Exception:
            continue

        required = ("OBJECT", "EXPTYPE", "NIGHT", "SHOE")
        if any(k not in hdr for k in required):
            skipped_missing += 1
            continue

        exptype_raw = str(hdr.get("EXPTYPE", ""))
        object_raw = str(hdr.get("OBJECT", ""))
        night = str(hdr.get("NIGHT", ""))
        shoe = str(hdr.get("SHOE", "")).upper()
        plate = str(hdr.get("PLATE", "")).strip()

        exptype_norm = normalize_header_value(exptype_raw)
        object_norm = normalize_header_value(object_raw)
        signature = _normalized_signature(exptype_norm, object_norm)

        image_type = str(hdr.get("IMAGE_TYPE", "")).strip().upper()
        override_image_type = overrides.get(signature)
        if override_image_type:
            image_type = override_image_type

        meta = {
            "path": path,
            "OBJECT": object_raw,
            "EXPTYPE": exptype_raw,
            "IMAGE_TYPE": image_type,
            "NIGHT": night,
            "SHOE": shoe,
            "PLATE": plate,
            "STACKED": bool(hdr.get("STACKED", False)),
            "STACKTYP": str(hdr.get("STACKTYP", hdr.get("STACKTYPE", ""))),
            "STACKMOD": str(hdr.get("STACKMOD", "")),
            "OPAMP": str(hdr.get("OPAMP", "")),
            "LC-TIME": str(hdr.get("LC-TIME", "")),
            "FILENAME": str(hdr.get("FILENAME", "")),
            "EXPTYPE_NORM": exptype_norm,
            "OBJECT_NORM": object_norm,
            "SIGNATURE": signature,
        }
        meta["CANDIDATE_ROLES"] = _candidate_roles_for_meta(meta)

        # DARK_MASTER fallback is allowed across NIGHT/PLATE; keep those as
        # informational metadata while still constraining non-dark roles.
        candidate_roles = meta.get("CANDIDATE_ROLES", set())
        if (
            (not allow_mixed_nights) and
            scope_night and night != scope_night and
            "dark" not in candidate_roles
        ):
            continue
        if scope_plate and plate != scope_plate and "dark" not in candidate_roles:
            continue

        candidates.append(meta)

    shoes = sorted({m["SHOE"] for m in candidates if m.get("SHOE")})
    selected_shoe = scope_shoe
    if selected_shoe:
        candidates = [m for m in candidates if m["SHOE"] == selected_shoe]
    elif len(shoes) == 1:
        selected_shoe = shoes[0]
        candidates = [m for m in candidates if m["SHOE"] == selected_shoe]

    grouped = {}
    for meta in candidates:
        key = (
            meta["NIGHT"],
            meta["SHOE"],
            meta.get("PLATE", ""),
            meta["EXPTYPE_NORM"],
            meta["OBJECT_NORM"],
        )
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "night": meta["NIGHT"],
                "shoe": meta["SHOE"],
                "plate": meta.get("PLATE", ""),
                "exptype_norm": meta["EXPTYPE_NORM"],
                "object_norm": meta["OBJECT_NORM"],
                "plate_label": meta.get("PLATE", ""),
                "exptype_label": meta["EXPTYPE"],
                "object_label": meta["OBJECT"],
                "signature": meta["SIGNATURE"],
                "candidates": [],
                "candidate_roles": set(),
            }
            grouped[key] = entry
        entry["candidates"].append(meta)
        entry["candidate_roles"].update(meta.get("CANDIDATE_ROLES", set()))

    groups = list(grouped.values())
    groups.sort(key=lambda g: (g["night"], g["shoe"], g.get("plate", ""), g["exptype_norm"], g["object_norm"]))

    return {
        "groups": groups,
        "all_groups": list(grouped.values()),
        "night": scope_night,
        "shoe": selected_shoe,
        "shoe_options": shoes,
        "plate": scope_plate,
        "skipped_missing": skipped_missing,
        "overrides": overrides,
        "override_path": override_path,
    }


def _format_inventory_label(group):
    return (
        f"PLATE={group.get('plate_label', '')} "
        f"EXPTYPE='{group['exptype_label']}' OBJECT='{group['object_label']}'"
    )


def _format_inventory_group_context_label(group):
    return (
        f"NIGHT={group.get('night', '')} "
        f"SHOE={group.get('shoe', '')} "
        f"PLATE={group.get('plate_label', group.get('plate', ''))} "
        f"EXPTYPE='{group.get('exptype_label', '')}' "
        f"OBJECT='{group.get('object_label', '')}'"
    )


def _format_manual_selection_summary(selections):
    if not selections:
        return "(none)"
    return ", ".join(
        f"{column}='{value}'" for column, value in selections.items()
    )


def _get_unique_group_values(groups, column):
    unique = []
    seen = set()
    for group in groups:
        value = group.get(column, "")
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def _filter_groups_by_value(groups, column, value):
    return [
        group for group in groups
        if str(group.get(column, "")) == str(value)
    ]


def _filter_groups_by_selections(groups, selections):
    survivors = list(groups)
    for column, value in selections.items():
        survivors = _filter_groups_by_value(survivors, column, value)
    return survivors


def _display_value_for_group_column(group, column):
    if column == "night":
        return str(group.get("night", "")).strip() or "(blank)"
    if column == "shoe":
        return str(group.get("shoe", "")).strip() or "(blank)"
    if column == "plate":
        plate_label = str(group.get("plate_label", "")).strip()
        plate_raw = str(group.get("plate", "")).strip()
        plate_value = plate_label or plate_raw
        return plate_value or "(blank)"
    if column == "exptype_norm":
        label = str(group.get("exptype_label", "")).strip()
        norm = str(group.get("exptype_norm", "")).strip()
        if label and norm:
            return f"'{label}'   [norm='{norm}']"
        return label or norm or "(blank)"
    if column == "object_norm":
        label = str(group.get("object_label", "")).strip()
        norm = str(group.get("object_norm", "")).strip()
        if label and norm:
            return f"'{label}'   [norm='{norm}']"
        return label or norm or "(blank)"
    value = str(group.get(column, "")).strip()
    return value or "(blank)"


def _prompt_manual_inventory_group_selection(
    candidates,
    parent_question,
    fixed_columns=None,
    ordered_columns=None,
    final_option_formatter=None,
):
    scoped_candidates = list(candidates)
    selections = {}
    fixed_columns = fixed_columns or {}
    ordered_columns = ordered_columns or (
        "night",
        "shoe",
        "plate",
        "exptype_norm",
        "object_norm",
    )
    final_option_formatter = final_option_formatter or _format_inventory_label

    for column in ordered_columns:
        fixed_value = fixed_columns.get(column)
        if fixed_value not in (None, ""):
            scoped_candidates = _filter_groups_by_value(scoped_candidates, column, fixed_value)
            selections[column] = fixed_value
            if len(scoped_candidates) <= 1:
                return scoped_candidates[0] if scoped_candidates else None

    for column in ordered_columns:
        fixed_value = fixed_columns.get(column)
        if fixed_value not in (None, ""):
            continue

        values = _get_unique_group_values(scoped_candidates, column)
        if not values:
            continue

        if len(values) == 1:
            selections[column] = values[0]
            matches = _filter_groups_by_selections(scoped_candidates, selections)
            if len(matches) == 1:
                return matches[0]
            continue

        print(f"\nManual selection for {parent_question}")
        if selections:
            print(f"Current filter: {_format_manual_selection_summary(selections)}")
        print(f"Choose {column}:")

        for idx, value in enumerate(values, start=1):
            sample_group = next(
                (
                    group for group in scoped_candidates
                    if str(group.get(column, "")) == str(value)
                ),
                None,
            )
            display_value = _display_value_for_group_column(sample_group or {}, column)
            print(f"[{idx}] {display_value}")
        print("[Enter] cancel manual mode")

        while True:
            answer = _prompt_input("Selection: ").strip()
            if not answer:
                return None
            try:
                selected = int(answer)
            except ValueError:
                print("Invalid selection. Please enter a menu number.")
                continue
            if 1 <= selected <= len(values):
                chosen_value = values[selected - 1]
                selections[column] = chosen_value
                matches = _filter_groups_by_selections(scoped_candidates, selections)
                if len(matches) == 1:
                    return matches[0]
                break
            print("Invalid selection. Please enter a menu number.")

    survivors = _filter_groups_by_selections(scoped_candidates, selections)

    if len(survivors) == 1:
        return survivors[0]

    if len(survivors) > 1:
        print("\nManual narrowing matched multiple candidates. Choose one:")
        for idx, group in enumerate(survivors, start=1):
            print(f"[{idx}] {final_option_formatter(group)}")
        print("[Enter] cancel manual mode")

        while True:
            answer = _prompt_input("Selection: ").strip()
            if not answer:
                return None
            try:
                selected = int(answer)
            except ValueError:
                print("Invalid selection. Please enter a menu number.")
                continue
            if 1 <= selected <= len(survivors):
                return survivors[selected - 1]
            print("Invalid selection. Please enter a menu number.")

    if selections:
        print("No candidates matched that exact column combination. Returning to parent prompt.")

    return None


def _group_from_candidate(meta):
    exptype_label = str(meta.get("EXPTYPE", ""))
    object_label = str(meta.get("OBJECT", ""))
    exptype_norm = normalize_header_value(exptype_label)
    object_norm = normalize_header_value(object_label)

    return {
        "night": str(meta.get("NIGHT", "")),
        "shoe": str(meta.get("SHOE", "")),
        "plate": str(meta.get("PLATE", "")).strip(),
        "plate_label": str(meta.get("PLATE", "")).strip(),
        "exptype_norm": exptype_norm,
        "object_norm": object_norm,
        "exptype_label": exptype_label,
        "object_label": object_label,
        "_candidate": meta,
    }


def _format_inventory_combinations(groups):
    if not groups:
        return ["  (none discovered)"]
    lines = []
    for g in groups:
        lines.append(
            "  - NIGHT={night} SHOE={shoe} {label} [n={count}]".format(
                night=g["night"],
                shoe=g["shoe"],
                label=_format_inventory_label(g),
                count=len(g["candidates"]),
            )
        )
    return lines


def _can_prompt_user():
    """Return True when interactive prompting is available."""
    stdin_tty = None
    try:
        stdin_tty = bool(sys.stdin.isatty())
        if stdin_tty:
            return True
    except Exception:
        stdin_tty = None

    # Respect explicit non-interactive stdin (e.g. tests, pipelines).
    if stdin_tty is False:
        return False

    try:
        return os.path.exists("/dev/tty") and os.access("/dev/tty", os.R_OK | os.W_OK)
    except Exception:
        return False


def _prompt_input(prompt):
    """Read interactive input from stdin, falling back to /dev/tty."""
    try:
        stdin_tty = bool(sys.stdin.isatty())
        if stdin_tty:
            return input(prompt)
        return ""
    except Exception:
        pass

    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(prompt)
            tty.flush()
            return tty.readline().rstrip("\n")
    except Exception:
        return ""


def _prompt_numbered_menu(question, options, text_aliases=None):
    print(question)
    for idx, option in enumerate(options, start=1):
        print(f"[{idx}] {option}")
    if len(options) > 1:
        print("[m] Enter values column-by-column")

    while True:
        answer = _prompt_input("Selection (Enter to skip): ").strip()
        if not answer:
            return None
        if answer.lower() == "m":
            return _MANUAL_MENU_SELECTION
        if text_aliases:
            alias_index = text_aliases.get(answer.upper())
            if alias_index is not None:
                return alias_index
        try:
            selected = int(answer)
        except ValueError:
            if text_aliases:
                valid_text = ", ".join(sorted(text_aliases.keys()))
                print(
                    f"Invalid selection. Please enter a menu number, one of: {valid_text}, or 'm'."
                )
            else:
                print("Invalid selection. Please enter a menu number or 'm'.")
            continue
        if 1 <= selected <= len(options):
            return selected - 1
        if text_aliases:
            valid_text = ", ".join(sorted(text_aliases.keys()))
            print(
                f"Invalid selection. Please enter a menu number, one of: {valid_text}, or 'm'."
            )
        else:
            print("Invalid selection. Please enter a menu number or 'm'.")


def _prompt_for_missing_roles_from_inventory(args, missing_roles, inventory):
    interactive = _can_prompt_user()
    groups = list(inventory.get("groups", []))
    selected = {}
    updated_overrides = False

    shoe_options = sorted({g["shoe"] for g in groups if g.get("shoe")})
    chosen_shoe = inventory.get("shoe")
    if chosen_shoe:
        groups = [g for g in groups if g["shoe"] == chosen_shoe]
    elif len(shoe_options) == 1:
        chosen_shoe = shoe_options[0]
        groups = [g for g in groups if g["shoe"] == chosen_shoe]
    elif len(shoe_options) > 1:
        if not interactive:
            return selected
        shoe_aliases = {str(option).upper(): idx for idx, option in enumerate(shoe_options)}
        while True:
            idx = _prompt_numbered_menu(
                "Which SHOE should be used for unresolved role mapping?",
                shoe_options,
                text_aliases=shoe_aliases,
            )
            if idx is None:
                return selected
            if idx == _MANUAL_MENU_SELECTION:
                manual_group = _prompt_manual_inventory_group_selection(
                    groups,
                    parent_question="SHOE selection",
                    fixed_columns={
                        "night": inventory.get("night"),
                        "plate": inventory.get("plate"),
                    },
                    final_option_formatter=_format_inventory_group_context_label,
                )
                if manual_group is None:
                    continue
                chosen_shoe = manual_group.get("shoe")
            else:
                chosen_shoe = shoe_options[idx]
            groups = [g for g in groups if g["shoe"] == chosen_shoe]
            break

    scope_night = inventory.get("night")
    if scope_night is None:
        nights = sorted({g["night"] for g in groups if g.get("night")})
        if len(nights) == 1:
            scope_night = nights[0]

    role_prompt_names = {
        "object": "science/object",
        "quartz": "quartz/flat",
        "thar": "ThAr/lamp/arc",
        "twilight": "twilight",
        "dark": "dark",
    }

    for role in missing_roles:
        if role == "dark":
            predicted = [
                g for g in groups
                if any(_is_dark_master_candidate(c) for c in g.get("candidates", []))
            ]
            role_groups = list(predicted)
        else:
            predicted = [g for g in groups if role in g.get("candidate_roles", set())]
            role_groups = predicted if predicted else list(groups)

        if (
            role == "object"
            and bool(getattr(args, "allow_mixed_nights", False))
            and getattr(args, "night", None)
        ):
            requested_night = str(args.night)
            role_groups = [
                g for g in role_groups
                if str(g.get("night", "")) == requested_night
            ]
            predicted = [
                g for g in predicted
                if str(g.get("night", "")) == requested_night
            ]

        if not role_groups:
            continue

        if interactive:
            while True:
                if role == "dark":
                    question = (
                        "\nWhich of the following is the DARK_MASTER image label for "
                        f"SHOE={chosen_shoe or '*'}? "
                        "(NIGHT/PLATE shown for context only)"
                    )
                    menu_options = [
                        _format_inventory_group_context_label(g)
                        for g in role_groups
                    ]
                else:
                    question = (
                        f"\nWhich of the following is the {role_prompt_names.get(role, role)} "
                        "image label for "
                        f"NIGHT={scope_night or '*'} SHOE={chosen_shoe or '*'} "
                        f"PLATE={inventory.get('plate') or '*'}:"
                    )
                    menu_options = [_format_inventory_label(g) for g in role_groups]

                idx = _prompt_numbered_menu(question, menu_options)
                if idx is None:
                    chosen_group = None
                    break

                if idx == _MANUAL_MENU_SELECTION:
                    manual_fixed_columns = {
                        "shoe": chosen_shoe,
                    }
                    if role != "dark":
                        manual_fixed_columns["night"] = scope_night
                        manual_fixed_columns["plate"] = inventory.get("plate")

                    manual_group = _prompt_manual_inventory_group_selection(
                        role_groups,
                        parent_question=role_prompt_names.get(role, role),
                        fixed_columns=manual_fixed_columns,
                        final_option_formatter=(
                            _format_inventory_group_context_label
                            if role == "dark"
                            else _format_inventory_label
                        ),
                    )
                    if manual_group is None:
                        continue
                    chosen_group = manual_group
                    break

                chosen_group = role_groups[idx]
                break

            if chosen_group is None:
                continue
        elif len(predicted) == 1:
            chosen_group = predicted[0]
        else:
            continue

        selected[role] = chosen_group

        image_type = ROLE_TO_IMAGE_TYPE.get(role)
        if image_type:
            signature = chosen_group.get("signature")
            if signature and inventory["overrides"].get(signature) != image_type:
                inventory["overrides"][signature] = image_type
                updated_overrides = True

    if updated_overrides:
        save_image_type_overrides(inventory["overrides"], inventory["override_path"])

    return selected


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


def _format_object_candidate_option(meta):
    """Build a concise disambiguation label for object candidates."""
    return (
        f"{os.path.basename(meta.get('path', ''))} | "
        f"OBJECT={meta.get('OBJECT', '')} "
        f"NIGHT={meta.get('NIGHT', '')} "
        f"SHOE={meta.get('SHOE', '')} "
        f"PLATE={meta.get('PLATE', '')} "
        f"IMAGE_TYPE={meta.get('IMAGE_TYPE', '')}"
    )


def _prompt_for_object_candidate(candidates):
    """Prompt interactively to select one ambiguous object candidate."""
    options = [_format_object_candidate_option(meta) for meta in candidates]
    while True:
        idx = _prompt_numbered_menu(
            "Multiple processed SCIENCE/object candidates were found. Select one:",
            options,
        )
        if idx is None:
            return None
        if idx == _MANUAL_MENU_SELECTION:
            candidate_groups = [_group_from_candidate(meta) for meta in candidates]
            selected_group = _prompt_manual_inventory_group_selection(
                candidate_groups,
                parent_question="science/object candidate selection",
                final_option_formatter=lambda g: _format_object_candidate_option(g.get("_candidate", {})),
            )
            if selected_group is None:
                continue
            return selected_group.get("_candidate")
        return candidates[idx]


def _prompt_for_role_candidate(role, candidates):
    """Prompt interactively to select one ambiguous non-object role candidate."""
    options = [_format_object_candidate_option(meta) for meta in candidates]
    while True:
        idx = _prompt_numbered_menu(
            f"Multiple processed {role} candidates were found. Select one:",
            options,
        )
        if idx is None:
            return None
        if idx == _MANUAL_MENU_SELECTION:
            candidate_groups = [_group_from_candidate(meta) for meta in candidates]
            selected_group = _prompt_manual_inventory_group_selection(
                candidate_groups,
                parent_question=f"{role} candidate selection",
                final_option_formatter=lambda g: _format_object_candidate_option(g.get("_candidate", {})),
            )
            if selected_group is None:
                continue
            return selected_group.get("_candidate")
        return candidates[idx]


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

        # Identical rank and same shoe: prompt if interactive, otherwise fail.
        if len(tied) > 1 and _can_prompt_user():
            if role == "object":
                selected = _prompt_for_object_candidate(tied)
            else:
                selected = _prompt_for_role_candidate(role, tied)
            if selected is not None:
                return selected

    msg = [f"Ambiguous candidates for role '{role}':"]
    for c in scored:
        rank = processing_rank(c)
        msg.append(
            f"  - {c['path']}  (rank={rank}, IMAGE_TYPE={c.get('IMAGE_TYPE', '')}, "
            f"EXPTYPE={c['EXPTYPE']}, OBJECT={c['OBJECT']}, NIGHT={c['NIGHT']}, "
            f"SHOE={c['SHOE']}, PLATE={c.get('PLATE', '')})"
        )
    if role == "object":
        msg.append(
            "Ambiguity detected in processed SCIENCE/object candidates. "
            "Select one interactively in a TTY session or pass explicit --object."
        )
    else:
        msg.append(f"Please pass an explicit --{role} file.")
    raise RuntimeError("\n".join(msg))


def _is_raw_amplifier_member(meta):
    """Return True when candidate path looks like a raw c1/c2/c3/c4 amplifier file."""
    name = os.path.basename(meta.get("path", "")).lower()
    return bool(re.match(r"^[br]\d{4}c[1-4]\.fits$", name))


def _is_stage_appropriate_reduce_input(meta):
    """Return True for processed products suitable as reduce_echelle role inputs."""
    path = meta.get("path", "")
    name = os.path.basename(path).lower()

    if _is_raw_amplifier_member(meta):
        return False
    if is_pipeline_intermediate(path):
        return False
    if is_stacked_product(meta):
        return True
    if any(tag in name for tag in ("-full", "-mcrr", "-ot")):
        return True
    return False


def _filter_reduce_stage_candidates(candidates):
    """Keep only candidates appropriate for reduce_echelle final role resolution."""
    return [meta for meta in candidates if _is_stage_appropriate_reduce_input(meta)]


def _resolve_role_from_inventory_group(role, group):
    """Resolve a prompted role-group mapping to a final processed file path."""
    all_candidates = list(group.get("candidates", []))

    if role == "dark":
        dark_master_candidates = [c for c in all_candidates if _is_dark_master_candidate(c)]
        if not dark_master_candidates:
            label = (
                f"NIGHT={group.get('night', '')} SHOE={group.get('shoe', '')} "
                f"PLATE={group.get('plate_label', group.get('plate', ''))} "
                f"EXPTYPE='{group.get('exptype_label', '')}' "
                f"OBJECT='{group.get('object_label', '')}'"
            )
            raise RuntimeError(
                f"Resolved role '{role}' from selected label:\n"
                f"  {label}\n"
                "but no DARK_MASTER candidate was found in this group.\n"
                "Pass an explicit --dark master-dark FITS path."
            )
        selected_meta = _select_unique_candidate(role, dark_master_candidates)
        return selected_meta["path"] if selected_meta is not None else None

    filtered = _filter_reduce_stage_candidates(all_candidates)
    if not filtered and len(all_candidates) == 1:
        # Inventory prompt flow may intentionally resolve from raw/header-only
        # groups (e.g., pre-preprocess labeling). Preserve that behavior.
        return all_candidates[0].get("path")

    if not filtered:
        label = (
            f"NIGHT={group.get('night', '')} SHOE={group.get('shoe', '')} "
            f"PLATE={group.get('plate_label', group.get('plate', ''))} "
            f"EXPTYPE='{group.get('exptype_label', '')}' "
            f"OBJECT='{group.get('object_label', '')}'"
        )
        raise RuntimeError(
            f"Resolved role '{role}' from unique header label:\n"
            f"  {label}\n"
            "but no stage-appropriate processed product was found.\n"
            "Only raw/repeated members were found in this group.\n"
            f"Run image_processing.py first or pass an explicit processed --{role} file."
        )

    selected_meta = _select_unique_candidate(role, filtered)
    return selected_meta["path"] if selected_meta is not None else None


def _resolve_prompted_roles_from_inventory(selection_by_role):
    """Resolve prompted role->group selections into role->path results."""
    resolved = {}
    for role, group in selection_by_role.items():
        selected_path = _resolve_role_from_inventory_group(role, group)
        if selected_path:
            resolved[role] = selected_path
    return resolved


def discover_inputs(input_dir, night=None, shoe=None, plate=None, object_name=None,
                    required_roles=None, allow_mixed_nights=False):
    """Auto-discover required role files from metadata-rich FITS products.

    In default mode, `night` constrains all non-dark roles.
    With `allow_mixed_nights=True`, `night` anchors science-object selection
    only; calibration roles may come from other nights.
    """
    required_roles = set(required_roles or ("quartz", "thar", "object", "twilight"))
    allow_mixed_nights = bool(allow_mixed_nights)
    fits_paths = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    by_role = {"quartz": [], "thar": [], "object": [], "twilight": [], "dark": []}
    skipped = 0

    for path in fits_paths:
        try:
            meta = read_required_metadata(path)
        except Exception:
            skipped += 1
            continue

        # Auto-discovery should only consider final stacked products, except
        # DARK_MASTER products that are valid dark-role inputs.
        if not is_stacked_product(meta) and not _is_dark_master_candidate(meta):
            continue

        # Exclude products generated by this script; role discovery should use
        # base stacked inputs from image_processing only.
        if is_pipeline_intermediate(meta["path"]):
            continue

        role = classify_role(meta)
        if role is None and _is_dark_master_candidate(meta):
            role = "dark"

        if night and str(meta["NIGHT"]) != str(night):
            if not allow_mixed_nights:
                if role != "dark":
                    continue
            elif role == "object":
                # In mixed-night mode, --night anchors science-target matching.
                continue
        if shoe and str(meta["SHOE"]).upper() != str(shoe).upper():
            continue
        if role != "dark" and plate and str(meta.get("PLATE", "")) != str(plate):
            continue
        if role is None:
            continue
        if object_name and role == "object":
            if object_name.lower() not in meta["OBJECT"].lower():
                continue
        if role == "dark" and not _is_dark_master_candidate(meta):
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

    inferred_night = None if allow_mixed_nights else (
        night if night else (object_meta["NIGHT"] if object_meta else None)
    )
    inferred_shoe = shoe if shoe else (object_meta["SHOE"] if object_meta else None)
    inferred_plate = plate if plate else (object_meta.get("PLATE") if object_meta else None)

    for role in ("quartz", "thar", "twilight"):
        if role not in required_roles:
            continue
        role_candidates = by_role[role]
        if inferred_night is not None:
            role_candidates = [c for c in role_candidates if str(c["NIGHT"]) == str(inferred_night)]
        if inferred_shoe is not None:
            role_candidates = [c for c in role_candidates if str(c["SHOE"]).upper() == str(inferred_shoe).upper()]
        if inferred_plate not in (None, ""):
            role_candidates = [
                c for c in role_candidates if str(c.get("PLATE", "")) == str(inferred_plate)
            ]
        selected_meta = _select_unique_candidate(role, role_candidates)
        if selected_meta is not None:
            selected[role] = selected_meta["path"]

    if "dark" in required_roles:
        dark_candidates = list(by_role["dark"])
        if inferred_shoe is not None:
            dark_candidates = [
                c for c in dark_candidates
                if str(c["SHOE"]).upper() == str(inferred_shoe).upper()
            ]
        if len(dark_candidates) == 1:
            selected["dark"] = dark_candidates[0]["path"]

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
        "dark": getattr(args, "dark", None),
    }

    if all(manual[r] for r in required_roles):
        resolved.update({role: path for role, path in manual.items() if path})
        return resolved

    if required_roles:
        discovered = discover_inputs(
            args.input_dir,
            night=args.night,
            shoe=args.shoe,
            plate=args.plate,
            object_name=args.object_name,
            required_roles=required_roles,
            allow_mixed_nights=getattr(args, "allow_mixed_nights", False),
        )
        resolved.update(discovered)

    for role, path in manual.items():
        if path:
            resolved[role] = path

    missing = [r for r in required_roles if r not in resolved]
    inventory = None
    if missing:
        inventory = collect_header_inventory(args, resolved, required_roles=required_roles)
        prompted_groups = _prompt_for_missing_roles_from_inventory(args, missing, inventory)
        prompted = _resolve_prompted_roles_from_inventory(prompted_groups)
        resolved.update(prompted)
        missing = [r for r in required_roles if r not in resolved]

    if "dark" in missing and _can_prompt_user():
        manual_dark = _prompt_input(
            "No DARK_MASTER was auto-discovered. Enter manual --dark path "
            "(or press Enter to skip): "
        ).strip()
        if manual_dark:
            if os.path.isfile(manual_dark):
                resolved["dark"] = manual_dark
                missing = [r for r in required_roles if r not in resolved]
            else:
                print(f"  [warning] Manual dark path does not exist: {manual_dark}")

    if missing:
        details = [
            "Could not resolve all required inputs. Missing roles: "
            f"{', '.join(missing)}."
        ]
        if inventory is not None:
            details.append("Discovered unique EXPTYPE/OBJECT combinations:")
            details.append("(with NIGHT/SHOE/PLATE context)")
            details.extend(_format_inventory_combinations(inventory.get("groups", [])))
            if inventory.get("skipped_missing", 0):
                details.append(
                    f"  [info] {inventory['skipped_missing']} FITS files lacked one or more "
                    "required inventory keys (OBJECT/EXPTYPE/NIGHT/SHOE)."
                )
            shoe_options = sorted({g["shoe"] for g in inventory.get("groups", []) if g.get("shoe")})
            if not inventory.get("shoe") and len(shoe_options) > 1:
                details.append(
                    "  [info] Multiple SHOEs discovered in current run context: "
                    f"{', '.join(shoe_options)}"
                )
        if not _can_prompt_user():
            details.append(
                "Non-interactive mode cannot prompt for unresolved roles. "
                "Provide explicit flags or rerun in a TTY."
            )
        if "dark" in missing:
            details.append(
                "For DARK_MASTER selection, NIGHT/PLATE are informational only. "
                "Provide explicit --dark when auto-discovery is ambiguous or empty."
            )
        details.append(
            "Provide explicit flags or refine --input-dir/--night/--shoe/--object-name."
        )
        raise RuntimeError(
            "\n".join(details)
        )
    return resolved


def validate_input_set(meta_by_role, args):
    """Validate metadata consistency and gate mismatches by confirmation."""
    info = []
    allow_mixed_nights = bool(getattr(args, "allow_mixed_nights", False))
    warnings = []
    nights = {meta_by_role[r]["NIGHT"] for r in meta_by_role}
    shoes = {meta_by_role[r]["SHOE"] for r in meta_by_role}
    plates = {
        str(meta_by_role[r].get("PLATE", "")).strip()
        for r in meta_by_role
        if str(meta_by_role[r].get("PLATE", "")).strip()
    }
    if len(nights) != 1:
        msg = f"Mixed NIGHT values: {sorted(nights)}"
        if allow_mixed_nights:
            info.append(msg + " (--allow-mixed-nights)")
        else:
            warnings.append(msg)
    if len(shoes) != 1:
        warnings.append(f"Mixed SHOE values: {sorted(shoes)}")
    if len(plates) > 1:
        warnings.append(f"Mixed PLATE values: {sorted(plates)}")

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
                f"(IMAGE_TYPE={meta.get('IMAGE_TYPE', '')}, EXPTYPE={meta['EXPTYPE']}, OBJECT={meta['OBJECT']})"
            )

    if info:
        section_banner("Input metadata notes")
        for item in info:
            print(f"  [info] {item}")

    if not warnings:
        return

    section_banner("Input metadata warnings")
    for w in warnings:
        print(f"  [warn] {w}")

    if args.yes:
        print("  --yes set: continuing despite metadata warnings.")
        return

    if not _can_prompt_user():
        raise RuntimeError(
            "Metadata mismatch detected in non-interactive mode. "
            "Re-run with --yes to continue anyway."
        )

    answer = _prompt_input("\nProceed anyway? Type 'yes' to continue: ").strip().lower()
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

def _apscatter_one(image, output, reference, proc_dir=None):
    iraf_input = os.path.basename(str(image).split("[", 1)[0])
    iraf_reference = quartz_reference_token(reference)

    try:
        iraf_cwd = iraf.pwd()
    except Exception:
        iraf_cwd = "<unavailable>"
    print(
        "  [debug step2 pre] "
        f"python_cwd={os.getcwd()} iraf_cwd={iraf_cwd} "
        f"input={image} iraf_input={iraf_input} "
        f"reference={reference} iraf_reference={iraf_reference} output={output} "
        "interactive=yes fitscatter=yes fitsmooth=yes"
    )

    iraf.echelle.apscatter.unlearn()
    iraf.echelle.apscatter(
        input       = iraf_input,
        output      = output,
        apertures   = "",
        scatter     = "",
        references  = iraf_reference,
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

    in_dir = os.path.dirname(str(image))
    in_dir_output = os.path.join(in_dir, output) if in_dir else output
    proc_dir_output = os.path.join(proc_dir, output) if proc_dir else output
    exists_local = os.path.exists(output)
    exists_input_dir = os.path.exists(in_dir_output)
    exists_proc_dir = os.path.exists(proc_dir_output)
    size_local = os.path.getsize(output) if exists_local else None
    size_input_dir = os.path.getsize(in_dir_output) if exists_input_dir else None
    size_proc_dir = os.path.getsize(proc_dir_output) if exists_proc_dir else None
    print(
        "  [debug step2 post] "
        f"output={output} exists_local={exists_local} size_local={size_local} "
        f"exists_proc_dir={exists_proc_dir} size_proc_dir={size_proc_dir} "
        f"exists_input_dir={exists_input_dir} size_input_dir={size_input_dir}"
    )

    print(f"  apscatter done: {image} -> {output}")


def apscatter_all(quartz, thar, obj, twilight, proc_dir=None):
    section_banner("Step 2 – apscatter (scattered-light subtraction)")
    inputs = [quartz, thar, obj, twilight]
    outputs = [step2_scattered(img) for img in inputs]

    for out in outputs:
        iraf_delete(out)

    for img, out in zip(inputs, outputs):
        print(f"\n  -- {img} -> {out}")
        _apscatter_one(img, out, reference=quartz, proc_dir=proc_dir)

    return tuple(outputs)


def _step2_interactive_preflight(require_interactive=True):
    """Best-effort preflight for interactive apscatter readiness."""
    stdin_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    stdout_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    stderr_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    display = os.environ.get("DISPLAY", "")
    gterm = os.environ.get("stdgraph", "")
    print(
        "  [debug step2 preflight] "
        f"stdin_tty={stdin_tty} stdout_tty={stdout_tty} stderr_tty={stderr_tty} "
        f"DISPLAY={display or '<unset>'} stdgraph={gterm or '<unset>'}"
    )

    if require_interactive and not (stdin_tty and stdout_tty and stderr_tty):
        raise RuntimeError(
            "Step 2 requires an interactive terminal (stdin/stdout/stderr TTY). "
            "Run in an interactive shell session."
        )


def _debug_apscatter_quartz_only(quartz, proc_dir):
    """Temporary debug helper: run apscatter only on quartz and validate output."""
    section_banner("Step 2 DEBUG – apscatter quartz-only")
    quartz_sl = step2_scattered(quartz)
    _apscatter_one(quartz, quartz_sl, reference=quartz, proc_dir=proc_dir)
    require_existing(quartz_sl, "step 2 debug quartz-sl output", proc_dir=proc_dir)
    print(f"  step2-debug quartz output verified: {quartz_sl}")


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
    median_flat = step3_median(quartz)
    master_flat = step3_master_flat(quartz)

    # 3a – fmedian
    iraf_delete(median_flat)
    print(f"  fmedian  {quartz}  ->  {median_flat}  (xwindow={xwindow}, ywindow=1)")
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
    outputs  = [step4_flat_corrected(t) for t in targets]

    context_meta = None
    for candidate in (obj, thar, twilight):
        try:
            context_meta = read_required_metadata(candidate)
            break
        except Exception:
            continue

    if context_meta:
        night_tok = _safe_token(context_meta.get("NIGHT", "na"))
        shoe_tok = _safe_token(context_meta.get("SHOE", "na"))
        plate_val = str(context_meta.get("PLATE", "")).strip()
        plate_tok = _safe_token(plate_val) if plate_val else "na"
        obj_tok = _safe_token(_normalize_token_text(context_meta.get("OBJECT", "")))
        list_tag = f"{night_tok}_{shoe_tok}_{plate_tok}_{obj_tok}"
    else:
        list_tag = "context_na"

    in_list  = f"_flatcorr_{list_tag}_in.list"
    out_list = f"_flatcorr_{list_tag}_out.list"
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

def run_aperture_preview(preview_image, n_stars, sep, preview_out=None,
                        interactive=True, pattern_override=None, aps_per_star=4):
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
        return _fallback_pattern(preview_image, n_stars, sep, aps_per_star=aps_per_star)

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
        interactive=interactive,
        pattern_override=pattern_override,
        aps_per_star=aps_per_star,
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


def _fallback_pattern(image_path, n_stars, sep, aps_per_star=4):
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
        pattern = _default_pattern(
            len(centers),
            n_stars=n_stars,
            aps_per_star=aps_per_star,
        )
    except Exception:
        pattern = (np.arange(len(centers), dtype=int) // aps_per_star) + 1
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
    out_spec = step6_star_extract(image, star_number)
    out_stem = stem(out_spec)
    iraf_delete(out_spec)

    print(f"      apall  apertures={aperture_string!r}  ->  {out_spec}")

    iraf_reference = quartz_reference_token(reference_quartz)

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


def _step6_debug_report_path(obj_ff, thar_ff):
    """Return deterministic Step-6 extraction debug report path."""
    return f"step6_extraction_debug_{stem(obj_ff)}_{stem(thar_ff)}.json"


def _step6_quartz_apertures_in_preview_order(quartz_path, expected_count):
    """Return quartz DB aperture IDs in the same order used by preview."""
    trace_details = _load_quartz_trace_details(quartz_path)
    entries = sorted(
        list(trace_details.get("entries", [])),
        key=lambda entry: int(entry.get("aperture", 0)),
    )
    aperture_ids = [int(entry.get("aperture")) for entry in entries if entry.get("aperture") is not None]

    if len(aperture_ids) < int(expected_count):
        db_path = trace_details.get("db_path") or "<missing>"
        raise RuntimeError(
            "Step 6 preflight could not map dense pattern to quartz DB order: "
            f"pattern has {int(expected_count)} apertures but quartz DB '{db_path}' "
            f"has only {len(aperture_ids)} parsed aperture entries."
        )

    if len(aperture_ids) > int(expected_count):
        aperture_ids = aperture_ids[: int(expected_count)]

    return aperture_ids, (trace_details.get("db_path") or "")


def _step6_focus_rows(star_rows, start_star=12, end_star=16):
    """Return compact debug rows for a local star range."""
    focus = []
    for row in sorted(star_rows, key=lambda item: int(item.get("star", 0))):
        star = int(row.get("star", 0))
        if star < int(start_star) or star > int(end_star):
            continue
        focus.append(
            {
                "star": star,
                "pattern_slots": list(row.get("pattern_slots", [])),
                "dense_apertures": list(row.get("dense_apertures", [])),
                "quartz_db_apertures": list(row.get("quartz_db_apertures", [])),
                "aperture_str": str(row.get("aperture_str", "")),
                "object_pre_renumber": list(row.get("object_pre_renumber", [])),
                "thar_pre_renumber": list(row.get("thar_pre_renumber", [])),
                "object_post_renumber": list(row.get("object_post_renumber", [])),
                "thar_post_renumber": list(row.get("thar_post_renumber", [])),
            }
        )
    return focus


def _write_step6_debug_report(report_path, payload):
    """Persist structured Step-6 extraction diagnostics."""
    payload["focus_stars_12_16"] = _step6_focus_rows(payload.get("stars", []), 12, 16)
    with open(report_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _print_step6_focus_report(payload, stage_label):
    """Print compact stars-12..16 diagnostics to terminal."""
    rows = payload.get("focus_stars_12_16") or _step6_focus_rows(payload.get("stars", []), 12, 16)
    if not rows:
        return

    print(f"\n  Step 6 debug focus ({stage_label}) stars 12-16:")
    for row in rows:
        print(
            "    star {star:02d} | slots={slots} | dense={dense} | quartz={quartz} | "
            "obj_pre={obj_pre} | thar_pre={thar_pre} | obj_post={obj_post} | thar_post={thar_post}".format(
                star=int(row.get("star", 0)),
                slots=row.get("pattern_slots", []),
                dense=row.get("dense_apertures", []),
                quartz=row.get("quartz_db_apertures", []),
                obj_pre=row.get("object_pre_renumber", []),
                thar_pre=row.get("thar_pre_renumber", []),
                obj_post=row.get("object_post_renumber", []),
                thar_post=row.get("thar_post_renumber", []),
            )
        )


def extract_all_stars(obj_ff, thar_ff, quartz, pattern):
    """
    For every unique star in *pattern*:
      1. Collect that star's 1-based aperture indices and build an IRAF range
         string.
      2. Run apall on the flat-corrected object  -> <obj_stem>_star<N>_ec.fits
      3. Run apall on the flat-corrected thar    -> <thar_stem>_star<N>_ec.fits
            using exactly the same aperture selection.
        4. Renumber both extracted outputs to local apertures 1..N so downstream
            wavelength steps can match apertures directly.

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

    debug_report_path = _step6_debug_report_path(obj_ff, thar_ff)
    debug_payload = {
        "schema_version": 1,
        "obj_ff": str(obj_ff),
        "thar_ff": str(thar_ff),
        "quartz": str(quartz),
        "pattern_length": int(len(pattern)),
        "stars": [],
    }

    quartz_apertures_ordered, quartz_db_path = _step6_quartz_apertures_in_preview_order(
        quartz,
        expected_count=len(pattern),
    )
    debug_payload["quartz_db_path"] = str(quartz_db_path)
    debug_payload["quartz_apertures_preview_order"] = [int(v) for v in quartz_apertures_ordered]

    first_preflight_mismatch = None
    star_debug_by_id = {}
    for star in unique_stars:
        slot_indices = np.where(pattern == star)[0]
        pattern_slots = [int(v + 1) for v in slot_indices.tolist()]
        dense_apertures = list(pattern_slots)
        quartz_apertures = [int(quartz_apertures_ordered[idx]) for idx in slot_indices.tolist()]
        aperture_str = _aperture_range_string(dense_apertures)

        row = {
            "star": int(star),
            "pattern_slots": [int(v) for v in pattern_slots],
            "dense_apertures": [int(v) for v in dense_apertures],
            "quartz_db_apertures": [int(v) for v in quartz_apertures],
            "aperture_str": str(aperture_str),
            "object_pre_renumber": [],
            "thar_pre_renumber": [],
            "object_post_renumber": [],
            "thar_post_renumber": [],
        }
        debug_payload["stars"].append(row)
        star_debug_by_id[int(star)] = row

        if first_preflight_mismatch is None and dense_apertures != quartz_apertures:
            first_preflight_mismatch = {
                "star": int(star),
                "dense_apertures": [int(v) for v in dense_apertures],
                "quartz_db_apertures": [int(v) for v in quartz_apertures],
            }

    if first_preflight_mismatch is not None:
        debug_payload["error"] = (
            "Step 6 preflight mismatch: dense pattern apertures differ from quartz DB preview order"
        )
        _write_step6_debug_report(debug_report_path, debug_payload)
        _print_step6_focus_report(debug_payload, stage_label="preflight-mismatch")
        star = int(first_preflight_mismatch["star"])
        dense = first_preflight_mismatch["dense_apertures"]
        quartz_ids = first_preflight_mismatch["quartz_db_apertures"]
        raise RuntimeError(
            "Step 6 preflight mismatch at star "
            f"{star:02d}: dense pattern apertures {dense} != quartz DB apertures {quartz_ids}. "
            f"Debug report: {debug_report_path}"
        )

    _write_step6_debug_report(debug_report_path, debug_payload)

    # Load IRAF packages for extraction.
    iraf.onedspec()
    iraf.noao()
    iraf.imred()
    iraf.echelle()

    for star in unique_stars:
        debug_row = star_debug_by_id[int(star)]
        # All 1-based aperture indices that belong to this star
        ap_indices   = list(np.where(pattern == star)[0] + 1)
        aperture_str = _aperture_range_string(ap_indices)

        print(
            f"\n  [step6-debug] star={int(star):02d} "
            f"slots={debug_row['pattern_slots']} "
            f"dense_apertures={debug_row['dense_apertures']} "
            f"aperture_str='{aperture_str}'"
        )
        if int(star) == 14:
            print(
                "  [step6-debug star14] "
                f"requested_dense={debug_row['dense_apertures']} "
                f"quartz_db={debug_row['quartz_db_apertures']}"
            )

        print(f"\n  -- Star {star:02d}  |  {len(ap_indices)} apertures  "
              f"|  IRAF range: {aperture_str}")

        print(f"    Extracting object ...")
        obj_ec = _apall_extract_star(
            obj_ff, quartz, aperture_str,
            star_number = star,
        )
        obj_pre = _read_apertures_from_apnum_cards(obj_ec)
        debug_row["object_pre_renumber"] = [int(v) for v in obj_pre]
        print(
            "    [step6-debug object pre] "
            f"requested={debug_row['dense_apertures']} extracted={obj_pre} "
            f"count={len(obj_pre)}/{len(ap_indices)}"
        )
        if len(obj_pre) != len(ap_indices):
            debug_payload["error"] = (
                f"Step 6 extraction mismatch for star {int(star):02d} (object): "
                f"requested {debug_row['dense_apertures']} extracted {obj_pre}"
            )
            _write_step6_debug_report(debug_report_path, debug_payload)
            _print_step6_focus_report(debug_payload, stage_label="object-pre-mismatch")
            raise RuntimeError(
                f"Step 6 extraction mismatch for star {int(star):02d}: "
                f"requested {debug_row['dense_apertures']}, extracted {obj_pre}"
            )

        print(f"    Extracting ThAr ...")
        thar_ec = _apall_extract_star(
            thar_ff, quartz, aperture_str,
            star_number = star,
        )
        thar_pre = _read_apertures_from_apnum_cards(thar_ec)
        debug_row["thar_pre_renumber"] = [int(v) for v in thar_pre]
        print(
            "    [step6-debug thar pre] "
            f"requested={debug_row['dense_apertures']} extracted={thar_pre} "
            f"count={len(thar_pre)}/{len(ap_indices)}"
        )
        if len(thar_pre) != len(ap_indices):
            debug_payload["error"] = (
                f"Step 6 extraction mismatch for star {int(star):02d} (thar): "
                f"requested {debug_row['dense_apertures']} extracted {thar_pre}"
            )
            _write_step6_debug_report(debug_report_path, debug_payload)
            _print_step6_focus_report(debug_payload, stage_label="thar-pre-mismatch")
            raise RuntimeError(
                f"Step 6 extraction mismatch for star {int(star):02d}: "
                f"requested {debug_row['dense_apertures']}, extracted {thar_pre}"
            )

        local_mapping = _build_local_aperture_mapping(ap_indices)
        print(f"    local apertures   : {_format_aperture_mapping(local_mapping)}")
        _renumber_multispec_fits_apertures(obj_ec, local_mapping)
        _renumber_multispec_fits_apertures(thar_ec, local_mapping)

        obj_post = _read_apertures_from_apnum_cards(obj_ec)
        thar_post = _read_apertures_from_apnum_cards(thar_ec)
        debug_row["object_post_renumber"] = [int(v) for v in obj_post]
        debug_row["thar_post_renumber"] = [int(v) for v in thar_post]

        expected_obj_local = list(range(1, len(obj_pre) + 1))
        expected_thar_local = list(range(1, len(thar_pre) + 1))
        if obj_post != expected_obj_local:
            debug_payload["error"] = (
                f"Step 6 renumber mismatch for star {int(star):02d} (object): "
                f"pre {obj_pre} post {obj_post}"
            )
            _write_step6_debug_report(debug_report_path, debug_payload)
            _print_step6_focus_report(debug_payload, stage_label="object-post-mismatch")
            raise RuntimeError(
                f"Step 6 renumber mismatch for star {int(star):02d} (object): "
                f"pre {obj_pre}, post {obj_post}"
            )
        if thar_post != expected_thar_local:
            debug_payload["error"] = (
                f"Step 6 renumber mismatch for star {int(star):02d} (thar): "
                f"pre {thar_pre} post {thar_post}"
            )
            _write_step6_debug_report(debug_report_path, debug_payload)
            _print_step6_focus_report(debug_payload, stage_label="thar-post-mismatch")
            raise RuntimeError(
                f"Step 6 renumber mismatch for star {int(star):02d} (thar): "
                f"pre {thar_pre}, post {thar_post}"
            )

        _write_step6_debug_report(debug_report_path, debug_payload)

        obj_outputs[star] = obj_ec
        thar_outputs[star] = thar_ec

    _write_step6_debug_report(debug_report_path, debug_payload)
    _print_step6_focus_report(debug_payload, stage_label="final")
    print(f"  Step 6 debug report : {debug_report_path}")

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
        out_spec = step7_crr2(in_spec)
        print(f"\n  -- Star {star:02d}  |  {in_spec}")
        _lineclean_one(in_spec, out_spec)
        crr2_outputs[star] = out_spec

    return crr2_outputs


def expected_step7_outputs(obj_outputs):
    """Return deterministic step-7 output paths."""
    return {star: step7_crr2(path) for star, path in obj_outputs.items()}


# ---------------------------------------------------------------------------
# Step 8/9/10/11 – line identification and wavelength assignment
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
    for probe in (reference_input, iraf_reference_token):
        found_db, _db_candidates = resolve_existing_wavelength_db(probe)
        if found_db:
            break

    if found_db:
        found_db = ensure_canonical_wavelength_db_entry(
            iraf_reference_token,
            source_db=found_db,
        )

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
    """Backward-compatible wrapper: canonicalize wavelength DB entry only."""
    return ensure_canonical_wavelength_db_entry(thar_path, source_db=source_db)


def normalize_ec_database_records(db_path, canonical_root):
    """Normalize wavelength DB record identity fields to canonical bare-root token."""
    if not db_path or not os.path.exists(db_path):
        return False

    canonical_token = canonical_wavelength_identity(canonical_root)
    if not canonical_token:
        return False

    with open(db_path, "r") as fh:
        lines = fh.readlines()

    rewritten = []
    changed = False
    for line in lines:
        raw = line.rstrip("\n")
        newline = "\n" if line.endswith("\n") else ""
        stripped = raw.strip()

        m_begin_ap = re.match(r"^(\s*begin\s+\S+\s+)(\S+)(\s+)(-?\d+)(\s*)$", raw)
        if m_begin_ap:
            old_ap = int(m_begin_ap.group(4))
            new_line = (
                f"{m_begin_ap.group(1)}{canonical_token}{m_begin_ap.group(3)}"
                f"{old_ap}{m_begin_ap.group(5)}{newline}"
            )
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_begin = re.match(r"^(\s*begin\s+\S+\s+)(\S+)(\s*)$", raw)
        if m_begin:
            new_line = f"{m_begin.group(1)}{canonical_token}{m_begin.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_image = re.match(r"^(\s*image\s+)(\S+)(\s*)$", raw)
        if m_image:
            new_line = f"{m_image.group(1)}{canonical_token}{m_image.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_id = re.match(r"^(\s*id\s+)(\S+)(\s*)$", raw)
        if m_id:
            new_line = f"{m_id.group(1)}{canonical_token}{m_id.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        rewritten.append(line)

    if changed:
        with open(db_path, "w") as fh:
            fh.writelines(rewritten)
    return changed


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


def _refspec_one(obj_ec, thar_ec, debug=False):
    """
    Assign the ThAr wavelength solution from *thar_ec* to the object
    spectrum *obj_ec* using refspec.

    select=match means IRAF pairs apertures by number (1→1, 2→2, …), which
    is correct because step 6 permanently renumbers each object+ThAr pair to
    the same local aperture IDs.
    """
    obj_iraf = os.path.basename(str(obj_ec).split("[", 1)[0])
    ref_iraf = os.path.basename(str(thar_ec).split("[", 1)[0])
    ref_db_token = iraf_spec_token(thar_ec)
    print(
        "  refspec: "
        f"object_file={obj_ec} object_iraf={obj_iraf}  <--  "
        f"ref_file={thar_ec} ref_iraf={ref_iraf} ref_db_token={ref_db_token} "
        f"[debug={'yes' if debug else 'no'}]"
    )
    iraf.noao.onedspec.refspec.unlearn()
    iraf.noao.onedspec.refspec(
        input    = obj_iraf,
        referenc = ref_iraf,
        aperture = "",
        refaps   = "",
        ignoreap = iraf.no,
        select   = "match",
        sort     = "",
        group    = "",
        time     = iraf.no,
        timewrap = 17.0,
        override = iraf.yes if debug else iraf.no,
        confirm  = iraf.yes if debug else iraf.no,
        assign   = iraf.yes,
        logfile  = "STDOUT,logfile",
        verbose  = iraf.yes if debug else iraf.no,
        mode     = "ql",
    )

    try:
        hdr = fits.getheader(obj_ec)
        refspec_keys = [
            str(key) for key in hdr.keys()
            if str(key) == "REFSPEC" or re.match(r"^REFSPEC\d+$", str(key))
        ]
        if refspec_keys:
            joined = ", ".join(f"{key}={hdr.get(key)}" for key in sorted(refspec_keys))
            print(f"  refspec header keys: {joined}")
        else:
            print("  refspec header keys: <none>")
    except Exception as exc:
        print(f"  [warn] could not inspect object REFSPEC header keys: {exc}")


def _dispcor_one(obj_ec, output_spec):
    """
    Apply (implant) the wavelength solution assigned by refspec onto *obj_ec*,
    writing the linearised spectrum to *output_spec*.

    flux=yes conserves total flux when resampling to a linear grid.
    """
    iraf_delete(output_spec)
    print(f"  dispcor: {obj_ec}  ->  {output_spec}")
    dispcor_task = iraf.noao.onedspec.dispcor
    dispcor_task.unlearn()

    params = {
        "input": obj_ec,
        "output": output_spec,
        "linearize": iraf.yes,
        "database": "database",
        "table": "",
        "w1": "INDEF",
        "w2": "INDEF",
        "dw": "INDEF",
        "nw": "INDEF",
        "log": iraf.no,
        "flux": iraf.yes,
        "blank": 0.0,
        "samedisp": iraf.no,
        "global": iraf.no,
        "ignoreaps": iraf.no,
        "confirm": iraf.no,
        "listonly": iraf.no,
        "verbose": iraf.yes,
        "logfile": "",
        "mode": "ql",
    }

    try:
        dispcor_task(**params)
    except TypeError:
        # Compatibility fallback for IRAF/PyRAF variants that expose short names.
        compat = dict(params)
        compat.pop("global", None)
        compat["lineariz"] = compat.pop("linearize")
        compat["ignoreap"] = compat.pop("ignoreaps")
        compat["listonl"] = compat.pop("listonly")
        dispcor_task(**compat)


def _choose_step8_mode(step8_mode):
    """Resolve Step-8 mode, prompting only for interactive TTY runs."""
    if step8_mode in {"manual", "reuse"}:
        return step8_mode

    if not _can_prompt_user():
        print("  step 8 mode: non-interactive run, defaulting to manual ecidentify")
        return "manual"

    print("\n  Step 8 mode for reference star:")
    print("    [m] manual ecidentify")
    print("    [r] reuse existing identified ThAr (run ecreidentify)")

    while True:
        answer = _prompt_input("  Choose mode [m/r] (default: m): ").strip().lower()
        if answer in {"", "m", "manual"}:
            return "manual"
        if answer in {"r", "reuse"}:
            return "reuse"
        print("  Please type 'm' for manual or 'r' for reuse.")


def _resolve_reuse_reference_thar(reference_thar):
    """Resolve and validate external reference ThAr path for Step-8 reuse mode."""
    ref_path = reference_thar

    if not ref_path and _can_prompt_user():
        ref_path = _prompt_input(
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

    ref_token = iraf_spec_token(ref_path)
    canonical_db = ensure_canonical_wavelength_db_entry(ref_token, source_db=found_db)
    if not canonical_db:
        raise RuntimeError(
            "Step 8 reuse mode could not normalize the wavelength DB to canonical form "
            f"for reference token '{ref_token}'."
        )

    print(f"  reuse reference ThAr : {ref_token}")
    print(f"  reuse wavelength DB  : {canonical_db}")
    return ref_token


def resolve_step8_runtime_config(step8_mode, step8_reference_thar=None):
    """Resolve step-8 execution mode and optional reuse token before core logic."""
    mode = _choose_step8_mode(step8_mode)
    reference_token = None
    if mode == "reuse":
        reference_token = _resolve_reuse_reference_thar(step8_reference_thar)
    return mode, reference_token


def manual_wavelength_identification(crr2_outputs, thar_outputs,
                                     coordlist="linelists$thar.dat",
                                     step8_mode="manual",
                                     step8_reference_token=None,
                                     drift_log_path=None,
                                     explicit_ref_star=None):
    """
    Step 8: Reference-star wavelength setup (manual or reuse mode).

    Manual mode runs ecidentify on one reference-star ThAr spectrum.
    Reuse mode runs ecreidentify on that reference star, using an already
    identified external ThAr reference and its existing IRAF DB entry.
    Refspec assignment is deferred to step 10.

    Reference-star selection defaults to the first star unless an explicit
    --ref-star override is provided by orchestration.

    Parameters
    ----------
    crr2_outputs : {star_number: path, ...}
        CR-cleaned object spectra from step 7.
    thar_outputs : {star_number: path, ...}
        ThAr spectra from step 6 (same aperture selection as objects).
    coordlist    : str
        IRAF line-list path (default: built-in ThAr list).
    step8_mode   : {'manual', 'reuse'}
        Step-8 mode resolved by orchestration code.
    step8_reference_token : str or None
        Canonical ThAr reference token for reuse mode.
    explicit_ref_star : int or None
        Optional explicit reference star ID.

    Returns
    -------
    ref_star : int
        The star number used as reference.
    """
    section_banner("Step 8 – Reference-star wavelength setup (manual/reuse)")
    iraf.noao()
    iraf.echelle()
    iraf.onedspec()

    available_stars = sorted(crr2_outputs.keys())
    if explicit_ref_star is None:
        ref_star = available_stars[0]
    else:
        ref_star = int(explicit_ref_star)
        if ref_star not in crr2_outputs:
            raise RuntimeError(
                f"Requested --ref-star {ref_star:02d} is not available in extracted spectra. "
                f"Available stars: {', '.join(f'{s:02d}' for s in available_stars)}"
            )
    if ref_star not in thar_outputs:
        available_thar = ", ".join(f"{s:02d}" for s in sorted(thar_outputs.keys()))
        raise RuntimeError(
            f"Requested reference star {ref_star:02d} is not available in extracted ThAr spectra. "
            f"Available stars: {available_thar}"
        )
    obj_crr2_ref = crr2_outputs[ref_star]
    thar_ec_ref  = thar_outputs[ref_star]

    print(f"\n  Reference star: {ref_star:02d}")
    print(f"    object : {obj_crr2_ref}")
    print(f"    thar   : {thar_ec_ref}")

    mode = step8_mode
    if mode not in {"manual", "reuse"}:
        raise RuntimeError(
            f"Step 8 received unresolved mode '{mode}'. "
            "Resolve mode before calling manual_wavelength_identification."
        )

    if mode == "manual":
        # 8a – ecidentify on reference ThAr (interactive)
        _ecidentify_thar(thar_ec_ref, coordlist=coordlist)
    else:
        # 8a-alt – reuse external identified reference via ecreidentify.
        ref_thar_external = step8_reference_token
        if not ref_thar_external:
            raise RuntimeError(
                "Step 8 reuse mode requires a resolved reference token. "
                "Resolve via resolve_step8_runtime_config before calling core step logic."
            )
        if stem(ref_thar_external) == stem(thar_ec_ref):
            print(
                "  reuse mode: reference-star ThAr already matches reuse reference; "
                "skipping self-reidentify"
            )
        else:
            print(
                "  reuse mode: ecreidentify on reference star using existing identified ThAr"
            )
            _ecreidentify_thar(
                thar_ec_ref,
                ref_thar_external,
                drift_log_path=drift_log_path,
                drift_stage="step8_reuse",
            )

    try:
        mark_spectrum_as_reference(thar_ec_ref)
    except Exception as exc:
        print(f"  [warn] could not stamp reference-star self REFSPEC header: {exc}")

    print("  Step 8 complete: reference ThAr wavelength solution ready for steps 9/10/11")

    return ref_star


def auto_wavelength_propagation(crr2_outputs, thar_outputs, ref_star,
                                coordlist="linelists$thar.dat",
                                drift_log_path=None,
                                step9_gate_mode="warn",
                                step9_min_found_frac=0.05,
                                step9_min_fit_frac=0.05,
                                step9_max_rms=0.30,
                                star_geometry=None,
                                aps_per_star=4,
                                chain_reid=False):
    """Step 9: automatic line-ID propagation and required review for non-reference stars."""
    section_banner("Step 9 – Automatic line-ID propagation (ecreidentify + review)")
    iraf.noao()
    iraf.echelle()
    iraf.onedspec()

    thar_ec_ref = thar_outputs[ref_star]
    active_reference_star = ref_star
    active_reference_thar = thar_ec_ref
    _assert_local_apertures(thar_ec_ref, expected_count=aps_per_star)
    print(f"  Step 9 reference mode: {'chain-reid' if chain_reid else 'fixed-reference'}")
    if chain_reid:
        print(
            "  chain-reid: each successfully reviewed ThAr becomes the "
            "reference for the next star in the Step-9 processing order."
        )

    reviewed_thar_outputs = {}
    failed_gate_stars = []

    if step9_gate_mode != "off":
        gate_action = "warn-only (continue review on failures)"
        if step9_gate_mode == "strict":
            gate_action = "strict (skip review on failure)"
        print(
            "  Step 9 gate: "
            f"mode={step9_gate_mode}, "
            f"min_found_frac={step9_min_found_frac:.3f}, "
            f"min_fit_frac={step9_min_fit_frac:.3f}, "
            f"max_rms={step9_max_rms:.3f}, "
            f"action={gate_action}"
        )

    if chain_reid:
        star_order = star_order_one_direction_from_reference(
            ref_star,
            list(crr2_outputs.keys()),
        )
        order_note = "one-direction numeric chain from reference (--chain-reid)"
    else:
        star_order = star_order_by_distance_from_reference(
            star_geometry,
            ref_star,
            list(crr2_outputs.keys()),
        )
        order_note = "bidirectional outward from reference when geometry is available"
    print(
        "  Step 9 order: "
        + ", ".join(f"{int(s):02d}" for s in star_order)
        + f" ({order_note})"
    )

    for star in star_order:
        obj_crr2 = crr2_outputs[star]
        thar_ec = thar_outputs[star]

        print(f"\n  -- Star {star:02d}")
        print(f"     object : {obj_crr2}")
        print(f"     thar   : {thar_ec}")
        print(f"     output : {thar_ec}  (reviewed line IDs in IRAF DB)")

        if star == ref_star:
            print("     (reference star — solution from step 8)")
            reviewed_thar_outputs[star] = thar_ec
        else:
            _assert_local_apertures(thar_ec, expected_count=aps_per_star)
            gate_failed = False
            reference_for_this_star = active_reference_thar if chain_reid else thar_ec_ref
            reference_star_for_this_star = active_reference_star if chain_reid else ref_star
            print("     ecreidentify on real target")
            print(f"     reference star   : {reference_star_for_this_star:02d}")
            print(f"     reference ThAr   : {reference_for_this_star}")
            metrics = _ecreidentify_thar(
                thar_ec,
                reference_for_this_star,
                drift_log_path=drift_log_path,
                drift_stage="step9_chain_reid" if chain_reid else "step9_propagation",
            )
            print("     ecreidentify     : complete")

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
                    if step9_gate_mode == "strict":
                        print("     gate action: strict -> skipping review on this star")
                        failed_gate_stars.append(star)
                        gate_failed = True
                    else:
                        print("     gate action: warn -> continuing to review")

            if not gate_failed:
                print("     review on real target")
                _review_reidentified_lines(thar_ec, coordlist=coordlist)

                stamp_ok = True
                try:
                    mark_spectrum_as_reference(thar_ec)
                except Exception as exc:
                    stamp_ok = False
                    print(f"     [warn] self-reference stamp failed for real target ThAr: {exc}")

                print("     reviewed real target DB ready")
                reviewed_thar_outputs[star] = thar_ec

                if chain_reid:
                    if stamp_ok:
                        active_reference_star = star
                        active_reference_thar = thar_ec
                        print(
                            f"     chain update     : next reference star={active_reference_star:02d} "
                            f"ThAr={active_reference_thar}"
                        )
                    else:
                        print(
                            "     chain update     : skipped because ThAr self-reference stamping failed"
                        )

            if gate_failed:
                continue

            continue

    if failed_gate_stars:
        failed_text = ", ".join(f"{s:02d}" for s in failed_gate_stars)
        print(f"\n  Step 9 gate skipped stars: {failed_text}")
        print("  These stars need manual fallback (step 8-style identification path).")

    return {
        "reviewed_thar_outputs": reviewed_thar_outputs,
        "failed_gate_stars": failed_gate_stars,
    }


def apply_refspec_to_objects(crr2_outputs, thar_outputs, skip_stars=None,
                             refspec_debug=False, target_stars=None):
    """Step 10: assign reviewed ThAr solutions to CR-cleaned object spectra."""
    section_banner("Step 10 – Refspec assignment to CR-cleaned object spectra")
    iraf.noao()
    iraf.onedspec()

    refspec_outputs = {}
    skip_stars = set(skip_stars or [])
    target_stars = set(int(s) for s in (target_stars or []))

    if target_stars:
        print(
            "  Step 10 star filter : "
            + ", ".join(f"{int(s):02d}" for s in sorted(target_stars))
        )

    for star in sorted(crr2_outputs):
        if target_stars and star not in target_stars:
            print(f"\n  -- Star {star:02d}")
            print("     skip reason      : filtered by --step10-stars")
            continue

        obj_crr2 = crr2_outputs[star]
        thar_ec = thar_outputs.get(star)

        print(f"\n  -- Star {star:02d}")
        print(f"     object file      : {obj_crr2}")

        if thar_ec is None:
            print("     skip reason      : missing ThAr extraction for this star")
            continue

        print(f"     real target ThAr : {thar_ec}")

        if star in skip_stars:
            print("     skip reason      : Step 9 gate failed for this star in this run")
            continue

        found_db, db_candidates = resolve_existing_wavelength_db(thar_ec)
        if not found_db:
            print("     skip reason      : no reviewed ThAr wavelength DB solution found")
            print(f"     checked DB paths : {', '.join(db_candidates)}")
            continue

        canonical_thar = iraf_spec_token(thar_ec)
        canonical_db = ensure_canonical_wavelength_db_entry(canonical_thar, source_db=found_db)
        if not canonical_db:
            print("     skip reason      : could not normalize reviewed ThAr DB to canonical form")
            continue

        refspec_db, refspec_token, alias_created = ensure_refspec_db_entry(
            thar_ec,
            source_db=canonical_db,
        )
        if not refspec_db:
            print("     skip reason      : missing reviewed ThAr DB for REFSPEC alias preparation")
            continue

        print(f"     wavelength token : {canonical_thar}")
        print(f"     wavelength DB    : {canonical_db}")
        if alias_created:
            print(f"     refspec alias DB : {refspec_db} ({refspec_token})")

        has_self_ref, expected_token, self_ref_status = spectrum_has_self_reference(thar_ec)
        if not has_self_ref:
            print(
                "     [repair] ThAr self-reference invalid "
                f"({self_ref_status}); stamping REFSPEC1={expected_token}"
            )
            try:
                mark_spectrum_as_reference(thar_ec, source_db=canonical_db)
            except Exception as exc:
                print(f"     skip reason      : could not repair ThAr self-reference ({exc})")
                continue

            has_self_ref, expected_token, self_ref_status = spectrum_has_self_reference(thar_ec)
            if not has_self_ref:
                print(
                    "     skip reason      : ThAr self-reference invalid after repair "
                    f"({self_ref_status})"
                )
                continue

        print(f"     ThAr REFSPEC1    : {self_ref_status}")
        print("     refspec start    : real object <- real target")
        _refspec_one(obj_crr2, thar_ec, debug=refspec_debug)
        print("     refspec end      : assignment complete; verifying object header")

        has_refspec, token_or_reason = object_has_refspec_assignment(obj_crr2, normalize=True)
        if not has_refspec:
            print(f"     [fallback] refspec did not assign object header ({token_or_reason})")
            try:
                set_object_refspec_from_reference(
                    obj_crr2,
                    thar_ec,
                    source_db=canonical_db,
                )
            except Exception as exc:
                print(f"     skip reason      : object fallback assignment failed ({exc})")
                continue

            has_refspec, token_or_reason = object_has_refspec_assignment(obj_crr2, normalize=True)
            if not has_refspec:
                print(f"     skip reason      : object fallback assignment unresolved ({token_or_reason})")
                continue

        print(f"     normalized token : {token_or_reason}")
        refspec_outputs[star] = obj_crr2

    return refspec_outputs


def _canonical_refspec_value(value):
    """Extract canonical bare-root REFSPEC token from a FITS header value."""
    text = str(value or "").strip()
    if not text or text.upper() in {"INDEF", "NONE"}:
        return ""

    token = re.split(r"[\s,]+", text, maxsplit=1)[0]
    return canonical_wavelength_identity(token)


REFSPEC_TOKEN_MAXLEN = 56


def refspec_assignment_token(spectrum_path):
    """Return a REFSPEC token safe for IRAF header parsing and DB lookup."""
    canonical = iraf_spec_token(spectrum_path)
    if len(canonical) <= REFSPEC_TOKEN_MAXLEN:
        return canonical

    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    keep = max(8, REFSPEC_TOKEN_MAXLEN - len(digest) - 1)
    return f"{canonical[:keep]}_{digest}"


def ensure_refspec_db_entry(spectrum_path, source_db=None):
    """Ensure DB entry exists for the REFSPEC token used in FITS headers."""
    canonical_token = iraf_spec_token(spectrum_path)
    refspec_token = refspec_assignment_token(spectrum_path)

    db_source = source_db
    if not db_source:
        db_source, _ = resolve_existing_wavelength_db(canonical_token)
    if not db_source:
        return None, refspec_token, False

    if refspec_token == canonical_token:
        return db_source, refspec_token, False

    os.makedirs("database", exist_ok=True)
    alias_db = os.path.join("database", f"ec{refspec_token}")
    source_token = os.path.basename(str(db_source))
    if source_token.startswith("ec"):
        source_token = source_token[2:]
    else:
        source_token = canonical_token

    with open(db_source, "r", encoding="utf-8", errors="ignore") as fh:
        db_text = fh.read()

    db_text = db_text.replace(source_token, refspec_token)
    if canonical_token not in {source_token, refspec_token}:
        db_text = db_text.replace(canonical_token, refspec_token)

    with open(alias_db, "w", encoding="utf-8") as fh:
        fh.write(db_text)

    return alias_db, refspec_token, True


def spectrum_has_self_reference(spectrum_path):
    """Return (ok, expected_token, status_text) for ThAr self-reference header."""
    expected = refspec_assignment_token(spectrum_path)
    try:
        hdr = fits.getheader(spectrum_path)
    except Exception as exc:
        return False, expected, f"header read failed ({exc})"

    if "REFSPEC1" not in hdr:
        return False, expected, "missing REFSPEC1"

    actual = _canonical_refspec_value(hdr.get("REFSPEC1", ""))
    if not actual:
        return False, expected, "empty/invalid REFSPEC1"

    if actual != expected:
        return False, expected, f"REFSPEC1={actual} expected={expected}"

    return True, expected, actual


def mark_spectrum_as_reference(spectrum_path, source_db=None):
    """Mark a spectrum as a self-reference by writing REFSPEC1=<self token>."""
    canonical_token = iraf_spec_token(spectrum_path)
    token = refspec_assignment_token(spectrum_path)
    alias_db = None
    alias_created = False

    if token != canonical_token:
        alias_db, token, alias_created = ensure_refspec_db_entry(
            spectrum_path,
            source_db=source_db,
        )
        if not alias_db:
            raise RuntimeError(
                "missing wavelength DB entry for long-token REFSPEC alias "
                f"({canonical_token} -> {token})"
            )

    changed = False

    with fits.open(spectrum_path, mode="update") as hdul:
        hdr = hdul[0].header
        old_refspec1 = str(hdr.get("REFSPEC1", "")).strip()
        old_refspec2 = str(hdr.get("REFSPEC2", "")).strip() if "REFSPEC2" in hdr else ""

        if old_refspec1 != token:
            hdr["REFSPEC1"] = token
            changed = True

        removed_refspec2 = False
        if "REFSPEC2" in hdr:
            del hdr["REFSPEC2"]
            removed_refspec2 = True
            changed = True

        if changed:
            hdul.flush()

    print(
        "  [refspec-self] "
        f"{os.path.basename(spectrum_path)}: "
        f"REFSPEC1='{old_refspec1 or '<unset>'}' -> '{token}'"
        + (f", removed REFSPEC2='{old_refspec2}'" if removed_refspec2 else "")
        + (f", alias_db='{alias_db}'" if alias_created else "")
    )
    return changed


def set_object_refspec_from_reference(obj_ec, thar_ec, source_db=None):
    """Write object-side REFSPEC1 directly from reference ThAr token."""
    _, token, _ = ensure_refspec_db_entry(thar_ec, source_db=source_db)
    if not token:
        raise RuntimeError("could not resolve REFSPEC token for object fallback assignment")

    with fits.open(obj_ec, mode="update") as hdul:
        hdr = hdul[0].header
        old_refspec1 = str(hdr.get("REFSPEC1", "")).strip()
        old_refspec2 = str(hdr.get("REFSPEC2", "")).strip() if "REFSPEC2" in hdr else ""

        hdr["REFSPEC1"] = token
        removed_refspec2 = False
        if "REFSPEC2" in hdr:
            del hdr["REFSPEC2"]
            removed_refspec2 = True
        hdul.flush()

    print(
        "     [fallback] object REFSPEC1 set directly: "
        f"'{old_refspec1 or '<unset>'}' -> '{token}'"
        + (f", removed REFSPEC2='{old_refspec2}'" if removed_refspec2 else "")
    )
    return token


def object_has_refspec_assignment(obj_ec, normalize=False):
    """Return (ok, token_or_reason) for object-side refspec assignment readiness."""
    try:
        if normalize:
            with fits.open(obj_ec, mode="update") as hdul:
                hdr = hdul[0].header
                refspec_keys = [
                    k for k in hdr.keys()
                    if str(k) == "REFSPEC" or re.match(r"^REFSPEC\d+$", str(k))
                ]
                if not refspec_keys:
                    return False, "missing REFSPEC assignment in object header"

                found_token = ""
                changed = False
                for key in sorted(refspec_keys):
                    raw = str(hdr.get(key, "")).strip()
                    token = _canonical_refspec_value(raw)
                    if token and raw != token:
                        hdr[key] = token
                        changed = True
                    if token and not found_token:
                        found_token = token

                if changed:
                    hdul.flush()

                if found_token:
                    return True, found_token
        else:
            hdr = fits.getheader(obj_ec)
    except Exception as exc:
        return False, f"could not read FITS header ({exc})"

    if normalize:
        return False, "REFSPEC assignment is empty/undefined after normalization"

    refspec_keys = [
        k for k in hdr.keys()
        if str(k) == "REFSPEC" or re.match(r"^REFSPEC\d+$", str(k))
    ]
    if not refspec_keys:
        return False, "missing REFSPEC assignment in object header"

    for key in sorted(refspec_keys):
        token = _canonical_refspec_value(hdr.get(key, ""))
        if token:
            return True, token

    return False, "REFSPEC assignment is empty/undefined in object header"


def thar_db_has_dispersion_function(thar_ref_token_or_path, resolved_db_path=None):
    """Return (ok, details) for usable echelle dispersion function in ThAr DB."""
    ref_token = canonical_wavelength_identity(thar_ref_token_or_path)
    if not ref_token:
        return False, "empty ThAr reference token"

    db_path = resolved_db_path
    db_candidates = []
    if not db_path:
        db_path, db_candidates = resolve_existing_wavelength_db(ref_token)
    if not db_path:
        return False, (
            f"missing wavelength DB for REFSPEC token '{ref_token}' "
            f"(checked: {', '.join(db_candidates)})"
        )

    db_path = ensure_canonical_wavelength_db_entry(ref_token, source_db=db_path)
    if not db_path:
        return False, f"could not resolve canonical wavelength DB for REFSPEC token '{ref_token}'"

    try:
        with open(db_path, "r") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return False, f"could not read wavelength DB '{db_path}' ({exc})"

    exact_found = False
    exact_with_coeff = False
    matching_records = set()
    current_image = None
    current_image_raw = None
    current_coeff = None

    def _flush_record(image_token, image_raw, coeff_count):
        nonlocal exact_found, exact_with_coeff, matching_records
        if image_token is None:
            return
        image_text = canonical_wavelength_identity(image_token)
        if not image_text:
            return
        has_coeff = coeff_count is not None and coeff_count > 0
        if image_text == ref_token:
            exact_found = True
            if image_raw:
                matching_records.add(str(image_raw).strip())
            if has_coeff:
                exact_with_coeff = True
            return

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        if line.startswith("begin"):
            _flush_record(current_image, current_image_raw, current_coeff)
            parts = line.split()
            current_image_raw = parts[2] if len(parts) >= 3 else None
            current_image = canonical_wavelength_identity(current_image_raw)
            current_coeff = None
            continue

        if current_image is None:
            continue

        if line.startswith("coefficients"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current_coeff = int(float(parts[1]))
                except ValueError:
                    current_coeff = 0

    _flush_record(current_image, current_image_raw, current_coeff)

    if exact_with_coeff:
        return True, db_path

    if exact_found:
        record_text = ", ".join(sorted(matching_records)) if matching_records else ref_token
        return False, (
            f"DB exists ({db_path}) but matching REFSPEC record(s) '{record_text}' "
            "lack usable coefficients"
        )

    return False, (
        f"DB exists ({db_path}) but contains no usable echelle dispersion coefficients"
    )


def infer_step10_ready_outputs(crr2_outputs):
    """Infer stars eligible for step 11 from object-side refspec readiness."""
    refspec_outputs = {}
    skip_reasons = {}
    readiness_sources = {}

    for star in sorted(crr2_outputs):
        obj_crr2 = crr2_outputs[star]
        has_refspec, token_or_reason = object_has_refspec_assignment(obj_crr2, normalize=True)
        if not has_refspec:
            skip_reasons[star] = token_or_reason
            continue

        ref_token = token_or_reason

        found_db, db_candidates = resolve_existing_wavelength_db(ref_token)
        if not found_db:
            skip_reasons[star] = (
                "object has REFSPEC assignment but referenced wavelength DB is missing "
                f"({', '.join(db_candidates)})"
            )
            continue

        canonical_db = ensure_canonical_wavelength_db_entry(ref_token, source_db=found_db)
        if not canonical_db:
            skip_reasons[star] = (
                "object has REFSPEC assignment but canonical DB normalization failed"
            )
            continue

        has_disp, disp_details = thar_db_has_dispersion_function(
            ref_token,
            resolved_db_path=canonical_db,
        )
        if not has_disp:
            skip_reasons[star] = (
                "object has REFSPEC assignment but DB dispersion validation failed "
                f"({disp_details})"
            )
            continue

        refspec_outputs[star] = obj_crr2
        readiness_sources[star] = f"standalone object header check ({ref_token})"

    return refspec_outputs, skip_reasons, readiness_sources


def apply_dispcor_to_objects(refspec_outputs, crr2_outputs=None, skip_reasons=None,
                             readiness_sources=None):
    """Step 11: run dispcor on the stars that successfully passed step 10."""
    section_banner("Step 11 – Dispcor wavelength linearization")
    iraf.noao()
    iraf.onedspec()

    dispcor_outputs = {}
    crr2_outputs = dict(crr2_outputs or {})
    skip_reasons = dict(skip_reasons or {})
    readiness_sources = dict(readiness_sources or {})
    star_order = sorted(set(crr2_outputs.keys()) | set(refspec_outputs.keys()))

    for star in star_order:
        input_obj = refspec_outputs.get(star, crr2_outputs.get(star))
        if input_obj is None:
            continue

        output_dc = step11_dispcor(input_obj)

        print(f"\n  -- Star {star:02d}")
        print(f"     input object     : {input_obj}")
        print(f"     output dispcor   : {output_dc}")

        if star not in refspec_outputs:
            reason = skip_reasons.get(star, "no successful Step 10 refspec output for this star")
            print(f"     skip reason      : {reason}")
            continue

        source = readiness_sources.get(star, "same-session Step 10 output")
        print(f"     readiness source : {source}")

        has_refspec, token_or_reason = object_has_refspec_assignment(input_obj, normalize=True)
        if not has_refspec:
            print(f"     skip reason      : {token_or_reason}")
            continue

        ref_token = token_or_reason
        print(f"     REFSPEC token    : {ref_token}")

        found_db, db_candidates = resolve_existing_wavelength_db(ref_token)
        if not found_db:
            print("     dispersion check : FAIL")
            print(
                "     skip reason      : "
                "referenced ThAr DB is missing for this REFSPEC token"
            )
            print(f"     checked DB paths : {', '.join(db_candidates)}")
            continue

        canonical_db = ensure_canonical_wavelength_db_entry(ref_token, source_db=found_db)
        if not canonical_db:
            print("     dispersion check : FAIL")
            print("     skip reason      : could not normalize referenced ThAr DB to canonical form")
            continue

        print(f"     resolved DB path : {canonical_db}")
        has_disp, disp_details = thar_db_has_dispersion_function(
            ref_token,
            resolved_db_path=canonical_db,
        )
        if not has_disp:
            print("     dispersion check : FAIL")
            print(f"     skip reason      : {disp_details}")
            continue

        print("     dispersion check : PASS")
        print("     dispcor start    : wavelength linearization")
        _dispcor_one(input_obj, output_dc)
        print("     dispcor end      : output written")
        dispcor_outputs[star] = output_dc

    return dispcor_outputs


def expected_step8_outputs(crr2_outputs):
    """Return deterministic step-8 outputs (no new FITS files)."""
    return {}


def expected_step9_outputs(crr2_outputs):
    """Return deterministic step-9 outputs (no new FITS files)."""
    return {}


def expected_step10_outputs(crr2_outputs):
    """Return deterministic step-10 outputs (refspec assignment in-place)."""
    return {star: path for star, path in crr2_outputs.items()}


def expected_step11_outputs(refspec_outputs):
    """Return deterministic step-11 outputs (new dispcor products)."""
    return {star: step11_dispcor(path) for star, path in refspec_outputs.items()}


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
                            9. Automatic wavelength propagation + required review
             10. Refspec assignment to CR-cleaned object spectra
             11. Dispcor wavelength linearization  ->  *_ec-crr2-dc.fits
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
    p.add_argument("--dark", help="Master dark FITS (manual override).")
    p.add_argument("--input-dir", default=".",
                   help="Night directory (raw) or proc directory for auto-discovery (default: .).")
    p.add_argument("--run-preprocess", action="store_true",
                   help="Run image_processing.py on --input-dir before echelle steps.")
    p.add_argument("--force-preprocess", action="store_true",
                   help="Force preprocessing rerun instead of reusing compatible proc products.")
    p.add_argument("--preprocess-from-step", type=int, default=None,
                   choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   help="Resume preprocessing from this step (1-10).")
    p.add_argument("--preprocess-only", action="store_true",
                   help="Run preprocessing orchestration only and stop before echelle steps.")
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
    p.add_argument(
        "--allow-mixed-nights",
        "--no-night-match",
        dest="allow_mixed_nights",
        action="store_true",
        help=(
            "Allow quartz/ThAr/object/twilight inputs to have different NIGHT "
            "metadata. With --night, the science object discovery is still "
            "restricted to that night, but calibration roles are not forced to "
            "match the object night."
        ),
    )
    p.add_argument("--shoe", default=None, choices=["B", "R", "b", "r"],
                   help="Restrict auto-discovery to this SHOE value.")
    p.add_argument("--plate", default=None,
                   help="Restrict auto-discovery to this PLATE value.")
    p.add_argument("--object-name", default=None,
                   help="Substring filter applied to auto-discovered science OBJECT.")
    p.add_argument("--yes", action="store_true",
                   help="Continue despite metadata mismatch warnings.")
    p.add_argument("--nstars", type=int, default=24,
                   help="Number of unique stars in the pattern (default: 24).")
    p.add_argument(
        "--aps-per-star",
        type=int,
        default=4,
        help=(
            "Number of apertures/orders per star used by aperture_preview.py for "
            "automatic grouping. Manual preview edits can override this."
        ),
    )
    p.add_argument("--sep", type=float, default=8.0,
                   help="Approx. spatial separation between apertures in px (default: 8).")
    p.add_argument("--nap", type=int, default=None,
                   help="Total apertures expected (default: IRAF auto-detect).")
    p.add_argument("--dispaxis", type=int, default=1, choices=[1, 2],
                   help="Dispersion axis: 1=columns, 2=rows (default: 1).")
    p.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   help="First pipeline step to execute (default: 1).")
    p.add_argument("--end-step", type=int, default=11, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   help="Last pipeline step to execute (default: 11).")
    p.add_argument("--use-auto-pattern", action="store_true",
                   help="Allow step 6 without step 5 by using fallback auto pattern.")
    p.add_argument("--affiliation-map", default=None,
                   help="Path to saved Step-5 affiliation map JSON.")
    p.add_argument("--coordlist", default="linelists$thar.dat",
                   help="IRAF line list for ecidentify steps 8-9 (default: linelists$thar.dat).")
    p.add_argument(
        "--ref-star",
        type=int,
        default=None,
        help=(
            "Reference star number for steps 8-9. "
            "Default: first available star (current behavior)."
        ),
    )
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
            "Default: context-specific reidentify_drift_<NIGHT>_<SHOE>[_<PLATE>]_<OBJECT>.csv when inferable, "
            "otherwise reidentify_drift.csv."
        ),
    )
    p.add_argument(
        "--debug-step2-quartz-only",
        action="store_true",
        help="Temporary debug mode: run Step 2 apscatter only on quartz and exit.",
    )
    p.add_argument(
        "--step9-gate-mode",
        default="warn",
        choices=["off", "warn", "strict"],
        help=(
            "Step-9 quality gate behavior: off (disabled), warn (report failures), "
            "strict (skip review for failed stars)."
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
    p.add_argument(
        "--chain-reid",
        action="store_true",
        help=(
            "Step-9 mode: use the last successfully reviewed/reidentified ThAr "
            "as the ecreidentify reference for the next star. Default is off, "
            "which uses the Step-8 reference ThAr for every star."
        ),
    )
    p.add_argument(
        "--step10-refspec-debug",
        action="store_true",
        help=(
            "Enable interactive/verbose IRAF refspec debugging in step 10 "
            "(confirm=yes, verbose=yes, override=yes)."
        ),
    )
    p.add_argument(
        "--step10-stars",
        default=None,
        help="Comma-separated star numbers to process in step 10 (others are skipped).",
    )
    p.add_argument(
        "--retrofit-thar-refspec",
        action="store_true",
        help=(
            "Scan existing extracted ThAr spectra in proc and stamp REFSPEC1=self-token "
            "when a reviewed wavelength DB solution exists."
        ),
    )
    p.add_argument(
        "--retrofit-star",
        type=int,
        action="append",
        default=None,
        help="Optional star number filter for --retrofit-thar-refspec (repeatable).",
    )
    p.add_argument(
        "--retrofit-only",
        action="store_true",
        help="Run retrofit operation and exit before pipeline steps.",
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


def parse_star_selection_csv(value, flag_name):
    """Parse comma-separated positive integer star IDs into a sorted set."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    stars = set()
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            star = int(token)
        except ValueError as exc:
            raise RuntimeError(f"{flag_name} contains non-integer token '{token}'") from exc
        if star < 1:
            raise RuntimeError(f"{flag_name} requires positive star numbers (got {star})")
        stars.add(star)

    return stars or None


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
    return normalize_step2_like_input(path)


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
    return step3_master_flat(quartz_sl)


def expected_step4_outputs(thar_sl, obj_sl, twilight_sl=None):
    """Return deterministic step-4 output paths for all roles."""
    outputs = {
        "thar_ff": step4_flat_corrected(thar_sl),
        "obj_ff": step4_flat_corrected(obj_sl),
    }
    if twilight_sl:
        outputs["twi_ff"] = step4_flat_corrected(twilight_sl)
    return outputs


def resolve_required_path(path, proc_dir=None):
    """Resolve required-path existence, preferring cwd then proc_dir for bare tokens."""
    if path is None:
        return None

    text = str(path)
    if os.path.exists(text):
        return text

    if proc_dir and not os.path.isabs(text):
        proc_candidate = os.path.join(proc_dir, text)
        if os.path.exists(proc_candidate):
            return proc_candidate

    return None


def require_existing(path, requirement, proc_dir=None):
    """Fail with a clear message if a required file is missing."""
    resolved = resolve_required_path(path, proc_dir=proc_dir)
    if resolved is None:
        extra = f" (proc_dir={proc_dir})" if proc_dir else ""
        raise RuntimeError(f"Missing required file for {requirement}: {path}{extra}")
    return resolved


def write_infiles_from_directory(input_dir, infiles_path=None, night=None, shoe=None, plate=None):
    """Create an image_processing-style infiles list from raw per-chip FITS files.

    Optional NIGHT/SHOE/PLATE filters are applied from FITS headers when provided.
    """
    fits_paths = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    raw_like = [
        path for path in fits_paths
        if re.match(r"^[br]\d{4}c[1-4]\.fits$", os.path.basename(path), re.IGNORECASE)
    ]
    if raw_like:
        fits_paths = raw_like

    def _is_dark_like(meta):
        image_type = str(meta.get("IMAGE_TYPE", "")).strip().upper()
        if image_type in {"DARK", "DARK_MASTER"}:
            return True
        exptype = str(meta.get("EXPTYPE", "")).strip().lower().replace("_", " ")
        return "dark" in exptype

    if any(v is not None for v in (night, shoe, plate)):
        filtered = []
        skipped_unreadable = 0
        for path in fits_paths:
            try:
                meta = read_required_metadata(path)
            except Exception:
                skipped_unreadable += 1
                continue

            dark_like = _is_dark_like(meta)
            if night is not None and (not dark_like) and str(meta.get("NIGHT", "")) != str(night):
                continue
            if shoe is not None and str(meta.get("SHOE", "")).upper() != str(shoe).upper():
                continue
            if plate is not None and (not dark_like) and str(meta.get("PLATE", "")).strip() != str(plate):
                continue
            filtered.append(path)

        fits_paths = filtered
        if skipped_unreadable:
            print(
                "  WARNING: skipped "
                f"{skipped_unreadable} FITS file(s) while filtering infiles by context."
            )

    if not fits_paths:
        if any(v is not None for v in (night, shoe, plate)):
            raise RuntimeError(
                "No FITS files in input directory matched requested preprocessing context: "
                f"night={night!r}, shoe={shoe!r}, plate={plate!r}"
            )
        raise RuntimeError(f"No FITS files found in input directory: {input_dir}")

    out_path = infiles_path or os.path.join(input_dir, "infiles")
    with open(out_path, "w") as fh:
        for path in fits_paths:
            fh.write(os.path.basename(path) + "\n")
    print(f"  Preprocess infiles written: {out_path} ({len(fits_paths)} files)")
    return out_path


def _preprocess_variant_stage(path):
    """Return preprocessing variant stage inferred from filename."""
    name = os.path.basename(str(path)).lower()
    if name.endswith("-mcrr.fits"):
        return "mcrr"
    if name.endswith("-d.fits"):
        return "darksub"
    if "-full" in name and name.endswith(".fits"):
        return "mosaic"
    return None


def inspect_preprocess_state(proc_dir, night=None, shoe=None, plate=None,
                             object_name=None, required_roles=None,
                             allow_mixed_nights=False):
    """Inspect proc products and summarize reusable preprocessing state."""
    required_roles = set(required_roles or ("quartz", "thar", "object", "twilight"))
    allow_mixed_nights = bool(allow_mixed_nights)
    object_filter = str(object_name or "").strip().lower()
    role_stage_sets = {role: set() for role in required_roles}
    stacked_by_role = {role: [] for role in required_roles}

    state = {
        "proc_dir": proc_dir,
        "required_roles": tuple(sorted(required_roles)),
        "global_counts": {
            "overscan_trim": len(glob.glob(os.path.join(proc_dir, "*-ot.fits"))),
            "master_bias": len(glob.glob(os.path.join(proc_dir, "Master_bias_*.fits"))),
            "mosaics": len(glob.glob(os.path.join(proc_dir, "*-full*.fits"))),
        },
        "dark_master_count": 0,
        "dark_mosaic_count": 0,
        "role_stage_sets": role_stage_sets,
        "stacked_by_role": stacked_by_role,
        "compatible_nonstacked": 0,
        "incompatible_context_count": 0,
    }

    for path in sorted(glob.glob(os.path.join(proc_dir, "*.fits"))):
        try:
            meta = read_required_metadata(path)
        except Exception:
            continue

        row_night = str(meta.get("NIGHT", ""))
        row_shoe = str(meta.get("SHOE", ""))
        row_plate = str(meta.get("PLATE", ""))
        row_role = classify_role(meta)
        row_image_type = str(meta.get("IMAGE_TYPE", "")).strip().upper()
        row_object = str(meta.get("OBJECT", ""))

        exptype_text = str(meta.get("EXPTYPE", "")).strip().lower().replace("_", " ")
        dark_like = row_image_type in {"DARK", "DARK_MASTER"} or ("dark" in exptype_text)

        context_ok = True
        if (
            (not allow_mixed_nights) and
            night is not None and
            (not dark_like) and
            row_night != str(night)
        ):
            context_ok = False
        if shoe is not None and row_shoe.upper() != str(shoe).upper():
            context_ok = False
        if plate is not None and (not dark_like) and row_plate != str(plate):
            context_ok = False
        if object_filter and row_role == "object" and object_filter not in row_object.lower():
            context_ok = False

        if not context_ok:
            if (row_role in required_roles and is_stacked_product(meta)) or row_image_type == "DARK_MASTER":
                state["incompatible_context_count"] += 1
            continue

        if row_image_type == "DARK_MASTER":
            state["dark_master_count"] += 1
        if row_image_type == "DARK" and _preprocess_variant_stage(path) == "mosaic":
            state["dark_mosaic_count"] += 1

        if row_role not in required_roles:
            continue

        if is_stacked_product(meta):
            stacked_by_role[row_role].append(path)
            continue

        stage = _preprocess_variant_stage(path)
        if stage is not None:
            role_stage_sets[row_role].add(stage)
            state["compatible_nonstacked"] += 1

    stacked_present = {role for role, items in stacked_by_role.items() if items}
    state["missing_roles"] = tuple(sorted(required_roles.difference(stacked_present)))
    return state


def choose_preprocess_plan(state, force_preprocess=False, preprocess_from_step=None):
    """Choose preprocessing execution plan from inspected proc state."""
    missing_roles = tuple(state.get("missing_roles", ()))

    if preprocess_from_step is not None:
        start_step = int(preprocess_from_step)
        return {
            "skip": False,
            "start_step": start_step,
            "resume_from_proc": start_step >= 6,
            "reason": "explicit --preprocess-from-step",
            "missing_roles": missing_roles,
        }

    if force_preprocess:
        return {
            "skip": False,
            "start_step": 1,
            "resume_from_proc": False,
            "reason": "forced rerun (--force-preprocess)",
            "missing_roles": missing_roles,
        }

    if not missing_roles:
        return {
            "skip": True,
            "start_step": None,
            "resume_from_proc": False,
            "reason": "all required stacked products already exist",
            "missing_roles": missing_roles,
        }

    stage_sets = state.get("role_stage_sets", {})
    dark_master_count = int(state.get("dark_master_count", 0))
    dark_mosaic_count = int(state.get("dark_mosaic_count", 0))

    def _has_stage(role, stage_name):
        return stage_name in set(stage_sets.get(role, set()))

    if all(_has_stage(role, "mcrr") for role in missing_roles):
        return {
            "skip": False,
            "start_step": 9,
            "resume_from_proc": True,
            "reason": "missing stacked products but CR-cleaned mosaics already exist",
            "missing_roles": missing_roles,
        }

    if dark_master_count > 0 and all(_has_stage(role, "darksub") for role in missing_roles):
        return {
            "skip": False,
            "start_step": 8,
            "resume_from_proc": True,
            "reason": "dark-subtracted mosaics found; resume at CR-cleaning/stacking",
            "missing_roles": missing_roles,
        }

    if dark_master_count > 0 and all(
        _has_stage(role, "mosaic") or _has_stage(role, "darksub") for role in missing_roles
    ):
        return {
            "skip": False,
            "start_step": 7,
            "resume_from_proc": True,
            "reason": "mosaics + dark master found; resume before dark subtraction",
            "missing_roles": missing_roles,
        }

    if dark_mosaic_count > 0 and all(_has_stage(role, "mosaic") for role in missing_roles):
        return {
            "skip": False,
            "start_step": 6,
            "resume_from_proc": True,
            "reason": "mosaic products found; resume at dark-master stage",
            "missing_roles": missing_roles,
        }

    if all(
        _has_stage(role, "mosaic") or _has_stage(role, "darksub") or _has_stage(role, "mcrr")
        for role in missing_roles
    ):
        return {
            "skip": False,
            "start_step": 9,
            "resume_from_proc": True,
            "reason": "partial downstream products found; regenerate stacked outputs only",
            "missing_roles": missing_roles,
        }

    return {
        "skip": False,
        "start_step": 1,
        "resume_from_proc": False,
        "reason": "insufficient reusable state; full preprocessing required",
        "missing_roles": missing_roles,
    }


def _print_preprocess_state_summary(state, plan):
    """Report detected preprocess state and chosen execution plan."""
    g = state.get("global_counts", {})
    print("  Preprocess state scan:")
    print(
        "    reusable globals: "
        f"overscan+trim={g.get('overscan_trim', 0)}, "
        f"master_bias={g.get('master_bias', 0)}, "
        f"mosaics={g.get('mosaics', 0)}, "
        f"dark_master={state.get('dark_master_count', 0)}"
    )
    print(f"    missing stacked roles: {list(plan.get('missing_roles', ())) or 'none'}")
    if state.get("incompatible_context_count", 0) > 0:
        print(
            "  WARNING: found "
            f"{state['incompatible_context_count']} proc product(s) outside requested context; "
            "they will not be reused."
        )
    if plan.get("skip"):
        print(f"  Preprocess decision: reuse existing products ({plan['reason']}).")
    else:
        mode = "resume-from-proc" if plan.get("resume_from_proc") else "full-from-raw"
        print(
            "  Preprocess decision: execute "
            f"step {plan['start_step']}..10 ({mode}; {plan['reason']})."
        )


def run_image_preprocessing(args, raw_input_dir, required_roles):
    """Run image_processing.py incrementally in the raw night directory when needed."""
    proc_dir = getattr(args, "proc_dir", os.path.join(raw_input_dir, "proc"))
    requested_object = args.preprocess_object or args.object_name

    state = inspect_preprocess_state(
        proc_dir,
        night=args.night,
        shoe=args.shoe,
        plate=args.plate,
        object_name=requested_object,
        required_roles=required_roles,
        allow_mixed_nights=getattr(args, "allow_mixed_nights", False),
    )
    plan = choose_preprocess_plan(
        state,
        force_preprocess=args.force_preprocess,
        preprocess_from_step=args.preprocess_from_step,
    )
    _print_preprocess_state_summary(state, plan)

    if plan["skip"]:
        return

    infiles_path = write_infiles_from_directory(
        raw_input_dir,
        args.preprocess_infiles,
        night=args.night,
        shoe=args.shoe,
        plate=args.plate,
    )
    script_path = os.path.join(os.path.dirname(__file__), "image_processing.py")
    cmd = [
        sys.executable,
        script_path,
        "--infiles",
        os.path.abspath(infiles_path),
        "--start-step",
        str(plan["start_step"]),
        "--end-step",
        "10",
    ]

    if plan.get("resume_from_proc"):
        cmd.append("--resume-from-proc")

    if args.night:
        cmd.extend(["--night", str(args.night)])
    if args.shoe:
        cmd.extend(["--shoe", str(args.shoe)])
    if args.plate:
        cmd.extend(["--plate", str(args.plate)])

    if requested_object:
        cmd.extend(["--object", requested_object])
    if args.dark:
        cmd.extend(["--dark", str(args.dark)])
    if args.preprocess_bias:
        cmd.append("--bias")
    if args.preprocess_flat:
        cmd.extend(["--flat", args.preprocess_flat])

    print("  Launching preprocessing:")
    print("    " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=raw_input_dir)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"image_processing.py failed with exit code {exc.returncode}"
        ) from exc

    # Validate that required stacked products now exist for requested context.
    post = discover_inputs(
        proc_dir,
        night=args.night,
        shoe=args.shoe,
        plate=args.plate,
        object_name=requested_object,
        required_roles=required_roles,
        allow_mixed_nights=getattr(args, "allow_mixed_nights", False),
    )
    missing_after = [role for role in required_roles if role not in post]
    if missing_after:
        raise RuntimeError(
            "Preprocessing finished but required stacked inputs are still missing: "
            + ", ".join(missing_after)
        )


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
    """Deprecated in canonical-only mode; return canonical DB path when present."""
    _ = source_quartz
    canonical = _canonical_quartz_trace_db_path(alias_quartz)
    if canonical and os.path.exists(canonical):
        return canonical
    return None


def _canonical_quartz_trace_db_path(quartz):
    """Return canonical quartz aperture DB path: database/ap.<stem(quartz)>."""
    if quartz is None:
        return None
    quartz_text = str(quartz).strip().split("[", 1)[0]
    quartz_dir = os.path.dirname(quartz_text)
    base = stem(os.path.basename(quartz_text))
    if quartz_dir:
        return os.path.join(quartz_dir, "database", f"ap.{base}")
    return os.path.join("database", f"ap.{base}")


def _legacy_quartz_trace_db_candidates(quartz):
    """Return legacy quartz DB filename variants for one-time migration/cleanup."""
    if quartz is None:
        return []

    quartz_text = str(quartz).strip().split("[", 1)[0]
    quartz_dir = os.path.dirname(quartz_text)
    base = stem(os.path.basename(quartz_text))

    def _db_join(name):
        if quartz_dir:
            return os.path.join(quartz_dir, "database", name)
        return os.path.join("database", name)

    candidates = [
        _db_join(f"ap._{base}"),
        _db_join(f"ap{base}"),
    ]

    for probe in (quartz_text, os.path.abspath(quartz_text)):
        root = os.path.splitext(probe)[0]
        token = root.replace("\\", "_").replace("/", "_")
        if token:
            candidates.append(_db_join(f"ap{token}"))

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def renumber_quartz_trace_db_apertures_sequential(quartz):
    """Normalize quartz aperture DB to canonical file and sequential aperture IDs."""
    canonical_db = _canonical_quartz_trace_db_path(quartz)
    if not canonical_db:
        raise RuntimeError("Cannot resolve canonical quartz DB path (missing quartz reference).")
    canonical_token = quartz_reference_token(quartz)

    os.makedirs(os.path.dirname(canonical_db) or ".", exist_ok=True)

    source_db = None
    if os.path.exists(canonical_db):
        source_db = canonical_db
    else:
        for candidate in _legacy_quartz_trace_db_candidates(quartz):
            if os.path.exists(candidate):
                source_db = candidate
                break

    if source_db is None:
        raise RuntimeError(
            "Step 1 postprocess could not find quartz aperture DB to renumber. "
            f"Expected canonical path: {canonical_db}"
        )

    if os.path.abspath(source_db) != os.path.abspath(canonical_db):
        shutil.copy2(source_db, canonical_db)

    with open(canonical_db, "r") as fh:
        lines = fh.readlines()

    begin_pattern = re.compile(r"^(\s*begin\s+aperture\s+)(\S+)(\s+)(-?\d+)(\s*.*)$")
    image_pattern = re.compile(r"^(\s*image\s+)(\S+)(\s*)$")
    old_ids = []
    for line in lines:
        raw = line.rstrip("\n")
        m_begin = begin_pattern.match(raw)
        if not m_begin:
            continue
        old_ids.append(int(m_begin.group(4)))

    if not old_ids:
        raise RuntimeError(
            f"Step 1 postprocess found no aperture blocks in quartz DB: {canonical_db}"
        )

    old_to_new = {}
    for old_ap in old_ids:
        if old_ap not in old_to_new:
            old_to_new[old_ap] = len(old_to_new) + 1

    rewritten = []
    changed = False
    for line in lines:
        raw = line.rstrip("\n")
        newline = "\n" if line.endswith("\n") else ""

        m_begin = begin_pattern.match(raw)
        if m_begin:
            old_ap = int(m_begin.group(4))
            new_ap = old_to_new.get(old_ap, old_ap)
            new_line = (
                f"{m_begin.group(1)}{canonical_token}{m_begin.group(3)}"
                f"{new_ap}{m_begin.group(5)}{newline}"
            )
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_image = image_pattern.match(raw)
        if m_image:
            new_line = f"{m_image.group(1)}{canonical_token}{m_image.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_ap = re.match(r"^(\s*aperture\s+)(-?\d+)(\s*)$", raw)
        if m_ap:
            old_ap = int(m_ap.group(2))
            new_ap = old_to_new.get(old_ap, old_ap)
            new_line = f"{m_ap.group(1)}{new_ap}{m_ap.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        m_beam = re.match(r"^(\s*beam\s+)(-?\d+)(\s*)$", raw)
        if m_beam:
            old_ap = int(m_beam.group(2))
            new_ap = old_to_new.get(old_ap, old_ap)
            new_line = f"{m_beam.group(1)}{new_ap}{m_beam.group(3)}{newline}"
            rewritten.append(new_line)
            if new_line != line:
                changed = True
            continue

        rewritten.append(line)

    if changed:
        with open(canonical_db, "w") as fh:
            fh.writelines(rewritten)

    for legacy in _legacy_quartz_trace_db_candidates(quartz):
        if os.path.abspath(legacy) == os.path.abspath(canonical_db):
            continue
        if os.path.exists(legacy):
            try:
                os.remove(legacy)
            except OSError:
                pass

    new_ids = list(range(1, len(old_to_new) + 1))
    print(
        "  Renumbered quartz DB apertures sequentially: "
        f"old_ids={old_ids} -> new_ids={new_ids}"
    )
    return canonical_db


def canonical_wavelength_db_path(thar_path, db_dir="./database"):
    """Return canonical wavelength DB path for a ThAr identity token/path."""
    root = canonical_wavelength_identity(thar_path)
    if not root:
        return None
    return os.path.join(db_dir, f"ec{root}")


def wavelength_db_candidates(thar_path, include_legacy=True):
    """Return candidate wavelength DB paths (canonical first, legacy read-only after)."""
    db_dir = "./database"
    canonical = canonical_wavelength_db_path(thar_path, db_dir=db_dir)
    candidates = [canonical] if canonical else []

    if include_legacy:
        tokens = [
            canonical_wavelength_identity(thar_path),
            stem(str(thar_path or "")),
            os.path.basename(str(thar_path or "")),
        ]
        for token in tokens:
            if not token:
                continue
            candidates.extend(
                [
                    f"{db_dir}/ec.{token}",
                    f"{db_dir}/ec_{token}",
                    f"{db_dir}/ec.ec{token}",
                    f"{db_dir}/ec{token}.fits",
                ]
            )

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def resolve_existing_wavelength_db(thar_path, include_legacy=True):
    """Return the first existing wavelength DB path and full candidate list."""
    candidates = wavelength_db_candidates(thar_path, include_legacy=include_legacy)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, candidates
    return None, candidates


def ensure_canonical_wavelength_db_entry(thar_path_or_token, source_db=None):
    """Ensure canonical DB path exists and has canonicalized record identity fields."""
    canonical_db = canonical_wavelength_db_path(thar_path_or_token)
    canonical_root = canonical_wavelength_identity(thar_path_or_token)
    if not canonical_db or not canonical_root:
        return None

    chosen_source = source_db
    if not chosen_source or not os.path.exists(chosen_source):
        chosen_source, _ = resolve_existing_wavelength_db(thar_path_or_token, include_legacy=True)

    os.makedirs(os.path.dirname(canonical_db) or ".", exist_ok=True)

    if chosen_source and os.path.exists(chosen_source):
        same_path = os.path.abspath(chosen_source) == os.path.abspath(canonical_db)
        if not same_path:
            needs_copy = not os.path.exists(canonical_db)
            if not needs_copy:
                try:
                    needs_copy = os.path.getmtime(chosen_source) > os.path.getmtime(canonical_db)
                except OSError:
                    needs_copy = True
            if needs_copy:
                shutil.copy2(chosen_source, canonical_db)

    if not os.path.exists(canonical_db):
        return None

    normalize_ec_database_records(canonical_db, canonical_root)
    return canonical_db


def quartz_trace_db_candidates(quartz):
    """Return canonical IRAF quartz aperture DB path only."""
    canonical = _canonical_quartz_trace_db_path(quartz)
    return [canonical] if canonical else []


def debug_quartz_trace_db_state(quartz, context):
    """Emit concise quartz DB diagnostics for dependency checks."""
    if quartz is None:
        print(f"[quartz-db:{context}] quartz=<none>")
        return

    expected = quartz_reference_token(quartz)
    candidates = quartz_trace_db_candidates(quartz)
    existing = [p for p in candidates if os.path.exists(p)]
    print(
        f"[quartz-db:{context}] quartz={quartz} expected_token={expected}"
    )
    if existing:
        print(f"[quartz-db:{context}] db_candidates_found={', '.join(existing)}")
    else:
        print(f"[quartz-db:{context}] db_candidates_missing={', '.join(candidates)}")


def require_quartz_trace_db(quartz, requirement):
    """Ensure step prerequisites include an aperture trace database."""
    if quartz is None:
        raise RuntimeError(
            f"Missing quartz reference file for {requirement}. "
            "Provide --quartz <quartz_file> or run steps 1-4 first."
        )

    candidates = quartz_trace_db_candidates(quartz)
    existing = [p for p in candidates if os.path.exists(p)]
    if existing:
        print(
            f"[quartz-db:{requirement}] dependency-ok token={quartz_reference_token(quartz)} "
            f"db={', '.join(existing)}"
        )
        return

    print(
        f"[quartz-db:{requirement}] dependency-missing token={quartz_reference_token(quartz)} "
        f"candidates={', '.join(candidates)}"
    )
    iraf_ref = quartz_reference_token(quartz)
    raise RuntimeError(
        f"Missing quartz aperture trace database for {requirement}. "
        f"Expected IRAF reference token: {iraf_ref}. "
        f"Looked for: {', '.join(candidates)}. "
        "Run step 1 first for this quartz reference."
    )


def select_quartz_trace_db_candidate(quartz, requirement, selection_cache=None):
    """Return canonical quartz trace DB path for step2+ resume workflows."""
    _ = selection_cache
    candidates = quartz_trace_db_candidates(quartz)
    existing = [p for p in candidates if os.path.exists(p)]

    if not existing:
        raise RuntimeError(
            f"No existing quartz aperture trace database found for {requirement}. "
            f"Expected IRAF token: {quartz_reference_token(quartz)}. "
            f"Looked for: {', '.join(candidates)}. "
            "Run step 1 first or provide/select a valid DB reference."
        )

    chosen = existing[0]
    print(f"[quartz-db:{requirement}] selected canonical DB: {chosen}")
    return chosen


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


def infer_plate(meta_by_role, reference_path=None, fallback_plate=None):
    """Infer PLATE from resolved role metadata with safe fallback."""
    if meta_by_role:
        for meta in meta_by_role.values():
            plate = str(meta.get("PLATE", "")).strip()
            if plate:
                return plate

    if reference_path:
        try:
            meta = read_required_metadata(reference_path)
            plate = str(meta.get("PLATE", "")).strip()
            if plate:
                return plate
        except Exception:
            pass

    return str(fallback_plate or "").strip()


def infer_object_token(meta_by_role, fallback_object=None):
    """Infer safe object token for context-specific helper products."""
    if "object" in meta_by_role:
        obj_value = _first_nonblank([
            _normalize_token_text(meta_by_role["object"].get("OBJECT", "")),
            meta_by_role["object"].get("OBJECT", ""),
        ])
        if obj_value:
            return _safe_token(obj_value)

    if fallback_object:
        return _safe_token(_normalize_token_text(fallback_object))

    return "all"


def default_affiliation_map_path(meta_by_role, reference_path=None,
                                 fallback_night=None, fallback_shoe=None,
                                 fallback_plate=None):
    """Return default affiliation-map filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    plate = infer_plate(
        meta_by_role,
        reference_path=reference_path,
        fallback_plate=fallback_plate,
    )
    plate_tag = f"_{_safe_token(plate)}" if plate else ""
    return f"affiliation_{_safe_token(night)}_{_safe_token(shoe)}{plate_tag}.json"


def default_geometry_path(meta_by_role, reference_path=None,
                          fallback_night=None, fallback_shoe=None,
                          fallback_plate=None):
    """Return default Step-5 geometry filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    plate = infer_plate(
        meta_by_role,
        reference_path=reference_path,
        fallback_plate=fallback_plate,
    )
    plate_tag = f"_{_safe_token(plate)}" if plate else ""
    return f"geometry_{_safe_token(night)}_{_safe_token(shoe)}{plate_tag}.json"


def default_extraction_pairs_path(meta_by_role, reference_path=None,
                                  fallback_night=None, fallback_shoe=None,
                                  fallback_plate=None, fallback_object=None):
    """Return default extraction-pairs CSV path for current context."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    plate = infer_plate(
        meta_by_role,
        reference_path=reference_path,
        fallback_plate=fallback_plate,
    )
    obj_token = infer_object_token(meta_by_role, fallback_object=fallback_object)
    plate_tag = f"_{_safe_token(plate)}" if plate else ""
    return (
        f"extraction_pairs_{_safe_token(night)}_{_safe_token(shoe)}"
        f"{plate_tag}_{obj_token}.csv"
    )


def default_reidentify_drift_log_path(meta_by_role, reference_path=None,
                                      fallback_night=None, fallback_shoe=None,
                                      fallback_plate=None, fallback_object=None):
    """Return default drift-log CSV path for current context."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    plate = infer_plate(
        meta_by_role,
        reference_path=reference_path,
        fallback_plate=fallback_plate,
    )
    obj_token = infer_object_token(meta_by_role, fallback_object=fallback_object)
    plate_tag = f"_{_safe_token(plate)}" if plate else ""
    return (
        f"reidentify_drift_{_safe_token(night)}_{_safe_token(shoe)}"
        f"{plate_tag}_{obj_token}.csv"
    )


def candidate_extraction_pair_indices(input_dir, night, shoe, plate=None, object_name=None):
    """Return candidate extraction-pairs CSV paths (new naming first, legacy last)."""
    if not night or not shoe:
        return []

    night_tok = _safe_token(night)
    shoe_tok = _safe_token(str(shoe).upper())
    base = f"extraction_pairs_{night_tok}_{shoe_tok}"
    candidates = []

    if plate:
        plate_tok = _safe_token(plate)
        if object_name:
            candidates.append(f"{base}_{plate_tok}_{_safe_token(_normalize_token_text(object_name))}.csv")
        candidates.extend(sorted(glob.glob(os.path.join(input_dir, f"{base}_{plate_tok}_*.csv"))))

    if object_name:
        obj_tok = _safe_token(_normalize_token_text(object_name))
        candidates.extend(sorted(glob.glob(os.path.join(input_dir, f"{base}_*_{obj_tok}.csv"))))

    candidates.extend(sorted(glob.glob(os.path.join(input_dir, f"{base}_*.csv"))))
    candidates.extend([
        os.path.join(input_dir, f"extraction_pairs_{night}_{str(shoe).upper()}.csv"),
        f"extraction_pairs_{night}_{str(shoe).upper()}.csv",
    ])

    dedup = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        dedup.append(candidate)
    return dedup


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
    # Keep trace ordering consistent with aperture_preview.load_iraf_traces,
    # which sorts parsed DB entries by aperture number and then trims/pads to nap.
    ordered_traces = sorted(
        list(trace_details.get("entries", [])),
        key=lambda entry: int(entry.get("aperture", 0)),
    )
    if len(ordered_traces) > len(pattern):
        ordered_traces = ordered_traces[: len(pattern)]
    elif len(ordered_traces) < len(pattern):
        ordered_traces.extend({} for _ in range(len(pattern) - len(ordered_traces)))

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
    for ap_idx, (center_y, star_value, trace) in enumerate(
        zip(centers, pattern, ordered_traces)
    ):
        aperture = ap_idx + 1
        star = int(star_value)
        bundle = ((star - 1) // 4 + 1) if star > 0 else None
        star_in_bundle = ((star - 1) % 4 + 1) if star > 0 else None

        aperture_records.append(
            {
                "aperture": aperture,
                "bundle": bundle,
                "star": star if star > 0 else None,
                "star_in_bundle": star_in_bundle,
                "order": order_index_by_ap.get(aperture),
                "y_center_ref": float(center_y),
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

    if os.path.exists(path):
        print(f"  [overwrite] replacing existing geometry file: {path}")
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
    if os.path.exists(path):
        print(f"  [overwrite] replacing existing affiliation map: {path}")
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
    if os.path.exists(path):
        print(f"  [overwrite] replacing existing extraction index: {path}")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["star", "object_file", "atlas_file", "apertures"])
        for star in stars:
            ap_indices = list(np.where(pattern == star)[0] + 1)
            aperture_str = _aperture_range_string(ap_indices)
            writer.writerow([star, obj_outputs[star], thar_outputs[star], aperture_str])
    print(f"  Extraction pairing index saved: {path}")


def default_star_geometry_path(meta_by_role, reference_path=None,
                               fallback_night=None, fallback_shoe=None,
                               fallback_plate=None, fallback_object=None):
    """Return default star-geometry filename for the current night+shoe."""
    night, shoe = infer_night_shoe(
        meta_by_role,
        reference_path=reference_path,
        fallback_night=fallback_night,
        fallback_shoe=fallback_shoe,
    )
    plate = infer_plate(
        meta_by_role,
        reference_path=reference_path,
        fallback_plate=fallback_plate,
    )
    obj_token = infer_object_token(meta_by_role, fallback_object=fallback_object)
    plate_tag = f"_{_safe_token(plate)}" if plate else ""
    return (
        f"star_geometry_{_safe_token(night)}_{_safe_token(shoe)}"
        f"{plate_tag}_{obj_token}.json"
    )


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
    if os.path.exists(path):
        print(f"  [overwrite] replacing existing star geometry: {path}")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  Star geometry saved  : {path}")


def load_star_geometry_table(path):
    """Load star-geometry table from disk."""
    with open(path, "r") as fh:
        return json.load(fh)


def star_order_by_distance_from_reference(star_geometry, ref_star, available_stars):
    """Return stars in bidirectional outward order around the reference star.

    When geometry is available, this alternates nearest neighbors above/below
    the reference by signed y distance (above1, below1, above2, below2, ...).
    If geometry is missing/incomplete, falls back to deterministic numeric order.
    """
    stars_sorted = sorted(int(s) for s in available_stars)
    if ref_star not in stars_sorted:
        return stars_sorted

    if not star_geometry:
        return stars_sorted

    y_by_star = {}
    for rec in star_geometry.get("stars", []):
        try:
            y_by_star[int(rec["star_id"])] = float(rec["y_star"])
        except (KeyError, TypeError, ValueError):
            continue

    if ref_star not in y_by_star:
        return stars_sorted

    y_ref = y_by_star[ref_star]
    equal_y = []
    above = []
    below = []
    without_y = []

    for star in stars_sorted:
        if star == ref_star:
            continue
        y_val = y_by_star.get(star)
        if y_val is None:
            without_y.append(star)
            continue
        dy = float(y_val - y_ref)
        if dy < 0:
            above.append((abs(dy), star))
        elif dy > 0:
            below.append((abs(dy), star))
        else:
            equal_y.append(star)

    above.sort(key=lambda item: (item[0], item[1]))
    below.sort(key=lambda item: (item[0], item[1]))
    equal_y.sort()
    without_y.sort()

    ordered = [ref_star]
    max_len = max(len(above), len(below))
    for idx in range(max_len):
        if idx < len(above):
            ordered.append(above[idx][1])
        if idx < len(below):
            ordered.append(below[idx][1])

    ordered.extend(equal_y)
    ordered.extend(without_y)
    return ordered


def star_order_one_direction_from_reference(ref_star, available_stars):
    """Return one-direction numeric chain order starting at ref_star.

    This is intended for --chain-reid. The reference must be at one edge
    of the selected stars so that the chain proceeds monotonically without
    jumping across the detector/fiber sequence.
    """
    stars_sorted = sorted(int(s) for s in available_stars)
    ref_star = int(ref_star)

    if ref_star not in stars_sorted:
        return stars_sorted

    ref_index = stars_sorted.index(ref_star)

    if ref_index == 0:
        return stars_sorted

    if ref_index == len(stars_sorted) - 1:
        return list(reversed(stars_sorted))

    raise RuntimeError(
        "--chain-reid requires --ref-star to be at one edge of the selected stars. "
        f"Available stars: {', '.join(f'{s:02d}' for s in stars_sorted)}; "
        f"ref_star={ref_star:02d}. "
        "Choose the first or last available star as the chain reference, "
        "or run without --chain-reid."
    )


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


def _build_local_aperture_mapping(ap_indices):
    """Map extracted aperture IDs for one star to local numbering 1..N."""
    apertures = [int(v) for v in ap_indices]
    if not apertures:
        raise RuntimeError("Cannot build local aperture mapping from an empty aperture list.")
    if len(set(apertures)) != len(apertures):
        raise RuntimeError(
            f"Local aperture mapping requires unique aperture IDs, got {apertures}."
        )

    sorted_apertures = sorted(apertures)
    return {int(ap): int(idx) for idx, ap in enumerate(sorted_apertures, start=1)}


def _format_aperture_mapping(aperture_mapping):
    """Return compact human-readable mapping string (e.g. '5->1, 6->2')."""
    return ", ".join(
        f"{int(src)}->{int(dst)}"
        for src, dst in sorted(aperture_mapping.items())
    )


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


def _assert_local_apertures(fits_path, expected_count=4):
    """Assert that extracted multispec APNUM apertures are local (1..N)."""
    apertures = _read_apertures_from_apnum_cards(fits_path)
    if not apertures:
        raise RuntimeError(
            f"Could not read APNUM apertures from extracted spectrum: {fits_path}. "
            "Re-run step 6 with the new local-aperture renumbering code."
        )

    if expected_count is None:
        expected = list(range(1, len(apertures) + 1))
    else:
        expected = list(range(1, int(expected_count) + 1))

    if apertures != expected:
        raise RuntimeError(
            f"Extracted spectrum has non-local APNUM apertures for {fits_path}. "
            f"Expected {expected}, got {apertures}. "
            "Re-run step 6 with the new local-aperture renumbering code."
        )


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


def _renumber_multispec_fits_apertures(fits_path, aperture_mapping):
    """Apply aperture remapping to APNUM* and WAT2 metadata in a multispec FITS."""
    with fits.open(fits_path, mode="update") as hdul:
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


def _absolute_from_launch(path_value, launch_cwd):
    """Resolve possibly-relative path against launch cwd into an absolute path."""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(launch_cwd, path_value))


def _prefer_crr2(path):
    """Prefer the CR-cleaned counterpart when available."""
    if not path:
        return path
    if path.endswith("_ec-crr2.fits"):
        return path
    if path.endswith("_ec.fits"):
        crr2 = step7_crr2(path)
        if os.path.exists(crr2):
            return crr2
    return path


def reconstruct_star_outputs_from_disk(args):
    """Reconstruct per-star object/ThAr outputs for standalone late-step runs."""
    search_dirs = [args.input_dir, os.getcwd()]
    pair_index_candidates = []
    if args.night and args.shoe:
        pair_index_candidates.extend(
            candidate_extraction_pair_indices(
                args.input_dir,
                night=args.night,
                shoe=args.shoe,
                plate=args.plate,
                object_name=args.object_name,
            )
        )

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
            "Expected extraction_pairs_<NIGHT>_<SHOE>[_<PLATE>]_<OBJECT>.csv "
            "(or legacy extraction_pairs_<NIGHT>_<SHOE>.csv) "
            "or matching *_starNN_ec*.fits pairs."
        )

    obj_outputs = {s: obj_candidates[s][1] for s in common_stars}
    thar_outputs = {s: thar_candidates[s] for s in common_stars}
    print(
        f"  Reconstructed {len(common_stars)} stars by scanning per-star extracted files in {args.input_dir}"
    )
    return obj_outputs, thar_outputs


def retrofit_thar_reference_headers(proc_dir, night=None, shoe=None, plate=None, stars=None):
    """Retrofit REFSPEC self-reference headers on existing extracted ThAr spectra."""
    section_banner("Retrofit – stamp ThAr self-reference headers")
    star_filter = set(int(s) for s in (stars or []))

    candidates = sorted(glob.glob(os.path.join(proc_dir, "*_star*_ec*.fits")))
    if not candidates:
        print("  No extracted per-star spectra found; nothing to retrofit.")
        return {
            "checked": 0,
            "updated": 0,
            "already_ok": 0,
            "skipped_no_db": 0,
            "skipped_scope": 0,
            "skipped_role": 0,
            "skipped_star": 0,
        }

    summary = {
        "checked": 0,
        "updated": 0,
        "already_ok": 0,
        "skipped_no_db": 0,
        "skipped_scope": 0,
        "skipped_role": 0,
        "skipped_star": 0,
    }

    for path in candidates:
        lower_name = os.path.basename(path).lower()
        if lower_name.endswith("-wl.fits"):
            continue

        star = _extract_star_number(path)
        if star is None:
            continue

        if star_filter and star not in star_filter:
            summary["skipped_star"] += 1
            continue

        meta = None
        role = None
        try:
            meta = read_required_metadata(path)
            role = classify_role(meta)
        except Exception:
            meta = None

        if role != "thar" and not any(tag in lower_name for tag in ("thar", "lamp", "arc")):
            summary["skipped_role"] += 1
            continue

        if meta is not None:
            if night and str(meta.get("NIGHT", "")) != str(night):
                summary["skipped_scope"] += 1
                continue
            if shoe and str(meta.get("SHOE", "")).upper() != str(shoe).upper():
                summary["skipped_scope"] += 1
                continue
            if plate and str(meta.get("PLATE", "")).strip() != str(plate):
                summary["skipped_scope"] += 1
                continue

        summary["checked"] += 1
        found_db, db_candidates = resolve_existing_wavelength_db(path)
        if not found_db:
            summary["skipped_no_db"] += 1
            print(
                f"  [retrofit skip] star {star:02d}: no wavelength DB solution "
                f"({', '.join(db_candidates)})"
            )
            continue

        canonical_db = ensure_canonical_wavelength_db_entry(path, source_db=found_db)
        if not canonical_db:
            summary["skipped_no_db"] += 1
            print(f"  [retrofit skip] star {star:02d}: canonical DB normalization failed")
            continue

        changed = mark_spectrum_as_reference(path, source_db=canonical_db)
        if changed:
            summary["updated"] += 1
        else:
            summary["already_ok"] += 1

    print(
        "  Retrofit summary: "
        f"checked={summary['checked']}, "
        f"updated={summary['updated']}, "
        f"already_ok={summary['already_ok']}, "
        f"skipped_no_db={summary['skipped_no_db']}, "
        f"skipped_scope={summary['skipped_scope']}, "
        f"skipped_role={summary['skipped_role']}, "
        f"skipped_star={summary['skipped_star']}"
    )
    return summary


def find_step2_outputs(input_dir, night, shoe, plate=None):
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
                    if plate is not None and str(meta.get("PLATE", "")) != str(plate):
                        continue
                except Exception:
                    if plate is not None:
                        continue
                    meta = None

                if role_matches_key(key, match, meta):
                    filtered.append((match, meta))
            
            if filtered:
                filtered.sort(key=lambda item: candidate_score(key, item[0], item[1]), reverse=True)
                step2_outputs[key] = filtered[0][0]
    
    return step2_outputs

def main():
    args = parse_args()
    launch_cwd = os.getcwd()

    raw_input_dir, proc_dir = resolve_processing_dirs(args.input_dir)
    raw_input_dir = os.path.abspath(raw_input_dir)
    try:
        proc_dir = anchor_proc_workdir(proc_dir)
    except Exception as exc:
        sys.exit(f"ERROR anchoring working directory to proc: {exc}")

    # Processed products are discovered and generated in proc.
    args.raw_input_dir = raw_input_dir
    args.proc_dir = proc_dir
    args.input_dir = proc_dir

    # Resolve path-like arguments to absolute paths.
    search_dirs = [launch_cwd, raw_input_dir, proc_dir]
    for attr in [
        'quartz',
        'quartz_reference',
        'thar',
        'object',
        'twilight',
        'dark',
        'step8_reference_thar',
        'preprocess_infiles',
        'preprocess_flat',
        'affiliation_map',
    ]:
        value = getattr(args, attr, None)
        if value:
            resolved_value = _resolve_existing_path(value, search_dirs)
            setattr(args, attr, _absolute_from_launch(resolved_value, launch_cwd))

    print(f"Raw input directory : {raw_input_dir}")
    print(f"Processing directory: {proc_dir}")

    if args.step9_min_found_frac < 0.0 or args.step9_min_found_frac > 1.0:
        sys.exit("ERROR: --step9-min-found-frac must be within [0, 1].")
    if args.step9_min_fit_frac < 0.0 or args.step9_min_fit_frac > 1.0:
        sys.exit("ERROR: --step9-min-fit-frac must be within [0, 1].")
    if args.step9_max_rms < 0.0:
        sys.exit("ERROR: --step9-max-rms must be non-negative.")
    if args.aps_per_star < 1:
        sys.exit("ERROR: --aps-per-star must be a positive integer.")
    if args.ref_star is not None and args.ref_star < 1:
        sys.exit("ERROR: --ref-star must be a positive integer.")
    if args.retrofit_only and not args.retrofit_thar_refspec:
        sys.exit("ERROR: --retrofit-only requires --retrofit-thar-refspec.")
    if args.retrofit_star:
        bad_retrofit_stars = [s for s in args.retrofit_star if int(s) < 1]
        if bad_retrofit_stars:
            sys.exit("ERROR: --retrofit-star values must be positive integers.")

    try:
        step10_star_filter = parse_star_selection_csv(args.step10_stars, "--step10-stars")
    except Exception as exc:
        sys.exit(f"ERROR parsing --step10-stars: {exc}")
    retrofit_star_filter = set(int(s) for s in (args.retrofit_star or []))

    try:
        selected_steps = selected_steps_from_args(args)
        required_roles = required_roles_for_steps(selected_steps)
    except Exception as exc:
        sys.exit(f"ERROR parsing step range: {exc}")

    if args.preprocess_only and not args.run_preprocess:
        sys.exit("ERROR: --preprocess-only requires --run-preprocess.")
    if args.preprocess_from_step is not None and not args.run_preprocess:
        sys.exit("ERROR: --preprocess-from-step requires --run-preprocess.")
    if args.force_preprocess and not args.run_preprocess:
        sys.exit("ERROR: --force-preprocess requires --run-preprocess.")

    if args.run_preprocess:
        section_banner("Preprocessing prelude (image_processing.py)")
        try:
            run_image_preprocessing(args, raw_input_dir=raw_input_dir, required_roles=required_roles)
        except Exception as exc:
            sys.exit(f"ERROR preprocessing: {exc}")

    if args.preprocess_only:
        print("  Preprocess-only run complete; exiting before echelle steps.")
        return

    if args.retrofit_thar_refspec:
        try:
            retrofit_thar_reference_headers(
                proc_dir,
                night=args.night,
                shoe=args.shoe,
                plate=args.plate,
                stars=retrofit_star_filter,
            )
        except Exception as exc:
            sys.exit(f"ERROR retrofitting ThAr REFSPEC headers: {exc}")

    if args.retrofit_only:
        print("  Retrofit-only run complete; exiting before echelle steps.")
        return

    try:
        resolved = resolve_inputs(args, required_roles=required_roles)
    except Exception as exc:
        sys.exit(f"ERROR resolving inputs: {exc}")

    resolved = {
        role: _absolute_from_launch(
            _resolve_existing_path(path, [launch_cwd, raw_input_dir, proc_dir]),
            launch_cwd,
        )
        for role, path in resolved.items()
    }

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
            ("raw_dir",  raw_input_dir),
            ("proc_dir", proc_dir),
            ("quartz",   quartz if quartz else "<not required>"),
            ("quartz_ref", quartz_ref if quartz_ref else "<not required>"),
            ("thar",     thar if thar else "<not required>"),
            ("object",   obj if obj else "<not required>"),
            ("twilight", twilight if twilight else "<not required>"),
            ("n_stars",  args.nstars),
            ("aps_per_star", args.aps_per_star),
            ("sep",      f"{args.sep} px"),
            ("dispaxis", args.dispaxis),
            ("mixedN",   "yes" if args.allow_mixed_nights else "no"),
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
            f"IMAGE_TYPE={meta.get('IMAGE_TYPE', ''):<10} OBJECT={meta['OBJECT']:<24} "
            f"NIGHT={meta['NIGHT']} SHOE={meta['SHOE']} PLATE={meta.get('PLATE', '')}"
        )
    print("="*72)

    load_packages()
    try:
        iraf.cd(proc_dir)
    except Exception as exc:
        sys.exit(f"ERROR anchoring IRAF directory to proc: {exc}")
    iraf.echelle.dispaxis = args.dispaxis

    drift_log_path = None
    if any(s in selected_steps for s in (8, 9)):
        if args.drift_log:
            drift_log_path = args.drift_log
        elif args.night and args.shoe:
            drift_log_path = default_reidentify_drift_log_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
                fallback_plate=args.plate,
                fallback_object=args.object_name,
            )
        else:
            drift_log_path = "reidentify_drift.csv"
        if os.path.exists(drift_log_path):
            os.remove(drift_log_path)
        print(f"  Reidentify drift log : {drift_log_path}")

    state = {}
    obj_outputs = {}
    thar_outputs = {}

    # Keep Python/IRAF workdirs explicitly aligned before step execution.
    try:
        proc_dir = anchor_proc_workdir(proc_dir)
    except Exception as exc:
        sys.exit(f"ERROR re-anchoring proc working directory: {exc}")

    def remember_quartz_trace_db_once(quartz_candidate, requirement):
        """Resolve canonical quartz DB for the current requirement."""
        selected = select_quartz_trace_db_candidate(
            quartz_candidate,
            requirement,
        )
        state["quartz_trace_db"] = selected
        return selected
    
    # ── 1. Trace apertures on quartz ─────────────────────────────────────────
    if 1 in selected_steps:
        # If --nap is given use it, otherwise pass a large number so IRAF finds
        # all peaks it can; the user will refine interactively.
        n_ap = args.nap if args.nap else 120
        apall_trace_quartz(
            quartz,
            n_ap    = n_ap,
        )
        try:
            renumber_quartz_trace_db_apertures_sequential(quartz)
            debug_quartz_trace_db_state(quartz, "step1-sequential")
        except Exception as exc:
            sys.exit(f"ERROR step 1 quartz DB postprocess: {exc}")
    elif any(s in selected_steps for s in (2, 6)):
        try:
            if 2 in selected_steps:
                selected_db = remember_quartz_trace_db_once(
                    quartz,
                    "step 2",
                )
                debug_quartz_trace_db_state(quartz, "step2-precheck")
                state["quartz_trace_db"] = selected_db
                require_quartz_trace_db(quartz, "step 2")
            if 6 in selected_steps:
                if not quartz_ref:
                    raise RuntimeError(
                        "Step 6 requires a quartz trace reference. Provide --quartz "
                        "or --quartz-reference, or run steps 1-4 first."
                    )
                selected_step6_db = remember_quartz_trace_db_once(
                    quartz_ref,
                    "step 6",
                )
                debug_quartz_trace_db_state(
                    quartz_ref,
                    "step6-early-precheck",
                )
                state["quartz_trace_db"] = selected_step6_db
                require_quartz_trace_db(quartz_ref, "step 6")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 2. apscatter on all four images ──────────────────────────────────────
    if 2 in selected_steps:
        try:
            _step2_interactive_preflight(require_interactive=True)
        except Exception as exc:
            sys.exit(f"ERROR step 2 preflight: {exc}")

        print(
            "  [debug step2 entry] "
            f"python_cwd={os.getcwd()} proc_dir={proc_dir}"
        )

        if getattr(args, "debug_step2_quartz_only", False):
            try:
                _debug_apscatter_quartz_only(quartz, proc_dir=proc_dir)
            except Exception as exc:
                sys.exit(f"ERROR step 2 quartz-only debug: {exc}")
            return

        quartz_sl, thar_sl, obj_sl, twilight_sl = apscatter_all(
            quartz, thar, obj, twilight,
            proc_dir=proc_dir,
        )

        # Immediate validation so step-2 failures are surfaced at step 2.
        try:
            require_existing(quartz_sl, "step 2 output quartz-sl", proc_dir=proc_dir)
            require_existing(thar_sl, "step 2 output thar-sl", proc_dir=proc_dir)
            require_existing(obj_sl, "step 2 output object-sl", proc_dir=proc_dir)
            require_existing(twilight_sl, "step 2 output twilight-sl", proc_dir=proc_dir)
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

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
                step2_expected = find_step2_outputs(args.input_dir, args.night, args.shoe, args.plate)
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
                require_existing(step2_expected[key], f"step 2 output ({key})", proc_dir=proc_dir)
                state[key] = step2_expected[key]
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 3. Normalised master flat ─────────────────────────────────────────────
    if 3 in selected_steps:
        try:
            step3_check_path = resolve_required_path(state["quartz_sl"], proc_dir=proc_dir)
            print(
                "  [debug step3 precheck] "
                f"quartz_sl_token={state['quartz_sl']} "
                f"resolved_path={step3_check_path if step3_check_path else '<missing>'}"
            )
            require_existing(state["quartz_sl"], "step 3 input quartz-sl", proc_dir=proc_dir)
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")
        state["master_flat"] = make_normalised_flat(state["quartz_sl"])
    elif 4 in selected_steps:
        try:
            state["master_flat"] = expected_master_flat(state["quartz_sl"])
            require_existing(state["master_flat"], "step 3 output master flat", proc_dir=proc_dir)
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

    # ── 4. Flat-field correction ──────────────────────────────────────────────
    if 4 in selected_steps:
        try:
            require_existing(state["thar_sl"], "step 4 input thar-sl", proc_dir=proc_dir)
            require_existing(state["obj_sl"], "step 4 input object-sl", proc_dir=proc_dir)
            require_existing(state["twilight_sl"], "step 4 input twilight-sl", proc_dir=proc_dir)
            require_existing(state["master_flat"], "step 4 input master flat", proc_dir=proc_dir)
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

        thar_ff, obj_ff, twi_ff = flatcorrect_images(
            state["thar_sl"], state["obj_sl"], state["twilight_sl"], state["master_flat"],
        )
        state.update({"thar_ff": thar_ff, "obj_ff": obj_ff, "twi_ff": twi_ff})

        write_output_metadata(thar_ff, {
            "OBJECT": meta_by_role["thar"]["OBJECT"],
            "EXPTYPE": meta_by_role["thar"]["EXPTYPE"],
            "IMAGE_TYPE": meta_by_role["thar"].get("IMAGE_TYPE", ""),
            "NIGHT": meta_by_role["thar"]["NIGHT"],
            "SHOE": meta_by_role["thar"]["SHOE"],
            "PLATE": meta_by_role["thar"].get("PLATE", ""),
            "PROCSTEP": "reduce_step4_flatcorr",
        })
        write_output_metadata(obj_ff, {
            "OBJECT": meta_by_role["object"]["OBJECT"],
            "EXPTYPE": meta_by_role["object"]["EXPTYPE"],
            "IMAGE_TYPE": meta_by_role["object"].get("IMAGE_TYPE", ""),
            "NIGHT": meta_by_role["object"]["NIGHT"],
            "SHOE": meta_by_role["object"]["SHOE"],
            "PLATE": meta_by_role["object"].get("PLATE", ""),
            "PROCSTEP": "reduce_step4_flatcorr",
        })
        write_output_metadata(twi_ff, {
            "OBJECT": meta_by_role["twilight"]["OBJECT"],
            "EXPTYPE": meta_by_role["twilight"]["EXPTYPE"],
            "IMAGE_TYPE": meta_by_role["twilight"].get("IMAGE_TYPE", ""),
            "NIGHT": meta_by_role["twilight"]["NIGHT"],
            "SHOE": meta_by_role["twilight"]["SHOE"],
            "PLATE": meta_by_role["twilight"].get("PLATE", ""),
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
            debug_quartz_trace_db_state(preview_quartz, "step5-precheck")
            require_quartz_trace_db(preview_quartz, "step 5")
        except Exception as exc:
            sys.exit(f"ERROR dependency check: {exc}")

        map_path = args.affiliation_map or default_affiliation_map_path(
            meta_by_role,
            reference_path=preview_quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
            fallback_plate=args.plate,
        )
        geometry_path = default_geometry_path(
            meta_by_role,
            reference_path=preview_quartz,
            fallback_night=args.night,
            fallback_shoe=args.shoe,
            fallback_plate=args.plate,
        )
        preview_out = os.path.splitext(map_path)[0] + "_preview_map.txt"

        try:
            print(f"  Aperture preview auto-grouping: {args.aps_per_star} apertures per star")
            centers, pattern = run_aperture_preview(
                preview_quartz,
                n_stars = args.nstars,
                sep     = args.sep,
                preview_out=preview_out,
                aps_per_star=args.aps_per_star,
            )
        except Exception as exc:
            sys.exit(f"ERROR step 5 preview: {exc}")
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
            fallback_plate=args.plate,
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
            _centers, pattern = _fallback_pattern(
                state["obj_ff"],
                args.nstars,
                args.sep,
                aps_per_star=args.aps_per_star,
            )
            state["pattern"] = pattern
            print(f"  Auto pattern generated: {len(pattern)} apertures")

    # ── 6. Per-star extraction ────────────────────────────────────────────────
    if 6 in selected_steps:
        try:
            debug_quartz_trace_db_state(quartz_ref, "step6-precheck")
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
            pair_index = default_extraction_pairs_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=night,
                fallback_shoe=shoe,
                fallback_plate=args.plate,
                fallback_object=obj_meta.get("OBJECT"),
            )
            write_extraction_pairs_index(pair_index, obj_outputs, thar_outputs, state["pattern"])

            star_geometry_path = default_star_geometry_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
                fallback_plate=args.plate,
                fallback_object=obj_meta.get("OBJECT"),
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
                    _obj_ff = state.get("obj_ff", step4_flat_corrected(obj or ""))
                    obj_outputs = {
                        _s: step6_star_extract(_obj_ff, _s)
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
                                    "or run step 8/9/10/11 directly if CR-cleaned spectra are already final."
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

    elif 8 in selected_steps or 9 in selected_steps or 10 in selected_steps or 11 in selected_steps:
        # Step 8/9/10/11 without step 7: expect step-7 outputs already on disk.
        try:
            if not obj_outputs or not thar_outputs:
                obj_outputs, thar_outputs = reconstruct_star_outputs_from_disk(args)

            crr2_outputs = {star: _prefer_crr2(path) for star, path in obj_outputs.items()}
            for _s, _p in crr2_outputs.items():
                require_existing(_p, f"step 8/9/10/11 input star {_s:02d} object extracted")
            for _s, _p in thar_outputs.items():
                require_existing(_p, f"step 8/9/10/11 input star {_s:02d} thar extracted")
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

        try:
            step8_mode_resolved, step8_reference_token = resolve_step8_runtime_config(
                args.step8_mode,
                args.step8_reference_thar,
            )
        except Exception as exc:
            sys.exit(f"ERROR resolving step 8 runtime config: {exc}")

        ref_star = manual_wavelength_identification(
            crr2_outputs, thar_outputs,
            coordlist=args.coordlist,
            step8_mode=step8_mode_resolved,
            step8_reference_token=step8_reference_token,
            drift_log_path=drift_log_path,
            explicit_ref_star=args.ref_star,
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
        # Determine reference star: explicit CLI override, then step-8 state, then first available.
        if args.ref_star is not None:
            ref_star = int(args.ref_star)
        else:
            ref_star = state.get("ref_star", min(crr2_outputs.keys()))

        if ref_star not in crr2_outputs:
            available = ", ".join(f"{s:02d}" for s in sorted(crr2_outputs.keys()))
            sys.exit(
                f"ERROR: selected reference star {ref_star:02d} not found in CR-cleaned outputs. "
                f"Available: {available}"
            )
        if ref_star not in thar_outputs:
            available = ", ".join(f"{s:02d}" for s in sorted(thar_outputs.keys()))
            sys.exit(
                f"ERROR: selected reference star {ref_star:02d} not found in ThAr outputs. "
                f"Available: {available}"
            )
        star_geometry = None
        try:
            star_geometry_path = default_star_geometry_path(
                meta_by_role,
                reference_path=quartz_ref or quartz,
                fallback_night=args.night,
                fallback_shoe=args.shoe,
                fallback_plate=args.plate,
                fallback_object=args.object_name,
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
            print(f"  [warn] Could not resolve star geometry for step 9: {exc}")
        
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
        
        step9_result = auto_wavelength_propagation(
            crr2_outputs, thar_outputs, ref_star,
            coordlist=args.coordlist,
            drift_log_path=drift_log_path,
            step9_gate_mode=args.step9_gate_mode,
            step9_min_found_frac=args.step9_min_found_frac,
            step9_min_fit_frac=args.step9_min_fit_frac,
            step9_max_rms=args.step9_max_rms,
            star_geometry=star_geometry,
            aps_per_star=args.aps_per_star,
            chain_reid=args.chain_reid,
        )
        state["step9_reviewed_thar_outputs"] = step9_result.get("reviewed_thar_outputs", {})
        state["step9_failed_gate_stars"] = step9_result.get("failed_gate_stars", [])

    # ── 10. Refspec assignment to CR-cleaned objects ────────────────────────
    if 10 in selected_steps:
        crr2_outputs = state.get("crr2_outputs", {})
        if not crr2_outputs:
            sys.exit("ERROR: step 10 has no CR-cleaned spectra to calibrate.")
        if not thar_outputs:
            sys.exit(
                "ERROR dependency check: step 10 requires ThAr extractions. "
                "Run step 6 first or include it in the step range."
            )

        refspec_outputs = apply_refspec_to_objects(
            crr2_outputs,
            thar_outputs,
            skip_stars=state.get("step9_failed_gate_stars", []),
            refspec_debug=args.step10_refspec_debug,
            target_stars=step10_star_filter,
        )
        state["refspec_outputs"] = refspec_outputs

    # ── 11. Dispcor wavelength linearization ────────────────────────────────
    if 11 in selected_steps:
        crr2_outputs = state.get("crr2_outputs", {})
        if not crr2_outputs:
            sys.exit("ERROR: step 11 has no CR-cleaned spectra to calibrate.")

        if 10 in selected_steps:
            refspec_outputs = state.get("refspec_outputs", {})
            step11_skip_reasons = {
                star: "no successful Step 10 refspec output for this star"
                for star in crr2_outputs
                if star not in refspec_outputs
            }
            step11_readiness_sources = {
                star: "same-session Step 10 output"
                for star in refspec_outputs
            }
        else:
            refspec_outputs, step11_skip_reasons, step11_readiness_sources = infer_step10_ready_outputs(
                crr2_outputs,
            )
            state["refspec_outputs"] = refspec_outputs
            print(
                "  Step 11 standalone: inferred "
                f"{len(refspec_outputs)} step-10-ready stars from object refspec assignments."
            )

        dispcor_outputs = apply_dispcor_to_objects(
            refspec_outputs,
            crr2_outputs=crr2_outputs,
            skip_reasons=step11_skip_reasons,
            readiness_sources=step11_readiness_sources,
        )
        state["dispcor_outputs"] = dispcor_outputs

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
    if (
        obj_outputs or
        thar_outputs or
        state.get("crr2_outputs") or
        state.get("step9_reviewed_thar_outputs") or
        state.get("refspec_outputs") or
        state.get("dispcor_outputs")
    ):
        print("")
        for s in sorted(set(list(obj_outputs) + list(thar_outputs))):
            if s in obj_outputs:
                print(f"  Star {s:02d}  object    : {obj_outputs[s]}")
            if s in thar_outputs:
                print(f"         thar      : {thar_outputs[s]}")
            if s in state.get("crr2_outputs", {}):
                print(f"         crr2      : {state['crr2_outputs'][s]}")
            if s in state.get("step9_reviewed_thar_outputs", {}):
                print(f"         line IDs  : {state['step9_reviewed_thar_outputs'][s]} (ThAr reviewed)")
            if s in state.get("refspec_outputs", {}):
                print(f"         refspec   : {state['refspec_outputs'][s]} (assigned)")
            if s in state.get("dispcor_outputs", {}):
                print(f"         dispcor   : {state['dispcor_outputs'][s]} (final)")
        if state.get("dispcor_outputs"):
            print(
                "\n  Step 11 complete: final dispcor wavelength-linearized spectra written."
            )
        elif state.get("refspec_outputs"):
            print(
                "\n  Step 10 complete: refspec assignments applied to CR-cleaned object spectra."
            )
        elif state.get("step9_reviewed_thar_outputs"):
            print(
                "\n  Step 9 complete: ThAr line IDs propagated/reviewed directly on real targets."
            )
        elif obj_outputs:
            print(
                "\n  Next steps for each star:\n"
                "    Step 7  lineclean         ->  *_star<N>_ec-crr2.fits\n"
                "    Step 8  ecidentify/ecreidentify on reference ThAr\n"
                "    Step 9  ecreidentify + required review on real target ThAr DB\n"
                "    Step 10 refspec assignment on *_ec-crr2.fits (in-place)\n"
                "    Step 11 dispcor linearization -> *_ec-crr2-dc.fits\n"
            )


if __name__ == "__main__":
    main()
