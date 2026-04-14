"""
image_processing.py
-------------------
CCD reduction pipeline for HERMES/HYDRA multi-amplifier spectroscopy.
Converted from image_processing.ipynb.

Usage:
    python image_processing.py --infiles /path/to/data/infiles [options]

Steps executed (in order):
    1.  Load & initialise IRAF/PyRAF
    2.  Overscan fit and trim (all frames)
    3.  Bias stacking         [skipped by default - pass --bias to enable]
    4.  Bias correction       [skipped by default - pass --bias to enable]
    5.  Mosaic assembly       (4 OPAMP chips -> single 2k x 2k frame)
    5b. CCDSEC header update
    6.  Dark CRR + stacking
    7.  Dark subtraction on science frames
    8.  Cosmic-ray removal on processable non-dark mosaic frames
    9.  Frame stacking by IMAGE_TYPE
   10.  Flat-field correction  [pass --flat <flatfile> to enable]
"""

import argparse
import glob
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from pyraf import iraf

from file_handler import (
    infiles_to_lists,
    normalize_header_value,
    split_path,
    table_to_list,
)
from naming_helper import with_suffix, with_suffix_stem
import pyraf_utils


PROCESSABLE_CRR_IMAGE_TYPES = {'SCIENCE', 'TWILIGHT', 'QUARTZ', 'LAMP'}
STACKING_STRATEGY = {
    'SCIENCE': ('sum', 'none'),
    'TWILIGHT': ('median', 'median'),
    'QUARTZ': ('median', 'median'),
    'LAMP': ('median', 'median'),
}

PIPELINE_STATS = {
    'skipped_groups': 0,
    'unresolved_ambiguous': 0,
}

STAGE_COL = 'STAGE'
STAGE_RAW = 'raw'
STAGE_TRIMMED = 'trimmed'
STAGE_BIAS_CORRECTED = 'bias_corrected'
STAGE_MOSAIC = 'mosaic'
STAGE_DARK_SUBTRACTED = 'dark_subtracted'
STAGE_CR_CLEANED = 'cr_cleaned'
STAGE_STACKED = 'stacked'
STAGE_FLAT_CORRECTED = 'flat_corrected'

_LEGACY_STAGE_ALIASES = {
    'mcrr': STAGE_CR_CLEANED,
    'darksub': STAGE_DARK_SUBTRACTED,
}


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _log_step(number, title):
    """Print a consistent step banner with a timestamp."""
    print(f'\n{"="*60}')
    print(f'STEP {number:<3} {title}  [{datetime.now()}]')
    print('=' * 60)


def resolve_processing_dirs(infiles_path):
    """Resolve raw night dir + proc dir from an infiles path."""
    infiles_abs = os.path.abspath(infiles_path)
    infiles_dir = os.path.dirname(infiles_abs)

    if os.path.basename(infiles_dir) == 'proc':
        proc_dir = infiles_dir
        raw_dir = os.path.dirname(proc_dir)
    else:
        raw_dir = infiles_dir
        proc_dir = os.path.join(raw_dir, 'proc')

    return raw_dir, proc_dir, infiles_abs


def _is_blank(value):
    if value is None:
        return True
    text = str(value).strip()
    return text == '' or text.lower() == 'nan'


def _require_columns(table, columns, context):
    missing = [col for col in columns if col not in table.colnames]
    if missing:
        print(f"WARNING [{context}]: missing required columns {missing}; skipping.")
        return False
    return True


def _grouped_row_indices(table, mask, keys, context):
    """Group selected rows by keys, skipping rows with missing grouping values."""
    if not _require_columns(table, keys, context):
        return {}

    groups = {}
    skipped_rows = 0
    selected = np.where(np.asarray(mask, dtype=bool))[0]
    for idx in selected:
        values = []
        bad = False
        for key in keys:
            val = table[key][idx]
            if _is_blank(val):
                bad = True
                break
            values.append(val)
        if bad:
            skipped_rows += 1
            continue
        groups.setdefault(tuple(values), []).append(idx)

    if skipped_rows:
        PIPELINE_STATS['skipped_groups'] += skipped_rows
        print(
            f"WARNING [{context}]: skipped {skipped_rows} row(s) due to blank grouping keys {keys}."
        )
    return groups


def _warn_skip_group(context, message):
    PIPELINE_STATS['skipped_groups'] += 1
    print(f"WARNING [{context}]: {message}")


def _is_interactive_stdin():
    try:
        return os.isatty(0)
    except Exception:
        return False


def _resolve_existing_dark_path(path_value):
    """Return a usable path to an existing dark file, or None."""
    if _is_blank(path_value):
        return None

    raw = str(path_value).strip()
    probes = [
        raw,
        os.path.abspath(raw),
        os.path.join(os.getcwd(), raw),
        os.path.join(os.getcwd(), os.path.basename(raw)),
    ]

    seen = set()
    for probe in probes:
        if probe in seen:
            continue
        seen.add(probe)
        if os.path.exists(probe):
            return probe
    return None


def _normalize_stage(value, default=STAGE_RAW):
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return _LEGACY_STAGE_ALIASES.get(text, text)


def _stage_array(table, default_stage=STAGE_RAW):
    if len(table) == 0:
        return np.array([], dtype='U32')

    if STAGE_COL in table.colnames:
        return np.array(
            [_normalize_stage(v, default=default_stage) for v in table[STAGE_COL]],
            dtype='U32',
        )

    inferred = []
    for row in table:
        image_type = str(row['IMAGE_TYPE']) if 'IMAGE_TYPE' in table.colnames else ''
        exptype = str(row['EXPTYPE']) if 'EXPTYPE' in table.colnames else ''
        if 'filename_input' in table.colnames:
            filename = str(row['filename_input'])
        elif 'FILENAME' in table.colnames:
            base = str(row['FILENAME'])
            filename = base if base.lower().endswith('.fits') else f'{base}.fits'
        else:
            filename = ''
        stacked_header = bool(row['STACKED']) if 'STACKED' in table.colnames else False

        if _is_stacked_row(image_type, exptype, filename, stacked_header=stacked_header):
            inferred.append(STAGE_STACKED)
            continue

        inferred_stage = _variant_stage_from_name(filename)
        inferred.append(inferred_stage if inferred_stage is not None else default_stage)

    return np.array([_normalize_stage(v, default=default_stage) for v in inferred], dtype='U32')


def _ensure_stage_column(table, default_stage=STAGE_RAW):
    table[STAGE_COL] = _stage_array(table, default_stage=default_stage)
    return table


def _stage_mask(table, allowed_stages, default_stage=STAGE_RAW):
    stages = _stage_array(table, default_stage=default_stage)
    allowed = {_normalize_stage(stage, default='') for stage in allowed_stages}
    return np.isin(stages, list(allowed))


def _update_table_filenames(table, mask, filenames, suffix, stage=None):
    """Update FILENAME and filename_input for selected rows."""
    affix = f'-{suffix}'
    table['FILENAME'][mask] = [with_suffix_stem(f, affix) for f in filenames]
    table['filename_input'][mask] = [with_suffix(f, affix) for f in filenames]
    if stage is not None:
        _ensure_stage_column(table)
        table[STAGE_COL][mask] = _normalize_stage(stage)


def _write_iraf_lists(listpath, input_frames, output_frames, suffix):
    """Write the input and output @-list files expected by IRAF batch tasks."""
    pre, ext = os.path.splitext(os.path.relpath(listpath))
    outlist_path = f'{pre}-{suffix}{ext}'
    with open(listpath, 'w') as f:
        f.writelines(f'{frame}\n' for frame in input_frames)
    with open(outlist_path, 'w') as f:
        f.writelines(f'{frame}\n' for frame in output_frames)
    return outlist_path


def _sync_output_header_metadata(filepath, metadata, context='', overwrite_conflicts=True):
    """Ensure key metadata cards exist on an output FITS file."""
    keys = list(metadata.keys())
    with fits.open(filepath, mode='update') as hdul:
        hdr = hdul[0].header
        present_keys = [k for k in keys if k in hdr]
        if present_keys:
            snapshot = ', '.join(f"{k}={hdr.get(k)!r}" for k in present_keys)
            print(f"[meta-check] {filepath} ({context}): existing {snapshot}")
        else:
            print(f"[meta-check] {filepath} ({context}): metadata keys not present yet")

        for key, value in metadata.items():
            current = hdr.get(key)
            if current is None:
                hdr[key] = value
                continue
            if str(current) != str(value):
                if overwrite_conflicts:
                    print(f"[meta-write] {filepath}: overwrite {key}={current!r} -> {value!r}")
                    hdr[key] = value
                else:
                    print(
                        f"[meta-skip] {filepath}: keep existing {key}={current!r}, expected {value!r}"
                    )


def _normalize_opamp(value):
    text = str(value).strip().lower()
    if text.startswith('c') and len(text) > 1 and text[1:].isdigit():
        return text[1:]
    if text.isdigit():
        return str(int(text))
    return text


def _safe_token(value):
    text = str(value).strip()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^A-Za-z0-9._-]', '-', text)
    return text or 'na'


def _replace_existing_output(path, context):
    """Explicitly replace existing output files instead of implicit IRAF overwrite."""
    if os.path.exists(path):
        print(f"[overwrite] replacing existing {context}: {path}")
        os.remove(path)


def _group_keys_with_optional_plate(table, base_keys, mask=None):
    keys = list(base_keys)
    if 'PLATE' in table.colnames and 'PLATE' not in keys:
        if mask is None:
            sample = table['PLATE']
        else:
            sample = table['PLATE'][np.asarray(mask, dtype=bool)]
        if any(not _is_blank(value) for value in sample):
            keys.append('PLATE')
    return keys


def _ensure_string_columns(table, columns, width=256):
    for col in columns:
        if col in table.colnames:
            table[col] = table[col].astype(f'U{width}')


def _print_image_type_counts(table, header):
    if 'IMAGE_TYPE' not in table.colnames:
        return
    counts = Counter(str(v) for v in table['IMAGE_TYPE'])
    print(header)
    for key in sorted(counts.keys()):
        print(f"  {key:<12} {counts[key]}")


def _mask_context_tag(table, mask, include_object=False):
    """Build a stable context tag from selected rows for temp/list filenames."""
    mask = np.asarray(mask, dtype=bool)
    if len(table) == 0 or not np.any(mask):
        return 'empty'

    subset = table[mask]

    def _uniq_token(colname):
        if colname not in subset.colnames:
            return 'na'
        vals = [str(v).strip() for v in subset[colname] if not _is_blank(v)]
        if not vals:
            return 'na'
        uniq = sorted(set(vals))
        if len(uniq) == 1:
            return _safe_token(uniq[0])
        return 'multi'

    parts = [
        _uniq_token('NIGHT'),
        _uniq_token('SHOE'),
        _uniq_token('PLATE'),
    ]
    if include_object:
        if 'OBJECT_NORM' in subset.colnames:
            obj_vals = [str(v).strip() for v in subset['OBJECT_NORM'] if not _is_blank(v)]
        elif 'OBJECT' in subset.colnames:
            obj_vals = [normalize_header_value(v) for v in subset['OBJECT'] if not _is_blank(v)]
        else:
            obj_vals = []
        if not obj_vals:
            parts.append('na')
        else:
            uniq = sorted(set(obj_vals))
            parts.append(_safe_token(uniq[0] if len(uniq) == 1 else 'multi'))

    return '_'.join(parts)


def _is_stacked_row(image_type, exptype, filename, stacked_header=False):
    """Return True when a row represents a stacked product."""
    if bool(stacked_header):
        return True

    exptype_norm = str(exptype or '').strip().lower()
    if exptype_norm.endswith('_stack'):
        return True

    image_type_norm = str(image_type or '').strip().upper()
    if image_type_norm == 'DARK_MASTER':
        return True

    name = os.path.basename(str(filename or '')).lower()
    return any(tag in name for tag in ('_sstack', '_mstack', '_astack'))


def _variant_stage_from_name(filename):
    """Classify preprocessing level for a mosaic-like filename."""
    name = os.path.basename(str(filename)).lower()
    if name.endswith('-mcrr.fits'):
        return STAGE_CR_CLEANED
    if name.endswith('-d.fits'):
        return STAGE_DARK_SUBTRACTED
    if name.endswith('-f.fits'):
        return STAGE_FLAT_CORRECTED
    if name.endswith('-b.fits'):
        return STAGE_BIAS_CORRECTED
    if name.endswith('-ot.fits'):
        return STAGE_TRIMMED
    if '-full' in name and name.endswith('.fits'):
        return STAGE_MOSAIC
    return None


def _base_variant_token(filename):
    """Normalize a filename stem by stripping dark/CR suffixes."""
    root = os.path.splitext(os.path.basename(str(filename)))[0]
    root = re.sub(r'(?i)-mcrr$', '', root)
    root = re.sub(r'(?i)-d$', '', root)
    root = re.sub(r'(?i)-f$', '', root)
    return root


def _preferred_stages_for_start_step(start_step):
    """Return preferred variant stage order for resume runs."""
    if start_step >= 9:
        return [STAGE_CR_CLEANED, STAGE_DARK_SUBTRACTED, STAGE_MOSAIC]
    if start_step == 8:
        return [STAGE_DARK_SUBTRACTED, STAGE_MOSAIC, STAGE_CR_CLEANED]
    return [STAGE_MOSAIC, STAGE_DARK_SUBTRACTED, STAGE_CR_CLEANED]


def build_resume_combined_images(proc_dir, start_step, night=None, shoe=None, plate=None, object_name=None):
    """Build combined-images table from existing proc products for resume runs.

    This scans metadata-rich proc FITS outputs and selects one variant per base
    frame (prefer newest stage suitable for the requested start step).
    """
    rows = []
    role_rows = {}
    dark_master_rows = []
    preferred = _preferred_stages_for_start_step(start_step)

    for path in sorted(glob.glob(os.path.join(proc_dir, '*.fits'))):
        name = os.path.basename(path)
        try:
            hdr = fits.getheader(path)
        except Exception:
            continue

        required = ['OBJECT', 'EXPTYPE', 'IMAGE_TYPE', 'NIGHT', 'SHOE']
        if any(k not in hdr for k in required):
            continue

        image_type = str(hdr.get('IMAGE_TYPE', '')).strip().upper()
        exptype = str(hdr.get('EXPTYPE', '')).strip()
        row_night = str(hdr.get('NIGHT', ''))
        row_shoe = str(hdr.get('SHOE', ''))
        row_plate = str(hdr.get('PLATE', '')).strip()
        dark_like = image_type in {'DARK', 'DARK_MASTER'}

        if night and (not dark_like) and row_night != str(night):
            continue
        if shoe and row_shoe.upper() != str(shoe).upper():
            continue
        if plate is not None and (not dark_like) and row_plate != str(plate):
            continue
        if image_type == 'DARK_MASTER' and plate is not None and row_plate != str(plate):
            continue

        object_text = str(hdr.get('OBJECT', '')).strip()
        object_norm = normalize_header_value(object_text)
        exptype_norm = normalize_header_value(exptype)
        stacked_header = bool(hdr.get('STACKED', False))
        header_stage = _normalize_stage(hdr.get(STAGE_COL), default='')

        if image_type == 'DARK_MASTER':
            dark_master_rows.append(
                {
                    'FILENAME': os.path.splitext(name)[0],
                    'filename_input': name,
                    'OBJECT': object_text,
                    'OBJECT_NORM': object_norm,
                    'EXPTYPE': exptype,
                    'EXPTYPE_NORM': exptype_norm,
                    'IMAGE_TYPE': image_type,
                    'CLASS_REASON': 'resume_scan_dark_master',
                    'NIGHT': row_night,
                    'SHOE': row_shoe,
                    'PLATE': row_plate,
                    'STACKED': True,
                    'STACKTYP': str(hdr.get('STACKTYP', hdr.get('STACKTYPE', ''))).strip(),
                    STAGE_COL: _normalize_stage(header_stage, default=STAGE_STACKED),
                }
            )
            continue

        # Resume inputs are mosaic-like non-stacked frames.
        stage = header_stage or _variant_stage_from_name(name)
        if stage is None:
            continue
        stage = _normalize_stage(stage, default=STAGE_MOSAIC)
        if _is_stacked_row(image_type, exptype, name, stacked_header=stacked_header):
            continue

        if image_type == 'SCIENCE' and object_name:
            if normalize_header_value(object_name) != object_norm:
                continue

        key = (
            row_night,
            row_shoe,
            row_plate,
            image_type,
            object_norm,
            _base_variant_token(name),
        )
        role_rows.setdefault(key, {})[stage] = {
            'mtime': os.path.getmtime(path),
            'row': {
                'FILENAME': os.path.splitext(name)[0],
                'filename_input': name,
                'OBJECT': object_text,
                'OBJECT_NORM': object_norm,
                'EXPTYPE': exptype,
                'EXPTYPE_NORM': exptype_norm,
                'IMAGE_TYPE': image_type,
                'CLASS_REASON': f'resume_scan_{stage}',
                'NIGHT': row_night,
                'SHOE': row_shoe,
                'PLATE': row_plate,
                'STACKED': False,
                'STACKTYP': str(hdr.get('STACKTYP', hdr.get('STACKTYPE', ''))).strip(),
                STAGE_COL: stage,
            },
        }

    for stage_map in role_rows.values():
        selected = None
        for stage in preferred:
            if stage in stage_map:
                selected = stage_map[stage]['row']
                break
        if selected is None:
            # Fallback by newest timestamp if no preferred stage matched.
            newest = sorted(stage_map.values(), key=lambda x: x['mtime'], reverse=True)
            selected = newest[0]['row']
        rows.append(selected)

    if dark_master_rows:
        deduped_dark_masters = []
        seen_dark_keys = set()
        for dark_row in dark_master_rows:
            key = (
                str(dark_row.get('SHOE', '')).strip().upper(),
                str(dark_row.get('PLATE', '')).strip(),
            )
            if key in seen_dark_keys:
                continue
            seen_dark_keys.add(key)
            deduped_dark_masters.append(dark_row)
        dark_master_rows = deduped_dark_masters

    rows.extend(dark_master_rows)

    if not rows:
        return Table(rows=[])

    combined = Table(rows=rows)
    _ensure_string_columns(
        combined,
        ['FILENAME', 'filename_input', 'OBJECT', 'OBJECT_NORM', 'EXPTYPE', 'EXPTYPE_NORM', 'IMAGE_TYPE', 'CLASS_REASON', 'NIGHT', 'SHOE', 'PLATE', 'STACKTYP', STAGE_COL],
    )
    _ensure_stage_column(combined, default_stage=STAGE_MOSAIC)
    return combined


def selected_steps_from_args(args):
    """Return inclusive selected step list from CLI args."""
    if args.start_step > args.end_step:
        raise RuntimeError(
            f'Invalid preprocessing step range: start-step ({args.start_step}) '
            f'is greater than end-step ({args.end_step}).'
        )
    return list(range(args.start_step, args.end_step + 1))


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def do_median_crr(intable, listpath, mask, suffix='mcrr', stage=STAGE_CR_CLEANED):
    """Run IRAF crmedian on the rows selected by mask.

    Returns (updated_table, outlist_path).
    """
    pyraf_utils.load_crutil()

    if mask is None:
        mask = np.ones(len(intable), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    filenames = [str(p) + '.fits' for p in intable[mask]['FILENAME']]
    input_frames = [f.replace('.fits', '.fits[0]') for f in filenames]
    output_frames = [with_suffix(f, f'-{suffix}') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'crmedian: {inp} -> {out}')
        pyraf_utils.median_crr_single(inp, out)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix, stage=stage)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
    _ensure_stage_column(outtable)
    return outtable, outlist_path


def do_flatfield_correction(intable, listpath, mask, flatpath, suffix='f', stage=STAGE_FLAT_CORRECTED):
    """Apply flat-field correction via IRAF ccdproc.

    Returns (updated_table, outlist_path).
    """
    pyraf_utils.load_ccdred()

    if mask is None:
        mask = np.ones(len(intable), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    filenames = [str(p) + '.fits' for p in intable[mask]['FILENAME']]
    input_frames = [f.replace('.fits', '.fits[0]') for f in filenames]
    output_frames = [with_suffix(f, f'-{suffix}') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'flat: {inp} -> {out}')
        pyraf_utils.run_ccdproc_flat_corr(inp, out, flatpath)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix, stage=stage)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
    _ensure_stage_column(outtable)
    return outtable, outlist_path


def stack_dark_frames(comb_img_table, method='median'):
    """Stack dark frames per shoe+plate; append DARK_MASTER rows.

    Darks are treated as reusable across nights for a given shoe+plate.
    """
    if not _require_columns(comb_img_table, ['IMAGE_TYPE', 'NIGHT', 'SHOE', 'PLATE'], 'stack_dark_frames'):
        return comb_img_table

    darkmask = comb_img_table['IMAGE_TYPE'] == 'DARK'
    group_keys = ['SHOE', 'PLATE']
    groups = _grouped_row_indices(
        comb_img_table,
        darkmask,
        group_keys,
        context='step6_dark_stack',
    )

    method_suffix = method[:2]
    new_rows = []
    for key_tuple, indices in groups.items():
        values = dict(zip(group_keys, key_tuple))
        shoe = values['SHOE']
        plate_value = str(values['PLATE']).strip()
        nights = sorted(set(str(comb_img_table['NIGHT'][i]) for i in indices if not _is_blank(comb_img_table['NIGHT'][i])))
        night_tag = nights[0] if len(nights) == 1 else 'multi'
        frames = [str(comb_img_table['filename_input'][i]) + '[0]' for i in indices]
        if not frames:
            continue

        plate_tag = f'_{_safe_token(plate_value)}' if plate_value else ''
        outpath = f'{night_tag}-Dark_master-{shoe}{plate_tag}{method_suffix}.fits'
        stack_listpath = f'dark_stack_{night_tag}_{shoe}{plate_tag}{method_suffix}.list'
        with open(stack_listpath, 'w') as f:
            f.writelines(f'{frame}\n' for frame in frames)

        _replace_existing_output(outpath, context='step6_dark_stack output')
        pyraf_utils.stack_science_images(stack_listpath, outpath, mode=method)
        _sync_output_header_metadata(
            outpath,
            {
                'OBJECT': str(comb_img_table['OBJECT'][indices[0]]),
                'EXPTYPE': 'Dark_master',
                'IMAGE_TYPE': 'DARK_MASTER',
                'NIGHT': str(night_tag),
                'SHOE': str(shoe),
                'PLATE': str(plate_value),
                'STACKED': True,
                'STACKTYP': str(method).lower(),
                'PROCSTEP': 'step6_dark_stack',
                STAGE_COL: STAGE_STACKED,
            },
            context='stack_dark_frames',
            overwrite_conflicts=True,
        )
        night_label = nights[0] if len(nights) == 1 else ','.join(nights)
        print(f'Nights {night_label}, shoe {shoe}, plate {plate_value} done. Saved in {outpath}')

        template = comb_img_table[indices[0]]
        row = {c: template[c] for c in comb_img_table.colnames}
        row['FILENAME'] = outpath.replace('.fits', '')
        row['filename_input'] = outpath
        row['EXPTYPE'] = 'Dark_master'
        row['EXPTYPE_NORM'] = normalize_header_value('Dark_master')
        row['IMAGE_TYPE'] = 'DARK_MASTER'
        row['CLASS_REASON'] = 'step6_dark_stack'
        if 'PLATE' in row:
            row['PLATE'] = str(plate_value)
        row['STACKED'] = True
        row['STACKTYP'] = str(method).lower()
        row[STAGE_COL] = STAGE_STACKED
        new_rows.append(row)

    if not new_rows:
        print('WARNING [step6_dark_stack]: no dark groups were stackable.')
        return comb_img_table

    result = vstack([comb_img_table, Table(new_rows)])
    _ensure_string_columns(result, ['FILENAME', 'filename_input'])
    _ensure_stage_column(result)
    return result


def subtract_dark_mask(intable, mask, master_dark_path, label='science'):
    """Subtract a master dark from selected frames."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return intable

    subset = intable[mask]
    suffix = 'D'
    listpath = f'darksub_{label}.list'
    inlist, outlist = table_to_list(subset, listpath, suffix=suffix)

    # Resume runs may start after earlier IRAF package init points.
    pyraf_utils.load_ccdred()
    pyraf_utils.run_ccdproc_subtract_dark(inlist, outlist, dark_image=master_dark_path)

    outtable = intable.copy()
    filenames = [str(p) + '.fits' for p in subset['FILENAME']]
    _update_table_filenames(outtable, mask, filenames, suffix, stage=STAGE_DARK_SUBTRACTED)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
    _ensure_stage_column(outtable)
    return outtable


def science_frame_stacking(comb_img_table, sci_mask, mode, scale='none', image_type_label='SCIENCE'):
    """Stack frames per night/shoe and append stacked rows.

    Returns (updated_table, list_of_output_paths).
    """
    assert len(sci_mask) == len(comb_img_table), 'science mask length mismatch'
    sci_mask = np.asarray(sci_mask, dtype=bool)
    if not np.any(sci_mask):
        return comb_img_table, []

    sci_frames = comb_img_table[sci_mask]
    grouping_keys = _group_keys_with_optional_plate(
        sci_frames,
        ['NIGHT', 'SHOE'],
    )
    if image_type_label == 'SCIENCE':
        if 'OBJECT_NORM' in sci_frames.colnames:
            grouping_keys.append('OBJECT_NORM')
        elif 'OBJECT' in sci_frames.colnames:
            grouping_keys.append('OBJECT')
    elif 'OBJECT_NORM' in sci_frames.colnames:
        if any(not _is_blank(v) for v in sci_frames['OBJECT_NORM']):
            grouping_keys.append('OBJECT_NORM')

    local_groups = _grouped_row_indices(
        sci_frames,
        np.ones(len(sci_frames), dtype=bool),
        grouping_keys,
        context='step9_stack',
    )

    new_rows = []
    path_to_result = []

    for key_tuple, local_idx in local_groups.items():
        values = dict(zip(grouping_keys, key_tuple))
        night = values['NIGHT']
        shoe = values['SHOE']
        plate_value = values.get('PLATE', '')
        image_type_token = _safe_token(str(image_type_label).lower())
        subset = sci_frames[local_idx]
        if len(subset) == 0:
            continue

        paths = [str(p) for p in subset['filename_input']]

        object_grouped = ('OBJECT_NORM' in grouping_keys) or ('OBJECT' in grouping_keys)
        object_token = ''
        if object_grouped:
            if 'OBJECT_NORM' in values and not _is_blank(values['OBJECT_NORM']):
                object_token = _safe_token(values['OBJECT_NORM'])
            elif 'OBJECT' in values and not _is_blank(values['OBJECT']):
                object_token = _safe_token(normalize_header_value(values['OBJECT']))
            elif 'OBJECT_NORM' in subset.colnames:
                object_token = _safe_token(subset['OBJECT_NORM'][0])
            else:
                object_token = _safe_token(normalize_header_value(subset['OBJECT'][0]))

        stack_parts = [image_type_token, _safe_token(night), _safe_token(shoe)]
        if plate_value:
            stack_parts.append(_safe_token(plate_value))
        if object_token:
            stack_parts.append(object_token)

        stack_base = '_'.join(stack_parts)
        stack_listpath = f'stack_{stack_base}.list'
        with open(stack_listpath, 'w') as f:
            f.writelines(f"{p.replace('.fits', '.fits[0]')}\n" for p in paths)

        if image_type_label == 'SCIENCE':
            science_parts = [
                object_token or 'science',
                _safe_token(night),
                _safe_token(shoe),
            ]
            if plate_value:
                science_parts.append(_safe_token(plate_value))
            out_filepath = '_'.join(science_parts) + f'-{mode[0]}stack.fits'
        else:
            out_filepath = f'{stack_base}-{mode[0]}stack.fits'

        _replace_existing_output(out_filepath, context='step9_stack output')
        pyraf_utils.stack_science_images(stack_listpath, out_filepath, mode=mode, scale=scale)

        gain = fits.getheader(paths[0]).get('EGAIN', 1.0)
        with fits.open(out_filepath, mode='update') as hdul:
            hdul[0].data = hdul[0].data * gain
            hdul[0].header['BUNIT'] = 'electron'

        stack_meta = {
            'OBJECT': str(subset['OBJECT'][0]),
            'EXPTYPE': str(subset['EXPTYPE'][0]),
            'IMAGE_TYPE': str(image_type_label),
            'NIGHT': str(night),
            'SHOE': str(shoe),
            'PLATE': str(plate_value),
            'STACKED': True,
            'STACKTYP': str(mode).lower(),
            'STACKMOD': str(mode),
            'PROCSTEP': 'step9_stack',
            STAGE_COL: STAGE_STACKED,
        }
        _sync_output_header_metadata(
            out_filepath,
            stack_meta,
            context='science_frame_stacking',
            overwrite_conflicts=True,
        )
        print(f'Stacked (electrons): {out_filepath}')

        template = subset[0]
        row = {c: template[c] for c in comb_img_table.colnames}
        row['FILENAME'] = out_filepath.replace('.fits', '')
        row['filename_input'] = out_filepath
        row['EXPTYPE'] = str(subset['EXPTYPE'][0]) + '_stack'
        row['EXPTYPE_NORM'] = normalize_header_value(row['EXPTYPE'])
        row['IMAGE_TYPE'] = str(image_type_label)
        row['CLASS_REASON'] = 'step9_stack'
        if 'PLATE' in row:
            row['PLATE'] = str(plate_value)
        row['STACKED'] = True
        row['STACKTYP'] = str(mode).lower()
        row[STAGE_COL] = STAGE_STACKED
        new_rows.append(row)
        path_to_result.append(out_filepath)

    if not new_rows:
        return comb_img_table, []

    result = vstack([comb_img_table, Table(new_rows)])
    _ensure_string_columns(result, ['FILENAME', 'filename_input'])
    _ensure_stage_column(result)
    return result, path_to_result


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_load(list_path, extra_columns, raw_dir, proc_dir):
    _log_step(1, 'Load & Initialise')
    pyraf_utils.load_ccdred()
    print(f'Raw input directory: {raw_dir}')
    print(f'Processing directory: {proc_dir}')

    images_table, list_of_exposures, master_list = infiles_to_lists(
        list_path,
        extra_columns=extra_columns,
    )
    _ensure_stage_column(images_table, default_stage=STAGE_RAW)

    unresolved = images_table.meta.get('UNRESOLVED_AMBIGUOUS_SIGNATURES', [])
    PIPELINE_STATS['unresolved_ambiguous'] = len(unresolved)
    if unresolved:
        print(f'WARNING: unresolved ambiguous subsets: {len(unresolved)}')
        for signature in unresolved:
            print(f'  - {signature}')

    _print_image_type_counts(images_table, 'Initial IMAGE_TYPE counts:')
    return images_table, list_of_exposures, master_list


def _make_step2_lists(master_list_path, raw_dir, suffix='ot'):
    with open(master_list_path) as fh:
        entries = [line.strip() for line in fh if line.strip()]

    raw_inputs = [os.path.join(raw_dir, os.path.basename(name)) for name in entries]
    outputs = [
        with_suffix(os.path.basename(name), f'-{suffix}')
        for name in entries
    ]

    pre, ext = os.path.splitext(master_list_path)
    inlist_raw = f'{pre}-raw{ext}'
    outlist = f'{pre}-{suffix}{ext}'

    with open(inlist_raw, 'w') as fh:
        fh.writelines(f'{frame}\n' for frame in raw_inputs)
    with open(outlist, 'w') as fh:
        fh.writelines(f'{frame}\n' for frame in outputs)

    return inlist_raw, outlist


def step2_overscan_trim(images_table, master_list, raw_dir):
    _log_step(2, 'Overscan fit & trim')
    suffix = 'ot'

    for inlist in master_list:
        raw_inlist, outlist = _make_step2_lists(inlist, raw_dir=raw_dir, suffix=suffix)
        pyraf_utils.run_ccdproc_ovefit_trim(raw_inlist, outlist)

    affix = f'-{suffix}'
    images_table['filename_input'] = [with_suffix(name, affix) for name in images_table['FILENAME']]
    images_table['FILENAME'] = [with_suffix_stem(name, affix) for name in images_table['FILENAME']]
    _ensure_string_columns(images_table, ['FILENAME', 'filename_input'])
    _ensure_stage_column(images_table, default_stage=STAGE_RAW)
    images_table[STAGE_COL] = STAGE_TRIMMED
    return images_table


def step3_bias_stack(images_table):
    _log_step(3, 'Bias stacking')
    if not _require_columns(images_table, ['IMAGE_TYPE', 'SHOE', 'OPAMP'], 'step3_bias_stack'):
        return images_table

    bias_mask = images_table['IMAGE_TYPE'] == 'BIAS'
    if not np.any(bias_mask):
        print('No BIAS frames found; skipping step 3.')
        return images_table

    groups = _grouped_row_indices(
        images_table,
        bias_mask,
        ['SHOE', 'OPAMP'],
        context='step3_bias_stack',
    )

    col_names = images_table.colnames
    new_rows = []
    for (shoe, opamp), indices in groups.items():
        subset = images_table[indices]
        inlist_path, _ = table_to_list(subset, f'Bias_stack_{shoe}{opamp}.list', 'TRASH')
        outpath = f'Master_bias_{shoe}{opamp}.fits'
        _replace_existing_output(outpath, context='step3_master_bias output')
        pyraf_utils.run_zerocombine_masterbias(inlist_path, outpath)

        hdr = fits.getheader(outpath)
        row = {c: hdr.get(c, '') for c in col_names}
        row.update({
            'EXPTYPE': 'Master_bias',
            'EXPTYPE_NORM': normalize_header_value('Master_bias'),
            'IMAGE_TYPE': 'MASTER_BIAS',
            'CLASS_REASON': 'step3_master_bias',
            'FILENAME': outpath.replace('.fits', ''),
            'filename_input': outpath,
            'SHOE': shoe,
            'OPAMP': opamp,
            STAGE_COL: STAGE_STACKED,
        })
        if 'OBJECT_NORM' in col_names and _is_blank(row.get('OBJECT_NORM')):
            row['OBJECT_NORM'] = normalize_header_value(row.get('OBJECT', ''))
        new_rows.append(row)
        print(f'\t{outpath} done')

    if not new_rows:
        print('No stackable BIAS groups found; skipping step 3 append.')
        return images_table

    result = vstack([images_table, Table(new_rows)])
    _ensure_string_columns(result, ['FILENAME', 'filename_input'])
    _ensure_stage_column(result)
    return result


def step4_bias_correct(images_table):
    _log_step(4, 'Bias correction')
    if not _require_columns(images_table, ['IMAGE_TYPE', 'SHOE', 'OPAMP'], 'step4_bias_correct'):
        return images_table

    suffix = 'B'
    non_bias = ~np.isin(images_table['IMAGE_TYPE'], ['BIAS', 'MASTER_BIAS'])
    groups = _grouped_row_indices(
        images_table,
        non_bias,
        ['SHOE', 'OPAMP'],
        context='step4_bias_correct',
    )

    for (shoe, opamp), indices in groups.items():
        mask = np.zeros(len(images_table), dtype=bool)
        mask[indices] = True

        subset = images_table[mask]
        if len(subset) == 0:
            continue

        inlist_path, outlist_path = table_to_list(
            subset,
            f'Bias_correction_{shoe}{opamp}.list',
            suffix=suffix,
        )

        mb_mask = (
            (images_table['IMAGE_TYPE'] == 'MASTER_BIAS') &
            (images_table['SHOE'] == shoe) &
            (images_table['OPAMP'] == opamp)
        )
        mb_files = images_table[mb_mask]['filename_input']
        if len(mb_files) != 1:
            _warn_skip_group(
                'step4_bias_correct',
                f'expected one MASTER_BIAS for {shoe}{opamp}, got {len(mb_files)}',
            )
            continue

        pyraf_utils.run_ccdproc_bias_corr(inlist_path, outlist_path, zero_image=str(mb_files[0]))
        filenames = [str(p) + '.fits' for p in subset['FILENAME']]
        _update_table_filenames(images_table, mask, filenames, suffix, stage=STAGE_BIAS_CORRECTED)
        print(f'\t{shoe}{opamp} bias correction done')

    _ensure_string_columns(images_table, ['FILENAME', 'filename_input'])
    _ensure_stage_column(images_table)
    return images_table


def step5_mosaic(images_table):
    _log_step(5, 'Mosaic assembly')
    pyraf_utils.load_images()

    required = ['IMAGE_TYPE', 'NIGHT', 'LC-TIME', 'SHOE', 'OPAMP']
    if not _require_columns(images_table, required, 'step5_mosaic'):
        return Table(rows=[])

    source_mask = ~np.isin(images_table['IMAGE_TYPE'], ['BIAS', 'MASTER_BIAS'])
    # Group one exposure strictly by NIGHT/LC-TIME/SHOE.
    # PLATE may be blank on individual chips and must not split/drop OPAMP sets.
    group_keys = ['NIGHT', 'LC-TIME', 'SHOE']
    groups = _grouped_row_indices(
        images_table,
        source_mask,
        group_keys,
        context='step5_mosaic',
    )

    mosaics = []
    for key_tuple, indices in groups.items():
        values = dict(zip(group_keys, key_tuple))
        night = values['NIGHT']
        lctime = values['LC-TIME']
        shoe = values['SHOE']
        subset = images_table[indices].copy()

        plate_values = []
        if 'PLATE' in subset.colnames:
            plate_values = sorted(
                {str(v).strip() for v in subset['PLATE'] if not _is_blank(v)}
            )
        if len(plate_values) > 1:
            print(
                "WARNING [step5_mosaic]: conflicting non-blank PLATE values "
                f"for NIGHT={night} LC-TIME={lctime} SHOE={shoe}: {plate_values}. "
                f"Using '{plate_values[0]}'."
            )
        plate_value = plate_values[0] if plate_values else ''

        opamp_tokens = {_normalize_opamp(v) for v in subset['OPAMP']}
        expected = {'1', '2', '3', '4'}

        if opamp_tokens != expected or len(subset) != 4:
            _warn_skip_group(
                'step5_mosaic',
                f'incomplete chip coverage for NIGHT={night} LC-TIME={lctime} SHOE={shoe}: '
                f'OPAMPs={sorted(opamp_tokens)} rows={len(subset)}',
            )
            continue

        base_name = str(subset['FILENAME'][0])
        for chip in ['c1', 'c2', 'c3', 'c4']:
            base_name = base_name.replace(chip, '')

        exptype = str(subset['EXPTYPE'][0])
        out_filename = f'{exptype}-{base_name}-full.fits'
        pyraf_utils.assemble_mosaic(subset, out_filename)

        image_type = str(subset['IMAGE_TYPE'][0])
        mosaic_meta = {
            'OBJECT': str(subset['OBJECT'][0]),
            'EXPTYPE': str(exptype),
            'IMAGE_TYPE': image_type,
            'NIGHT': str(night),
            'SHOE': str(shoe),
            'PLATE': str(plate_value),
            'PROCSTEP': 'step5_mosaic',
            STAGE_COL: STAGE_MOSAIC,
        }
        _sync_output_header_metadata(
            out_filename,
            mosaic_meta,
            context='step5_mosaic',
            overwrite_conflicts=True,
        )

        row = {c: subset[0][c] for c in images_table.colnames}
        row['FILENAME'] = out_filename.replace('.fits', '')
        row['filename_input'] = out_filename
        row['OPAMP'] = '0'
        row['NIGHT'] = night
        row['LC-TIME'] = lctime
        row['SHOE'] = shoe
        if 'PLATE' in row:
            row['PLATE'] = str(plate_value)
        row['IMAGE_TYPE'] = image_type
        row['CLASS_REASON'] = 'step5_mosaic'
        row['EXPTYPE_NORM'] = normalize_header_value(row['EXPTYPE'])
        row['OBJECT_NORM'] = normalize_header_value(row['OBJECT'])
        row[STAGE_COL] = STAGE_MOSAIC
        mosaics.append(row)

    if not mosaics:
        print('WARNING [step5_mosaic]: no complete mosaic groups were generated.')
        return Table(rows=[])

    combined_images = Table(mosaics)
    _ensure_string_columns(combined_images, ['FILENAME', 'filename_input'])
    _ensure_stage_column(combined_images, default_stage=STAGE_MOSAIC)
    return combined_images


def step5b_update_ccdsec(combined_images):
    _log_step('5b', 'CCDSEC header update')
    pyraf_utils.load_imutil()
    if len(combined_images) == 0:
        print('No mosaics available; skipping step 5b.')
        return

    for filename in combined_images['filename_input']:
        filename = str(filename)
        if not filename.endswith('.fits'):
            filename += '.fits'
        if not os.path.exists(filename):
            print(f'Skipping missing file: {filename}')
            continue
        print(f'Updating CCDSEC in {filename}')
        pyraf_utils.hedit_ccdsec(filename)


def step6_dark_crr_stack(combined_images):
    _log_step(6, 'Dark CRR + stacking')
    if not _require_columns(combined_images, ['IMAGE_TYPE'], 'step6_dark_crr_stack'):
        return combined_images

    if np.any(combined_images['IMAGE_TYPE'] == 'DARK_MASTER'):
        print('Dark masters already present, skipping dark stacking.')
        return combined_images

    dmask = combined_images['IMAGE_TYPE'] == 'DARK'
    if not np.any(dmask):
        print('No DARK frames present; skipping dark CRR/stacking.')
        return combined_images

    dark_tag = _mask_context_tag(combined_images, dmask, include_object=False)
    combined_images, _ = do_median_crr(
        combined_images,
        f'darklist_{dark_tag}.list',
        mask=dmask,
    )
    print('CRR on darks done')
    combined_images = stack_dark_frames(combined_images, method='median')
    return combined_images


def step7_dark_subtract(combined_images, object_name=None, dark_override=None):
    _log_step(7, 'Dark subtraction')
    if not _require_columns(combined_images, ['IMAGE_TYPE', 'NIGHT', 'SHOE', 'PLATE'], 'step7_dark_subtract'):
        return combined_images, np.ones(len(combined_images), dtype=bool)

    non_dark_mask = ~np.isin(combined_images['IMAGE_TYPE'], ['DARK', 'DARK_MASTER'])
    target_mask = non_dark_mask & np.isin(
        combined_images['IMAGE_TYPE'],
        sorted(PROCESSABLE_CRR_IMAGE_TYPES),
    )
    if STAGE_COL in combined_images.colnames:
        target_mask = target_mask & _stage_mask(
            combined_images,
            [STAGE_MOSAIC],
            default_stage=STAGE_MOSAIC,
        )

    if object_name:
        object_norm = normalize_header_value(object_name)
        if 'OBJECT_NORM' in combined_images.colnames:
            sci_mask = combined_images['IMAGE_TYPE'] == 'SCIENCE'
            target_mask = target_mask & (
                ~sci_mask | (combined_images['OBJECT_NORM'] == object_norm)
            )
        else:
            print('WARNING [step7_dark_subtract]: OBJECT_NORM missing; cannot apply --object filter.')

    if not np.any(target_mask):
        print('No mosaic-stage processable non-dark frames selected for dark subtraction; continuing.')
        return combined_images, non_dark_mask

    if not np.any(combined_images['IMAGE_TYPE'] == 'DARK_MASTER'):
        print('No DARK_MASTER frames available; skipping dark subtraction.')
        return combined_images, non_dark_mask

    group_keys = ['SHOE', 'PLATE']

    def _same_shoe(val_a, val_b):
        return str(val_a).strip().upper() == str(val_b).strip().upper()

    def _candidate_records_from_table(shoe_value):
        rows = []
        for row in combined_images[combined_images['IMAGE_TYPE'] == 'DARK_MASTER']:
            if not _same_shoe(row['SHOE'], shoe_value):
                continue
            raw_path = str(row['filename_input']).strip()
            if not raw_path:
                continue
            resolved = _resolve_existing_dark_path(raw_path)
            rows.append(
                {
                    'path': resolved if resolved is not None else raw_path,
                    'night': str(row['NIGHT']).strip(),
                    'plate': str(row['PLATE']).strip(),
                    'source': 'table',
                }
            )
        return rows

    def _candidate_records_from_cwd(shoe_value):
        rows = []
        for path in sorted(glob.glob(os.path.join(os.getcwd(), '*.fits'))):
            try:
                hdr = fits.getheader(path)
            except Exception:
                continue
            if str(hdr.get('IMAGE_TYPE', '')).strip().upper() != 'DARK_MASTER':
                continue
            if not _same_shoe(hdr.get('SHOE', ''), shoe_value):
                continue
            rows.append(
                {
                    'path': path,
                    'night': str(hdr.get('NIGHT', '')).strip(),
                    'plate': str(hdr.get('PLATE', '')).strip(),
                    'source': 'cwd',
                }
            )
        return rows

    def _dedupe_candidates(records):
        deduped = []
        seen = set()
        for rec in records:
            key = os.path.abspath(rec['path'])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        return deduped

    def _prompt_candidate_choice(records, shoe_value):
        print(
            'INFO [step7_dark_subtract]: multiple same-SHOE DARK_MASTER candidates '
            f"for SHOE={shoe_value}."
        )
        for i, rec in enumerate(records, start=1):
            print(
                f"  {i:>2}) NIGHT={rec['night'] or 'na'} "
                f"PLATE={rec['plate'] or 'na'} FILE={rec['path']}"
            )

        while True:
            choice = input(
                f"Select DARK_MASTER [1-{len(records)}], path, or Enter to skip: "
            ).strip()
            if choice == '':
                return None
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(records):
                    return records[idx - 1]['path']
            manual = _resolve_existing_dark_path(choice)
            if manual is not None:
                return manual
            print(f"WARNING [step7_dark_subtract]: invalid selection '{choice}'.")

    def _prompt_manual_dark_path(shoe_value):
        while True:
            entered = input(
                f"No DARK_MASTER found for SHOE={shoe_value}. "
                'Enter path to master dark (or Enter to skip): '
            ).strip()
            if entered == '':
                return None
            resolved = _resolve_existing_dark_path(entered)
            if resolved is not None:
                return resolved
            print(
                f"WARNING [step7_dark_subtract]: path does not exist: {entered}"
            )

    def _master_dark_path(group_values):
        if dark_override:
            chosen = _resolve_existing_dark_path(dark_override)
            if chosen is None:
                _warn_skip_group(
                    'step7_dark_subtract',
                    f"--dark path not found: {dark_override}",
                )
                return None
            return chosen

        shoe_value = group_values['SHOE']
        plate_value = str(group_values.get('PLATE', '')).strip()
        records = _candidate_records_from_table(shoe_value)
        records.extend(_candidate_records_from_cwd(shoe_value))
        records = _dedupe_candidates(records)

        if plate_value:
            plate_matched = [
                rec for rec in records
                if str(rec.get('plate', '')).strip() == plate_value
            ]
            if plate_matched:
                records = plate_matched

        if len(records) == 1:
            return records[0]['path']

        if len(records) > 1:
            if _is_interactive_stdin():
                return _prompt_candidate_choice(records, shoe_value)
            print(
                'INFO [step7_dark_subtract]: multiple DARK_MASTER candidates found for '
                f"SHOE={shoe_value}"
                + (f" PLATE={plate_value}" if plate_value else "")
                + f"; selecting first in non-interactive mode: {records[0]['path']}"
            )
            return records[0]['path']

        if _is_interactive_stdin():
            return _prompt_manual_dark_path(shoe_value)

        _warn_skip_group(
            'step7_dark_subtract',
            'expected at least one same-SHOE DARK_MASTER for '
            f"SHOE={shoe_value}, got 0. Provide --dark or rerun interactively.",
        )
        return None

    target_groups = _grouped_row_indices(
        combined_images,
        target_mask,
        group_keys,
        context='step7_dark_subtract',
    )

    for key_tuple, indices in target_groups.items():
        values = dict(zip(group_keys, key_tuple))
        master_dark = _master_dark_path(values)
        if master_dark is None:
            continue
        group_mask = np.zeros(len(combined_images), dtype=bool)
        group_mask[indices] = True
        combined_images = subtract_dark_mask(
            combined_images,
            group_mask,
            master_dark,
            label=(
                f"{values.get('SHOE')}_{_safe_token(values.get('PLATE', ''))}"
            ),
        )

    return combined_images, non_dark_mask


def step8_crr_science(combined_images):
    _log_step(8, 'Cosmic-ray removal on processable non-dark frames')
    if not _require_columns(combined_images, ['IMAGE_TYPE'], 'step8_crr_science'):
        return combined_images

    crr_mask = np.isin(combined_images['IMAGE_TYPE'], sorted(PROCESSABLE_CRR_IMAGE_TYPES))
    crr_mask = crr_mask & _stage_mask(
        combined_images,
        [STAGE_DARK_SUBTRACTED, STAGE_MOSAIC],
        default_stage=STAGE_MOSAIC,
    )
    if not np.any(crr_mask):
        print('No SCIENCE/TWILIGHT/QUARTZ/LAMP frames to CR-clean; continuing.')
        return combined_images

    crr_tag = _mask_context_tag(combined_images, crr_mask, include_object=True)
    combined_images, _ = do_median_crr(
        combined_images,
        f'crr_{crr_tag}.list',
        mask=crr_mask,
    )
    _ensure_string_columns(combined_images, ['FILENAME', 'filename_input'])
    return combined_images


def step9_stack_science(combined_images, object_name=None):
    _log_step(9, 'Frame stacking by IMAGE_TYPE')
    if not _require_columns(combined_images, ['IMAGE_TYPE'], 'step9_stack_science'):
        return combined_images

    stacked = combined_images.copy()
    object_norm = normalize_header_value(object_name) if object_name else None

    for image_type, (mode, scale) in STACKING_STRATEGY.items():
        mask = stacked['IMAGE_TYPE'] == image_type
        mask = mask & _stage_mask(
            stacked,
            [STAGE_CR_CLEANED, STAGE_DARK_SUBTRACTED, STAGE_MOSAIC],
            default_stage=STAGE_MOSAIC,
        )
        if image_type == 'SCIENCE' and object_norm:
            if 'OBJECT_NORM' in stacked.colnames:
                mask = mask & (stacked['OBJECT_NORM'] == object_norm)
            else:
                print('WARNING [step9_stack_science]: OBJECT_NORM missing; SCIENCE object filter skipped.')

        if not np.any(mask):
            continue

        stacked, _ = science_frame_stacking(
            stacked,
            mask,
            mode=mode,
            scale=scale,
            image_type_label=image_type,
        )

    return stacked


def step10_flatfield(combined_images, flatpath, object_name=None):
    _log_step(10, 'Flat-field correction')
    if not _require_columns(combined_images, ['IMAGE_TYPE'], 'step10_flatfield'):
        return combined_images

    sci_mask = combined_images['IMAGE_TYPE'] == 'SCIENCE'
    sci_mask = sci_mask & _stage_mask(
        combined_images,
        [STAGE_STACKED],
        default_stage=STAGE_RAW,
    )
    if object_name and 'OBJECT_NORM' in combined_images.colnames:
        sci_mask = sci_mask & (combined_images['OBJECT_NORM'] == normalize_header_value(object_name))

    if not np.any(sci_mask):
        print('No SCIENCE frames selected for flat-field correction; skipping step 10.')
        return combined_images

    flat_tag = _mask_context_tag(combined_images, sci_mask, include_object=True)
    combined_images, _ = do_flatfield_correction(
        combined_images,
        f'flatlist_{flat_tag}.list',
        mask=sci_mask,
        flatpath=flatpath,
    )
    return combined_images


def _print_end_summary(table):
    _log_step('SUM', 'End-of-run summary')
    if len(table) == 0 or 'IMAGE_TYPE' not in table.colnames:
        print('No final table rows available.')
        return

    counts = Counter(str(v) for v in table['IMAGE_TYPE'])
    print('Counts by IMAGE_TYPE:')
    for key in sorted(counts.keys()):
        print(f'  {key:<12} {counts[key]}')

    print(f"UNKNOWN count      : {counts.get('UNKNOWN', 0)}")
    print(f"FIBERMAP count     : {counts.get('FIBERMAP', 0)}")
    print(f"Skipped-group count: {PIPELINE_STATS.get('skipped_groups', 0)}")

    unresolved_count = PIPELINE_STATS.get('unresolved_ambiguous', 0)
    print(f"Unresolved ambiguous subsets: {unresolved_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='CCD reduction pipeline for HERMES/HYDRA spectroscopy.'
    )
    parser.add_argument(
        '--infiles',
        required=True,
        help='Path to the infiles list in the night directory (e.g. /data/night/infiles)',
    )
    parser.add_argument(
        '--object',
        default=None,
        help='Optional SCIENCE object label filter for stacking/flat-field (normalized match).',
    )
    parser.add_argument(
        '--bias',
        action='store_true',
        help='Run bias stacking and correction (steps 3-4). Omit if data use only overscan.',
    )
    parser.add_argument(
        '--flat',
        default=None,
        metavar='FLATFILE',
        help='Path to master flat FITS file; enables step 10.',
    )
    parser.add_argument(
        '--dark',
        default=None,
        metavar='DARKFILE',
        help='Path to master dark FITS file for step-7 override.',
    )
    parser.add_argument(
        '--extra-columns',
        nargs='*',
        default=['EXPTIME', 'OBJECT', 'PLATE'],
        metavar='COL',
        help='Extra FITS header columns to include in the image table.',
    )
    parser.add_argument(
        '--start-step',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='First preprocessing step to execute (default: 1).',
    )
    parser.add_argument(
        '--end-step',
        type=int,
        default=10,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Last preprocessing step to execute (default: 10).',
    )
    parser.add_argument(
        '--resume-from-proc',
        action='store_true',
        help='Resume from existing proc products (requires --start-step >= 6).',
    )
    parser.add_argument(
        '--night',
        default=None,
        help='Restrict resume scan to this NIGHT value.',
    )
    parser.add_argument(
        '--shoe',
        default=None,
        help='Restrict resume scan to this SHOE value.',
    )
    parser.add_argument(
        '--plate',
        default=None,
        help='Restrict resume scan to this PLATE value.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        selected_steps = selected_steps_from_args(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    raw_dir, proc_dir, infiles_abs = resolve_processing_dirs(args.infiles)
    os.makedirs(proc_dir, exist_ok=True)

    print(f'Raw input directory: {raw_dir}')
    print(f'Processing directory: {proc_dir}')

    os.chdir(proc_dir)
    print(f'Current working directory: {os.getcwd()}')

    images_table = None
    combined_images = None
    stacked_images = None

    if args.resume_from_proc:
        if args.start_step < 6:
            raise SystemExit('ERROR: --resume-from-proc requires --start-step >= 6.')
        print(
            f'Resume mode: using proc products for steps {args.start_step}..{args.end_step}'
        )
        combined_images = build_resume_combined_images(
            proc_dir,
            start_step=args.start_step,
            night=args.night,
            shoe=args.shoe,
            plate=args.plate,
            object_name=args.object,
        )
        if len(combined_images) == 0:
            raise SystemExit(
                'ERROR: no compatible proc products found for resume mode. '
                'Try lower --start-step, remove restrictive filters, or run without --resume-from-proc.'
            )
    else:
        if args.start_step != 1:
            print(
                'WARNING: raw-mode preprocessing requires early prerequisites; '
                'restarting from step 1.'
            )
            selected_steps = list(range(1, args.end_step + 1))

        images_table, _, master_list = step1_load(infiles_abs, args.extra_columns, raw_dir, proc_dir)

        if 2 in selected_steps:
            images_table = step2_overscan_trim(images_table, master_list, raw_dir=raw_dir)

        if 3 in selected_steps or 4 in selected_steps:
            if args.bias:
                if 3 in selected_steps:
                    images_table = step3_bias_stack(images_table)
                if 4 in selected_steps:
                    images_table = step4_bias_correct(images_table)
            else:
                print('\nSteps 3-4 requested but --bias not set; skipping bias steps.')

        if any(step in selected_steps for step in (5, 6, 7, 8, 9, 10)):
            combined_images = step5_mosaic(images_table)
            step5b_update_ccdsec(combined_images)

    if combined_images is None:
        combined_images = Table(rows=[])

    if 6 in selected_steps:
        combined_images = step6_dark_crr_stack(combined_images)
    if 7 in selected_steps:
        combined_images, _ = step7_dark_subtract(
            combined_images,
            args.object,
            dark_override=args.dark,
        )
    if 8 in selected_steps:
        combined_images = step8_crr_science(combined_images)

    if 9 in selected_steps:
        stacked_images = step9_stack_science(combined_images, args.object)
    else:
        stacked_images = combined_images

    if 10 in selected_steps:
        if args.flat:
            stacked_images = step10_flatfield(stacked_images, args.flat, args.object)
        else:
            print('\nStep 10 requested but --flat not set; skipping flat-field correction.')

    final_table = stacked_images
    if final_table is None or len(final_table) == 0:
        final_table = combined_images if combined_images is not None else images_table

    _print_end_summary(final_table)
    _log_step('OK', 'Pipeline complete')


if __name__ == '__main__':
    main()
