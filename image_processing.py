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
import os
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
            values.append(str(val))
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


def _update_table_filenames(table, mask, filenames, suffix):
    """Update FILENAME and filename_input for selected rows."""
    table['FILENAME'][mask] = [f.replace('.fits', f'-{suffix}') for f in filenames]
    table['filename_input'][mask] = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]


def _write_iraf_lists(listpath, input_frames, output_frames, suffix):
    """Write the input and output @-list files expected by IRAF batch tasks."""
    pre, ext = os.path.splitext(os.path.relpath(listpath))
    outlist_path = f'{pre}-{suffix}{ext}'
    with open(listpath, 'w') as f:
        f.write('\n'.join(input_frames))
    with open(outlist_path, 'w') as f:
        f.write('\n'.join(output_frames))
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


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def do_median_crr(intable, listpath, mask, suffix='mcrr'):
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
    output_frames = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'crmedian: {inp} -> {out}')
        pyraf_utils.median_crr_single(inp, out)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
    return outtable, outlist_path


def do_flatfield_correction(intable, listpath, mask, flatpath, suffix='f'):
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
    output_frames = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'flat: {inp} -> {out}')
        pyraf_utils.run_ccdproc_flat_corr(inp, out, flatpath)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
    return outtable, outlist_path


def stack_dark_frames(comb_img_table, method='median'):
    """Stack dark frames per night and shoe; append DARK_MASTER rows."""
    if not _require_columns(comb_img_table, ['IMAGE_TYPE', 'NIGHT', 'SHOE'], 'stack_dark_frames'):
        return comb_img_table

    darkmask = comb_img_table['IMAGE_TYPE'] == 'DARK'
    groups = _grouped_row_indices(
        comb_img_table,
        darkmask,
        ['NIGHT', 'SHOE'],
        context='step6_dark_stack',
    )

    method_suffix = method[:2]
    new_rows = []
    for (night, shoe), indices in groups.items():
        frames = [str(comb_img_table['filename_input'][i]) + '[0]' for i in indices]
        if not frames:
            continue

        outpath = f'{night}-Dark_master-{shoe}{method_suffix}.fits'
        stack_listpath = f'dark_stack_{night}_{shoe}{method_suffix}.list'
        with open(stack_listpath, 'w') as f:
            f.write('\n'.join(frames))

        pyraf_utils.stack_science_images(stack_listpath, outpath, mode=method)
        _sync_output_header_metadata(
            outpath,
            {
                'OBJECT': str(comb_img_table['OBJECT'][indices[0]]),
                'EXPTYPE': 'Dark_master',
                'IMAGE_TYPE': 'DARK_MASTER',
                'NIGHT': str(night),
                'SHOE': str(shoe),
                'STACKED': True,
                'STACKTYPE': str(method).lower(),
                'PROCSTEP': 'step6_dark_stack',
            },
            context='stack_dark_frames',
            overwrite_conflicts=True,
        )
        print(f'Night {night}, shoe {shoe} done. Saved in {outpath}')

        template = comb_img_table[indices[0]]
        row = {c: template[c] for c in comb_img_table.colnames}
        row['FILENAME'] = outpath.replace('.fits', '')
        row['filename_input'] = outpath
        row['EXPTYPE'] = 'Dark_master'
        row['EXPTYPE_NORM'] = normalize_header_value('Dark_master')
        row['IMAGE_TYPE'] = 'DARK_MASTER'
        row['CLASS_REASON'] = 'step6_dark_stack'
        row['STACKED'] = True
        row['STACKTYPE'] = str(method).lower()
        new_rows.append(row)

    if not new_rows:
        print('WARNING [step6_dark_stack]: no dark groups were stackable.')
        return comb_img_table

    result = vstack([comb_img_table, Table(new_rows)])
    _ensure_string_columns(result, ['FILENAME', 'filename_input'])
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

    pyraf_utils.run_ccdproc_subtract_dark(inlist, outlist, dark_image=master_dark_path)

    outtable = intable.copy()
    filenames = [str(p) + '.fits' for p in subset['FILENAME']]
    _update_table_filenames(outtable, mask, filenames, suffix)
    _ensure_string_columns(outtable, ['FILENAME', 'filename_input'])
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
    local_groups = _grouped_row_indices(
        sci_frames,
        np.ones(len(sci_frames), dtype=bool),
        ['NIGHT', 'SHOE'],
        context='step9_stack',
    )

    new_rows = []
    path_to_result = []

    for (night, shoe), local_idx in local_groups.items():
        subset = sci_frames[local_idx]
        if len(subset) == 0:
            continue

        paths = [str(p) for p in subset['filename_input']]
        name = str(subset['OBJECT'][0]).replace(' ', '').replace('&', '')

        stack_listpath = f'{name}_{night}_{shoe}-d_in_onlystack.list'
        with open(stack_listpath, 'w') as f:
            f.write('\n'.join(p.replace('.fits', '.fits[0]') for p in paths))

        out_filepath = f'{name}_{night}_{shoe}-d_{mode[0]}stack.fits'
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
            'STACKED': True,
            'STACKTYPE': str(mode).lower(),
            'STACKMOD': str(mode),
            'PROCSTEP': 'step9_stack',
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
        row['STACKED'] = True
        row['STACKTYPE'] = str(mode).lower()
        new_rows.append(row)
        path_to_result.append(out_filepath)

    if not new_rows:
        return comb_img_table, []

    result = vstack([comb_img_table, Table(new_rows)])
    _ensure_string_columns(result, ['FILENAME', 'filename_input'])
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
        os.path.splitext(os.path.basename(name))[0] + f'-{suffix}.fits'
        for name in entries
    ]

    pre, ext = os.path.splitext(master_list_path)
    inlist_raw = f'{pre}-raw{ext}'
    outlist = f'{pre}-{suffix}{ext}'

    with open(inlist_raw, 'w') as fh:
        fh.write('\n'.join(raw_inputs))
    with open(outlist, 'w') as fh:
        fh.write('\n'.join(outputs))

    return inlist_raw, outlist


def step2_overscan_trim(images_table, master_list, raw_dir):
    _log_step(2, 'Overscan fit & trim')
    suffix = 'ot'

    for inlist in master_list:
        raw_inlist, outlist = _make_step2_lists(inlist, raw_dir=raw_dir, suffix=suffix)
        pyraf_utils.run_ccdproc_ovefit_trim(raw_inlist, outlist)

    images_table['filename_input'] = images_table['FILENAME'] + f'-{suffix}.fits'
    images_table['FILENAME'] = images_table['FILENAME'] + f'-{suffix}'
    _ensure_string_columns(images_table, ['FILENAME', 'filename_input'])
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
        images_table['FILENAME'][mask] = images_table['FILENAME'][mask] + '-' + suffix
        images_table['filename_input'][mask] = images_table['FILENAME'][mask] + '.fits'
        print(f'\t{shoe}{opamp} bias correction done')

    _ensure_string_columns(images_table, ['FILENAME', 'filename_input'])
    return images_table


def step5_mosaic(images_table):
    _log_step(5, 'Mosaic assembly')
    pyraf_utils.load_images()

    required = ['IMAGE_TYPE', 'NIGHT', 'LC-TIME', 'SHOE', 'OPAMP']
    if not _require_columns(images_table, required, 'step5_mosaic'):
        return Table(rows=[])

    source_mask = ~np.isin(images_table['IMAGE_TYPE'], ['BIAS', 'MASTER_BIAS'])
    groups = _grouped_row_indices(
        images_table,
        source_mask,
        ['NIGHT', 'LC-TIME', 'SHOE'],
        context='step5_mosaic',
    )

    mosaics = []
    for (night, lctime, shoe), indices in groups.items():
        subset = images_table[indices].copy()
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
            'PROCSTEP': 'step5_mosaic',
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
        row['IMAGE_TYPE'] = image_type
        row['CLASS_REASON'] = 'step5_mosaic'
        row['EXPTYPE_NORM'] = normalize_header_value(row['EXPTYPE'])
        row['OBJECT_NORM'] = normalize_header_value(row['OBJECT'])
        mosaics.append(row)

    if not mosaics:
        print('WARNING [step5_mosaic]: no complete mosaic groups were generated.')
        return Table(rows=[])

    combined_images = Table(mosaics)
    _ensure_string_columns(combined_images, ['FILENAME', 'filename_input'])
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

    combined_images, _ = do_median_crr(combined_images, 'darklist_temp.list', mask=dmask)
    print('CRR on darks done')
    combined_images = stack_dark_frames(combined_images, method='median')
    return combined_images


def step7_dark_subtract(combined_images, object_name=None):
    _log_step(7, 'Dark subtraction')
    if not _require_columns(combined_images, ['IMAGE_TYPE', 'SHOE'], 'step7_dark_subtract'):
        return combined_images, np.ones(len(combined_images), dtype=bool)

    non_dark_mask = ~np.isin(combined_images['IMAGE_TYPE'], ['DARK', 'DARK_MASTER'])
    target_mask = non_dark_mask & (combined_images['IMAGE_TYPE'] == 'SCIENCE')

    if object_name:
        object_norm = normalize_header_value(object_name)
        if 'OBJECT_NORM' in combined_images.colnames:
            target_mask = target_mask & (combined_images['OBJECT_NORM'] == object_norm)
        else:
            print('WARNING [step7_dark_subtract]: OBJECT_NORM missing; cannot apply --object filter.')

    if not np.any(target_mask):
        print('No SCIENCE frames selected for dark subtraction; continuing.')
        return combined_images, non_dark_mask

    def _master_dark_path(shoe):
        m_mask = (
            (combined_images['IMAGE_TYPE'] == 'DARK_MASTER') &
            (combined_images['SHOE'] == shoe)
        )
        candidates = combined_images[m_mask]['filename_input']
        if len(candidates) != 1:
            _warn_skip_group(
                'step7_dark_subtract',
                f'expected one DARK_MASTER for shoe={shoe}, got {len(candidates)}',
            )
            return None
        return str(candidates[0])

    shoes = sorted({str(v) for v in combined_images[target_mask]['SHOE']})
    for shoe in shoes:
        master_dark = _master_dark_path(shoe)
        if master_dark is None:
            continue
        shoe_mask = target_mask & (combined_images['SHOE'] == shoe)
        combined_images = subtract_dark_mask(
            combined_images,
            shoe_mask,
            master_dark,
            label=f'science_{shoe}',
        )

    return combined_images, non_dark_mask


def step8_crr_science(combined_images):
    _log_step(8, 'Cosmic-ray removal on processable non-dark frames')
    if not _require_columns(combined_images, ['IMAGE_TYPE'], 'step8_crr_science'):
        return combined_images

    crr_mask = np.isin(combined_images['IMAGE_TYPE'], sorted(PROCESSABLE_CRR_IMAGE_TYPES))
    if not np.any(crr_mask):
        print('No SCIENCE/TWILIGHT/QUARTZ/LAMP frames to CR-clean; continuing.')
        return combined_images

    combined_images, _ = do_median_crr(combined_images, 'templist.list', mask=crr_mask)
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
    if object_name and 'OBJECT_NORM' in combined_images.colnames:
        sci_mask = sci_mask & (combined_images['OBJECT_NORM'] == normalize_header_value(object_name))

    if not np.any(sci_mask):
        print('No SCIENCE frames selected for flat-field correction; skipping step 10.')
        return combined_images

    combined_images, _ = do_flatfield_correction(
        combined_images,
        'flatlist_temp.list',
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
        '--extra-columns',
        nargs='*',
        default=['EXPTIME', 'OBJECT', 'PLATE'],
        metavar='COL',
        help='Extra FITS header columns to include in the image table.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir, proc_dir, infiles_abs = resolve_processing_dirs(args.infiles)
    os.makedirs(proc_dir, exist_ok=True)

    print(f'Raw input directory: {raw_dir}')
    print(f'Processing directory: {proc_dir}')

    os.chdir(proc_dir)
    print(f'Current working directory: {os.getcwd()}')

    images_table, _, master_list = step1_load(infiles_abs, args.extra_columns, raw_dir, proc_dir)
    images_table = step2_overscan_trim(images_table, master_list, raw_dir=raw_dir)

    if args.bias:
        images_table = step3_bias_stack(images_table)
        images_table = step4_bias_correct(images_table)
    else:
        print('\nSteps 3-4 (bias) skipped -- pass --bias to enable.')

    combined_images = step5_mosaic(images_table)
    step5b_update_ccdsec(combined_images)
    combined_images = step6_dark_crr_stack(combined_images)
    combined_images, _ = step7_dark_subtract(combined_images, args.object)
    combined_images = step8_crr_science(combined_images)
    stacked_images = step9_stack_science(combined_images, args.object)

    if args.flat:
        stacked_images = step10_flatfield(stacked_images, args.flat, args.object)
    else:
        print('\nStep 10 (flat-field) skipped -- pass --flat <file> to enable.')

    _print_end_summary(stacked_images)
    _log_step('OK', 'Pipeline complete')


if __name__ == '__main__':
    main()
