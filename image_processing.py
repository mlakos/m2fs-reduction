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
    3.  Bias stacking         [skipped by default – pass --bias to enable]
    4.  Bias correction       [skipped by default – pass --bias to enable]
    5.  Mosaic assembly       (4 OPAMP chips -> single 2k x 2k frame)
    5b. CCDSEC header update
    6.  Dark CRR + stacking
    7.  Dark subtraction on science frames
    8.  Cosmic-ray removal on all non-dark mosaic frames
    9.  Science frame stacking
   10.  Flat-field correction  [pass --flat <flatfile> to enable]
"""

import argparse
import fnmatch
import os
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from pyraf import iraf

from file_handler import infiles_to_lists, generate_output_list, table_to_list, split_path
import pyraf_utils


# ---------------------------------------------------------------------------
# Object names used during stacking (Step 9).
# Edit these constants if your target / calibration labels differ between runs.
# ---------------------------------------------------------------------------
SCIENCE_OBJECT  = 'JSimon H3'
TWILIGHT_OBJECT = 'Config 14 Twilight'
QUARTZ_OBJECT   = 'JSimon H3 Quartz'
LAMP_OBJECT     = 'JSimon H3 ThAr & ThNe'


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _log_step(number, title):
    """Print a consistent step banner with a timestamp."""
    print(f'\n{"="*60}')
    print(f'STEP {number:<3} {title}  [{datetime.now()}]')
    print('='*60)


def _update_table_filenames(table, mask, filenames, suffix):
    """Update FILENAME and filename_input for the rows selected by *mask*.

    Parameters
    ----------
    table     : astropy Table, modified in place
    mask      : boolean array selecting the rows to update
    filenames : list of base .fits paths for the masked rows
    suffix    : string appended before .fits in the output names
    """
    table['FILENAME'][mask]       = [f.replace('.fits', f'-{suffix}')      for f in filenames]
    table['filename_input'][mask] = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]


def _write_iraf_lists(listpath, input_frames, output_frames, suffix):
    """Write the input and output @-list files expected by IRAF batch tasks.

    Returns the output list path.
    """
    pre, ext = os.path.splitext(os.path.relpath(listpath))
    outlist_path = f'{pre}-{suffix}{ext}'
    with open(listpath, 'w') as f:
        f.write('\n'.join(input_frames))
    with open(outlist_path, 'w') as f:
        f.write('\n'.join(output_frames))
    return outlist_path


def _sync_output_header_metadata(filepath, metadata, context='', overwrite_conflicts=True):
    """Ensure key metadata cards exist on an output FITS file.

    A short pre-check is always printed so it is explicit whether keys already
    existed before the write operation.
    """
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
                    print(
                        f"[meta-write] {filepath}: overwrite {key}={current!r} -> {value!r}"
                    )
                    hdr[key] = value
                else:
                    print(
                        f"[meta-skip] {filepath}: keep existing {key}={current!r}, expected {value!r}"
                    )


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def do_median_crr(intable, listpath, mask, suffix='mcrr'):
    """Run IRAF crmedian on the rows selected by *mask*.

    Returns (updated_table, outlist_path).
    """
    pyraf_utils.load_crutil()

    if mask is None:
        mask = np.ones(len(intable), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    filenames     = [str(p) + '.fits' for p in intable[mask]['FILENAME']]
    input_frames  = [f.replace('.fits', '.fits[0]') for f in filenames]
    output_frames = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'crmedian: {inp} -> {out}')
        pyraf_utils.median_crr_single(inp, out)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix)
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

    filenames     = [str(p) + '.fits' for p in intable[mask]['FILENAME']]
    input_frames  = [f.replace('.fits', '.fits[0]') for f in filenames]
    output_frames = [f.replace('.fits', f'-{suffix}.fits') for f in filenames]

    outlist_path = _write_iraf_lists(listpath, input_frames, output_frames, suffix)

    iraf.cd(os.getcwd())
    for inp, out in zip(input_frames, output_frames):
        print(f'flat: {inp} -> {out}')
        pyraf_utils.run_ccdproc_flat_corr(inp, out, flatpath)
    print(f'done\n{listpath} -> {outlist_path}')

    outtable = intable.copy()
    _update_table_filenames(outtable, mask, filenames, suffix)
    return outtable, outlist_path


def stack_dark_frames(comb_img_table, method='median'):
    """Stack dark frames per night and detector; append Dark_master rows.

    The *exposures_dic* lookup from the notebook has been removed: frames are
    read directly from the table rows already filtered to the current night and
    shoe, which is equivalent and avoids the intermediate dictionary entirely.

    Returns the updated table.
    """
    darkmask      = comb_img_table['EXPTYPE'] == 'Dark'
    method_suffix = method[:2]
    new_rows      = []

    for night in np.unique(comb_img_table[darkmask]['NIGHT']):
        for shoe in np.unique(comb_img_table[darkmask]['SHOE']):
            nmask  = darkmask & (comb_img_table['NIGHT'] == night) & (comb_img_table['SHOE'] == shoe)
            frames = [str(f) + '[0]' for f in comb_img_table[nmask]['filename_input']]
            outpath = f'{night}-Dark_master-{shoe}{method_suffix}.fits'

            stack_listpath = f'dark_stack_{night}_{shoe}{method_suffix}.list'
            with open(stack_listpath, 'w') as f:
                f.write('\n'.join(frames))

            pyraf_utils.stack_science_images(stack_listpath, outpath, mode=method)
            _sync_output_header_metadata(
                outpath,
                {
                    'OBJECT': str(comb_img_table[nmask]['OBJECT'][0]),
                    'EXPTYPE': 'Dark_master',
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

            template = comb_img_table[nmask][0]
            row = {c: template[c] for c in comb_img_table.colnames}
            row['FILENAME']       = outpath.replace('.fits', '')
            row['filename_input'] = outpath
            row['EXPTYPE']        = 'Dark_master'
            row['STACKED']        = True
            row['STACKTYPE']      = str(method).lower()
            new_rows.append(row)

    result = vstack([comb_img_table, Table(new_rows)])
    for col in ['FILENAME', 'filename_input']:
        result[col] = result[col].astype('U64')
    return result


def subtract_dark(intable, object_name, master_dark_path, shoe=None, extra_mask=None):
    """Subtract a master dark from object frames via IRAF ccdproc (darkcor).

    Writes input/output list files, calls run_ccdproc_subtract_dark, then
    updates FILENAME and filename_input for the affected rows.
    Returns a new table with updated filenames for the affected rows.
    """
    mask = intable['OBJECT'] == object_name
    if shoe is not None:
        mask = mask & (intable['SHOE'] == shoe)
    if extra_mask is not None:
        mask = mask & extra_mask

    suffix   = 'D'
    subset   = intable[mask]
    listpath = f'darksub_{object_name.replace(" ", "")}_{shoe}.list'
    inlist, outlist = table_to_list(subset, listpath, suffix=suffix)

    pyraf_utils.run_ccdproc_subtract_dark(inlist, outlist, dark_image=master_dark_path)

    outtable = intable.copy()
    filenames = [str(p) + '.fits' for p in subset['FILENAME']]
    _update_table_filenames(outtable, mask, filenames, suffix)
    return outtable


def science_frame_stacking(comb_img_table, sci_mask, mode, scale='none'):
    """Stack science frames per night and detector, convert output to electrons.

    Returns (updated_table, list_of_output_paths).
    """
    assert len(sci_mask) == len(comb_img_table), 'science mask length mismatch'
    sci_frames     = comb_img_table[sci_mask]
    new_rows       = []
    path_to_result = []

    for night in np.unique(sci_frames['NIGHT']):
        for shoe in np.unique(sci_frames['SHOE']):
            local_mask = (sci_frames['NIGHT'] == night) & (sci_frames['SHOE'] == shoe)
            if not np.any(local_mask):
                continue
            subset = sci_frames[local_mask]
            paths  = subset['filename_input'].tolist()
            name   = subset['OBJECT'][0].replace(' ', '').replace('&', '')

            stack_listpath = f'{name}_{night}_{shoe}-d_in_onlystack.list'
            with open(stack_listpath, 'w') as f:
                f.write('\n'.join(p.replace('.fits', '.fits[0]') for p in paths))

            out_filepath = f'{name}_{night}_{shoe}-d_{mode[0]}stack.fits'
            pyraf_utils.stack_science_images(stack_listpath, out_filepath, mode=mode, scale=scale)

            gain = fits.getheader(paths[0]).get('EGAIN', 1.0)
            with fits.open(out_filepath, mode='update') as hdul:
                hdul[0].data            = hdul[0].data * gain
                hdul[0].header['BUNIT'] = 'electron'

            stack_meta = {
                'OBJECT': str(subset['OBJECT'][0]),
                'EXPTYPE': str(subset['EXPTYPE'][0]),
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
            row['FILENAME']       = out_filepath.replace('.fits', '')
            row['filename_input'] = out_filepath
            row['EXPTYPE']        = str(subset['EXPTYPE'][0]) + '_stack'
            row['STACKED']        = True
            row['STACKTYPE']      = str(mode).lower()
            new_rows.append(row)
            path_to_result.append(out_filepath)

    result = vstack([comb_img_table, Table(new_rows)])
    for col in ['FILENAME', 'filename_input']:
        result[col] = result[col].astype('U128')
    return result, path_to_result


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_load(list_path, extra_columns):
    _log_step(1, 'Load & Initialise')
    pyraf_utils.load_ccdred()
    dir_path, _, _ = split_path(list_path)
    print(f'Data directory: {dir_path}')
    images_table, list_of_exposures, master_list = infiles_to_lists(
        list_path, extra_columns=extra_columns)
    return images_table, list_of_exposures, master_list


def step2_overscan_trim(images_table, master_list):
    _log_step(2, 'Overscan fit & trim')
    suffix     = 'ot'
    outlist_ot = generate_output_list(master_list, suffix=suffix)
    for inlist, outlist in zip(master_list, outlist_ot):
        pyraf_utils.run_ccdproc_ovefit_trim(inlist, outlist)
    images_table['filename_input'] = images_table['FILENAME'] + f'-{suffix}.fits'
    images_table['FILENAME']       = images_table['FILENAME'] + f'-{suffix}'
    return images_table


def step3_bias_stack(images_table):
    _log_step(3, 'Bias stacking')
    bmask     = images_table['EXPTYPE'] == 'Bias'
    col_names = images_table.colnames
    new_rows  = []

    for shoe in np.unique(images_table['SHOE']):
        for opamp in np.unique(images_table['OPAMP']):
            mask = bmask & (images_table['SHOE'] == shoe) & (images_table['OPAMP'] == opamp)
            inlist_path, _ = table_to_list(
                images_table[mask], f'Bias_stack_{shoe}{opamp}.list', 'TRASH')
            outpath = f'Master_bias_{shoe}{opamp}.fits'
            pyraf_utils.run_zerocombine_masterbias(inlist_path, outpath)
            hdr = fits.getheader(outpath)
            row = {c: hdr.get(c, '') for c in col_names}
            row.update({'EXPTYPE': 'Master_bias', 'FILENAME': outpath, 'filename_input': outpath})
            new_rows.append(row)
            print(f'\t{outpath} done')

    return vstack([images_table, Table(new_rows)]) if new_rows else images_table


def step4_bias_correct(images_table):
    _log_step(4, 'Bias correction')
    suffix   = 'B'
    non_bias = ~np.isin(images_table['EXPTYPE'], ['Bias', 'Master_bias'])

    for shoe in np.unique(images_table['SHOE']):
        for opamp in np.unique(images_table['OPAMP']):
            mask = non_bias & (images_table['SHOE'] == shoe) & (images_table['OPAMP'] == opamp)
            inlist_path, outlist_path = table_to_list(
                images_table[mask], f'Bias_correction_{shoe}{opamp}.list', suffix=suffix)

            mb_mask  = ((images_table['EXPTYPE'] == 'Master_bias') &
                        (images_table['SHOE']    == shoe) &
                        (images_table['OPAMP']   == opamp))
            mb_files = images_table[mb_mask]['filename_input']
            if len(mb_files) != 1:
                raise RuntimeError(
                    f'Expected one master bias for {shoe}{opamp}, got {len(mb_files)}')

            pyraf_utils.run_ccdproc_bias_corr(
                inlist_path, outlist_path, zero_image=str(mb_files[0]))
            images_table['FILENAME'][mask]       = images_table['FILENAME'][mask] + '-' + suffix
            images_table['filename_input'][mask] = images_table['FILENAME'][mask] + '.fits'
            print(f'\t{shoe}{opamp} bias correction done')

    return images_table


def step5_mosaic(images_table):
    _log_step(5, 'Mosaic assembly')
    pyraf_utils.load_images()
    non_bias = ~np.isin(images_table['EXPTYPE'], ['Bias', 'Master_bias'])
    mosaics  = []

    for t in np.unique(images_table['LC-TIME']):
        for shoe in np.unique(images_table[images_table['LC-TIME'] == t]['SHOE']):
            mask = (images_table['LC-TIME'] == t) & (images_table['SHOE'] == shoe) & non_bias
            if not np.any(mask):
                continue

            base_name = images_table[mask]['FILENAME'][0]
            for chip in ['c1', 'c2', 'c3', 'c4']:
                base_name = base_name.replace(chip, '')

            exptype      = images_table[mask]['EXPTYPE'][0]
            out_filename = f'{exptype}-{base_name}-full.fits'
            pyraf_utils.assemble_mosaic(images_table[mask], out_filename)

            mosaic_meta = {
                'OBJECT': str(images_table[mask]['OBJECT'][0]),
                'EXPTYPE': str(exptype),
                'NIGHT': str(images_table[mask]['NIGHT'][0]),
                'SHOE': str(images_table[mask]['SHOE'][0]),
                'PROCSTEP': 'step5_mosaic',
            }
            _sync_output_header_metadata(
                out_filename,
                mosaic_meta,
                context='step5_mosaic',
                overwrite_conflicts=True,
            )

            row = {c: images_table[mask][0][c] for c in images_table.colnames}
            row['FILENAME']       = f'{exptype}-{base_name}-full'
            row['filename_input'] = out_filename
            row['OPAMP']          = 0
            mosaics.append(row)

    combined_images = Table(mosaics)
    for col in ['FILENAME', 'filename_input']:
        combined_images[col] = combined_images[col].astype('U64')
    return combined_images


def step5b_update_ccdsec(combined_images):
    _log_step('5b', 'CCDSEC header update')
    pyraf_utils.load_imutil()
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
    if 'Dark_master' in np.unique(combined_images['EXPTYPE']):
        print('Dark masters already present, skipping.')
        return combined_images

    dmask = combined_images['EXPTYPE'] == 'Dark'
    combined_images, _ = do_median_crr(combined_images, 'darklist_temp.list', mask=dmask)
    print('CRR on darks done')
    combined_images = stack_dark_frames(combined_images, method='median')
    return combined_images


def step7_dark_subtract(combined_images, object_name):
    _log_step(7, 'Dark subtraction')
    # Exclude the dark calibration frames themselves from the subtraction mask
    dark_object = fnmatch.filter(np.unique(combined_images['OBJECT']).tolist(), '*Dark*')[0]
    typemask    = combined_images['OBJECT'] != dark_object

    def _master_dark_path(shoe):
        return str(combined_images[
            (combined_images['EXPTYPE'] == 'Dark_master') &
            (combined_images['SHOE']    == shoe)
        ]['filename_input'][0])

    combined_images = subtract_dark(
        combined_images, object_name, _master_dark_path('B'), shoe='B', extra_mask=typemask)
    combined_images = subtract_dark(
        combined_images, object_name, _master_dark_path('R'), shoe='R', extra_mask=typemask)
    return combined_images, typemask


def step8_crr_science(combined_images, typemask):
    _log_step(8, 'Cosmic-ray removal on science frames')
    combined_images, _ = do_median_crr(combined_images, 'templist.list', mask=typemask)
    for col in ['FILENAME', 'filename_input']:
        combined_images[col] = combined_images[col].astype('U128')
    return combined_images


def step9_stack_science(combined_images, object_name):
    _log_step(9, 'Science frame stacking')
    stacked = combined_images.copy()

    # Science target: sum-combine to preserve total photon counts
    stacked, _ = science_frame_stacking(stacked, stacked['OBJECT'] == object_name,     mode='sum')
    # Calibration frames: median-combine with median scaling
    stacked, _ = science_frame_stacking(stacked, stacked['OBJECT'] == TWILIGHT_OBJECT, mode='median', scale='median')
    stacked, _ = science_frame_stacking(stacked, stacked['OBJECT'] == QUARTZ_OBJECT,   mode='median', scale='median')
    stacked, _ = science_frame_stacking(stacked, stacked['OBJECT'] == LAMP_OBJECT,     mode='median', scale='median')

    return stacked


def step10_flatfield(combined_images, flatpath):
    _log_step(10, 'Flat-field correction')
    sci_mask = combined_images['EXPTYPE'] == 'Object'
    combined_images, _ = do_flatfield_correction(
        combined_images, 'flatlist_temp.list', mask=sci_mask, flatpath=flatpath)
    return combined_images


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='CCD reduction pipeline for HERMES/HYDRA spectroscopy.')
    parser.add_argument('--infiles', required=True,
                        help='Path to the infiles list (e.g. /data/infiles)')
    parser.add_argument('--object', default=SCIENCE_OBJECT,
                        help=f'Science target OBJECT keyword (default: "{SCIENCE_OBJECT}")')
    parser.add_argument('--bias', action='store_true',
                        help='Run bias stacking and correction (Steps 3-4). '
                             'Omit if data use only overscan.')
    parser.add_argument('--flat', default=None, metavar='FLATFILE',
                        help='Path to master flat FITS file; enables Step 10.')
    parser.add_argument('--extra-columns', nargs='*',
                        default=['EXPTIME', 'OBJECT', 'PLATE'], metavar='COL',
                        help='Extra FITS header columns to include in the image table.')
    return parser.parse_args()


def main():
    args = parse_args()

    images_table, _, master_list = step1_load(args.infiles, args.extra_columns)
    images_table = step2_overscan_trim(images_table, master_list)

    if args.bias:
        images_table = step3_bias_stack(images_table)
        images_table = step4_bias_correct(images_table)
    else:
        print('\nSteps 3-4 (bias) skipped -- pass --bias to enable.')

    combined_images = step5_mosaic(images_table)
    step5b_update_ccdsec(combined_images)
    combined_images = step6_dark_crr_stack(combined_images)
    combined_images, typemask = step7_dark_subtract(combined_images, args.object)
    combined_images = step8_crr_science(combined_images, typemask)
    stacked_images  = step9_stack_science(combined_images, args.object)

    if args.flat:
        stacked_images = step10_flatfield(stacked_images, args.flat)
    else:
        print('\nStep 10 (flat-field) skipped -- pass --flat <file> to enable.')

    _log_step('OK', 'Pipeline complete')


if __name__ == '__main__':
    main()