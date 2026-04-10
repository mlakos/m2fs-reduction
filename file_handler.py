from astropy.table import Table
from astropy.io import fits

from datetime import datetime, timedelta
import json

import numpy as np
import os
import re
import sys


CANONICAL_IMAGE_TYPES = {
    'BIAS',
    'MASTER_BIAS',
    'DARK',
    'DARK_MASTER',
    'SCIENCE',
    'TWILIGHT',
    'QUARTZ',
    'LAMP',
    'FIBERMAP',
    'UNKNOWN',
}

USER_CLASS_TO_IMAGE_TYPE = {
    'object': 'SCIENCE',
    'dark': 'DARK',
    'flat': 'QUARTZ',
    'twilight': 'TWILIGHT',
    'lamp': 'LAMP',
    'fibermap': 'FIBERMAP',
}

PROMPT_IMAGE_TYPE_ROLES = [
    ('dark', 'DARK', 'dark'),
    ('science/object', 'SCIENCE', 'science/object'),
    ('flat', 'QUARTZ', 'flat'),
    ('lamp/thar/arc', 'LAMP', 'lamp'),
    ('twilight', 'TWILIGHT', 'twilight'),
    ('fibermap', 'FIBERMAP', 'fibermap'),
]


def normalize_header_value(value):
    """Normalize a FITS header value for deterministic matching."""
    text = '' if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def _normalized_signature(exptype_norm, object_norm):
    return f'exptype={exptype_norm}|object={object_norm}'


def _resolve_override_path(override_path=None):
    if override_path is not None:
        return override_path
    return os.path.join(os.path.dirname(__file__), 'image_type_overrides.json')


def _normalize_override_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in CANONICAL_IMAGE_TYPES:
        return upper
    lower = text.lower()
    if lower in USER_CLASS_TO_IMAGE_TYPE:
        return USER_CLASS_TO_IMAGE_TYPE[lower]
    return None


def load_image_type_overrides(override_path=None):
    """Load image-type overrides keyed by normalized EXPTYPE/OBJECT signature."""
    path = _resolve_override_path(override_path)
    if not os.path.exists(path):
        return {}, path

    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f'WARNING: failed to read overrides from {path}: {exc}')
        return {}, path

    if isinstance(payload, dict) and isinstance(payload.get('mappings'), dict):
        raw_map = payload['mappings']
    elif isinstance(payload, dict):
        raw_map = payload
    else:
        raw_map = {}

    cleaned = {}
    for key, value in raw_map.items():
        mapped = _normalize_override_value(value)
        if mapped:
            cleaned[str(key)] = mapped
    return cleaned, path


def save_image_type_overrides(overrides, override_path=None):
    """Persist image-type overrides in a small JSON file."""
    path = _resolve_override_path(override_path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'version': 1,
        'mappings': dict(sorted(overrides.items())),
    }
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return path


def classify_image_type(exptype_norm, object_norm, overrides=None):
    """Classify one exposure into a canonical IMAGE_TYPE and CLASS_REASON."""
    overrides = overrides or {}
    signature = _normalized_signature(exptype_norm, object_norm)

    bias_aliases = {'bias', 'zero'}
    master_bias_aliases = {'master bias', 'master_bias', 'bias master', 'bias_master'}
    dark_aliases = {'dark'}
    dark_master_aliases = {'dark master', 'dark_master', 'master dark', 'master_dark'}

    # 1) EXPTYPE-driven classes first.
    if exptype_norm in bias_aliases:
        return 'BIAS', 'exptype_bias_zero'
    if exptype_norm in master_bias_aliases:
        return 'MASTER_BIAS', 'exptype_master_bias'
    if exptype_norm in dark_aliases:
        return 'DARK', 'exptype_dark'
    if exptype_norm in dark_master_aliases:
        return 'DARK_MASTER', 'exptype_dark_master'

    has_twilight = ('twilight' in object_norm) or ('dawn sky' in object_norm)
    has_quartz = (
        ('quartz' in object_norm) or
        ('flat' in object_norm) or
        ('domeflat' in object_norm) or
        ('dome flat' in object_norm)
    )
    has_lamp = (
        ('thar' in object_norm) or
        ('tharne' in object_norm) or
        ('thne' in object_norm) or
        ('lamp' in object_norm) or
        ('arc' in object_norm)
    )
    has_fibermap = (
        ('fibermap' in object_norm) or
        ('fiber map' in object_norm) or
        ('fibre map' in object_norm) or
        ('fibremap' in object_norm)
    )

    calibration_hits = int(has_twilight) + int(has_quartz) + int(has_lamp) + int(has_fibermap)
    if calibration_hits > 1 and signature in overrides:
        return overrides[signature], 'override_ambiguous_object_tokens'
    if calibration_hits > 1:
        return 'UNKNOWN', 'ambiguous_object_tokens'

    # 2) OBJECT-driven subclasses.
    if has_quartz:
        return 'QUARTZ', 'object_quartz_flat_like'
    if has_lamp:
        return 'LAMP', 'object_lamp_like'
    if has_twilight:
        return 'TWILIGHT', 'object_twilight_like'
    if has_fibermap:
        return 'FIBERMAP', 'object_fibermap_like'

    science_aliases = {'object', 'science', 'target', 'star'}
    if object_norm in science_aliases:
        return 'SCIENCE', 'object_science_alias'

    # 3) EXPTYPE fallback.
    if exptype_norm == 'lamp':
        return 'LAMP', 'exptype_fallback_lamp'
    if exptype_norm == 'object':
        bookkeeping_tokens = (
            'bias', 'dark', 'flat', 'quartz', 'lamp', 'thar',
            'arc', 'twilight', 'fiber', 'fibre', 'calib', 'config'
        )
        if not any(token in object_norm for token in bookkeeping_tokens):
            return 'SCIENCE', 'exptype_fallback_object'

    # Apply persisted user override before UNKNOWN fallback.
    if signature in overrides:
        return overrides[signature], 'override_unresolved_signature'

    # 4) Unknown.
    return 'UNKNOWN', 'unknown_fallback'


def _iter_ambiguous_groups(table, overrides, min_count=2):
    signatures = {}
    for idx, row in enumerate(table):
        signature = _normalized_signature(row['EXPTYPE_NORM'], row['OBJECT_NORM'])
        signatures.setdefault(signature, []).append(idx)

    ambiguous = []
    for signature, indices in signatures.items():
        if len(indices) < min_count:
            continue
        if signature in overrides:
            continue

        reasons = {str(table[i]['CLASS_REASON']) for i in indices}
        image_types = {str(table[i]['IMAGE_TYPE']) for i in indices}
        if image_types == {'UNKNOWN'} or 'ambiguous_object_tokens' in reasons:
            sample = table[indices[0]]
            ambiguous.append({
                'signature': signature,
                'count': len(indices),
                'indices': indices,
                'exptype_norm': str(sample['EXPTYPE_NORM']),
                'object_norm': str(sample['OBJECT_NORM']),
            })
    return sorted(ambiguous, key=lambda item: (-item['count'], item['signature']))


def _iter_unresolved_signatures(table):
    signatures = []
    seen = set()
    for row in table:
        if str(row['IMAGE_TYPE']) != 'UNKNOWN':
            continue
        signature = _normalized_signature(row['EXPTYPE_NORM'], row['OBJECT_NORM'])
        if signature in seen:
            continue
        seen.add(signature)
        signatures.append(signature)
    return signatures


def _is_unusable_label(text):
    value = normalize_header_value(text)
    return value in {'', 'unknown', 'none', 'n/a', 'na'}


def _preferred_unresolved_label(row):
    object_label = str(row.get('OBJECT', '')).strip()
    if not _is_unusable_label(object_label):
        return object_label
    exptype_label = str(row.get('EXPTYPE', '')).strip()
    if not _is_unusable_label(exptype_label):
        return exptype_label
    return str(row.get('OBJECT_NORM', '')).strip() or str(row.get('EXPTYPE_NORM', '')).strip() or 'unknown'


def _build_unresolved_label_options(table):
    options = []
    by_label = {}

    for row in table:
        if str(row['IMAGE_TYPE']) != 'UNKNOWN':
            continue

        signature = _normalized_signature(row['EXPTYPE_NORM'], row['OBJECT_NORM'])
        label = _preferred_unresolved_label(row)
        payload = by_label.get(label)
        if payload is None:
            payload = {'label': label, 'signatures': []}
            by_label[label] = payload
            options.append(payload)

        if signature not in payload['signatures']:
            payload['signatures'].append(signature)

    return options


def _prompt_numbered_label_selection(question, options):
    print(f"\nWhich of the following is the {question} image label:")
    for idx, entry in enumerate(options, start=1):
        print(f"[{idx}] {entry['label']}")

    while True:
        answer = input('Selection (Enter to skip): ').strip()
        if not answer:
            return None
        try:
            selected = int(answer)
        except ValueError:
            print('Invalid selection. Please enter a number from the menu.')
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print('Invalid selection. Please enter a number from the menu.')


def _prompt_unresolved_classifications(table, overrides, override_path):
    assigned_labels = set()
    updated = False

    for _role, image_type, question in PROMPT_IMAGE_TYPE_ROLES:
        if 'UNKNOWN' not in {str(v) for v in table['IMAGE_TYPE']}:
            break

        present_types = {str(v) for v in table['IMAGE_TYPE'] if str(v) != 'UNKNOWN'}
        if image_type in present_types:
            continue

        options = [
            entry for entry in _build_unresolved_label_options(table)
            if entry['label'] not in assigned_labels
        ]
        if not options:
            break

        chosen = _prompt_numbered_label_selection(question, options)
        if chosen is None:
            continue

        for signature in chosen['signatures']:
            overrides[signature] = image_type
        assigned_labels.add(chosen['label'])
        save_image_type_overrides(overrides, override_path)
        table = _apply_classification_rows(table, overrides)
        updated = True

    return table, updated


def _prompt_ambiguous_classifications(ambiguous_groups):
    prompts = {}
    choices = '/'.join(USER_CLASS_TO_IMAGE_TYPE.keys())
    print('\nAmbiguous recurring subsets detected:')
    for group in ambiguous_groups:
        print(
            f"  - EXPTYPE_NORM='{group['exptype_norm']}' "
            f"OBJECT_NORM='{group['object_norm']}' count={group['count']}"
        )

    for group in ambiguous_groups:
        while True:
            answer = input(
                "Classify subset "
                f"(EXPTYPE_NORM='{group['exptype_norm']}', "
                f"OBJECT_NORM='{group['object_norm']}') as "
                f"[{choices}] (Enter to skip): "
            ).strip().lower()
            if not answer:
                break
            if answer in USER_CLASS_TO_IMAGE_TYPE:
                prompts[group['signature']] = USER_CLASS_TO_IMAGE_TYPE[answer]
                break
            print(f"Invalid choice '{answer}'. Use one of: {choices}")
    return prompts


def _apply_classification_rows(table, overrides):
    image_type = []
    class_reason = []
    for exptype_norm, object_norm in zip(table['EXPTYPE_NORM'], table['OBJECT_NORM']):
        image_t, reason = classify_image_type(
            str(exptype_norm), str(object_norm), overrides=overrides
        )
        image_type.append(image_t)
        class_reason.append(reason)

    table['IMAGE_TYPE'] = np.array(image_type, dtype='U16')
    table['CLASS_REASON'] = np.array(class_reason, dtype='U64')
    return table

def construct_table_of_images(
    path_to_list,
    extra_columns=None,
    override_path=None,
    interactive=None,
):
    '''
    Creates an astropy table from a list of input images.
    By default only the columns:\n
    OBJECT\n
    OPAMP\n
    SHOE\n
    EXPTYPE\n
    LC-TIME\n
    NIGHT\n
    filename_input (the actual filename in the input list)\n
    will be included in the table, extra columns may be added via the extra_column argument.
    '''
    if extra_columns is None:
        extra_columns = []

    prepath = os.path.dirname(path_to_list)
    list_file = open(path_to_list)
    
    columns=['OBJECT', 'OPAMP', 'SHOE', 'EXPTYPE', 'LC-TIME', 'NIGHT']
    columns.extend(extra_columns)
    
    rows = []
    for f in list_file:
        PATH = os.path.join(prepath, f.strip())
        hdr = fits.getheader(PATH)
        row = {'FILENAME': hdr.get('FILENAME', '')}
        for c in columns:
            row.update({c: hdr.get(c, '')})
        row.update({'filename_input':f.strip()})
        rows.append(row)
        
    table_out = Table(rows)
    table_out['filename_input'] = table_out['filename_input'].astype('U256')

    table_out['EXPTYPE_NORM'] = np.array(
        [normalize_header_value(v) for v in table_out['EXPTYPE']],
        dtype='U128'
    )
    table_out['OBJECT_NORM'] = np.array(
        [normalize_header_value(v) for v in table_out['OBJECT']],
        dtype='U256'
    )

    overrides, resolved_override_path = load_image_type_overrides(override_path)
    table_out = _apply_classification_rows(table_out, overrides)

    ambiguous_groups = _iter_ambiguous_groups(table_out, overrides)
    unresolved = _iter_unresolved_signatures(table_out)

    if interactive is None:
        interactive = sys.stdin.isatty()

    if ambiguous_groups and interactive:
        print(f"\nFound {len(ambiguous_groups)} ambiguous recurring subset(s).")
        new_overrides = _prompt_ambiguous_classifications(ambiguous_groups)
        if new_overrides:
            overrides.update(new_overrides)
            save_image_type_overrides(overrides, resolved_override_path)
            table_out = _apply_classification_rows(table_out, overrides)
            print(
                f"Saved {len(new_overrides)} classification override(s) to "
                f"{resolved_override_path}"
            )
    elif ambiguous_groups:
        print(
            f"WARNING: {len(ambiguous_groups)} ambiguous recurring subset(s) "
            "left as UNKNOWN in non-interactive mode."
        )

    if interactive and 'UNKNOWN' in {str(v) for v in table_out['IMAGE_TYPE']}:
        table_out, prompted_updates = _prompt_unresolved_classifications(
            table_out,
            overrides,
            resolved_override_path,
        )
        if prompted_updates:
            print(f"Saved classification override(s) to {resolved_override_path}")

    unresolved = _iter_unresolved_signatures(table_out)

    table_out.meta['IMAGE_TYPE_OVERRIDE_PATH'] = resolved_override_path
    table_out.meta['UNRESOLVED_AMBIGUOUS_SIGNATURES'] = unresolved
    
    list_file.close()
    return table_out

def get_timeline(intable, merge_times=True):
    '''
    Split the input astropy table of images into tables by LC-TIME
    As some images may only be 1 or 2 seconds apart, this function
    attempts to crrect this by changing the times to the lower value.
    This is set via the merge_times argument
    
    Ouputs: for each observation(images taken at the same time)
    this produces a list of filenames that belong to the same observation,
    saves it and returns a list of lists of filenames for later processing.
    '''
    timesteps = np.unique(intable['LC-TIME'])
    intable_tcorr = intable.copy()
    if merge_times:
        dtimes = [datetime.strptime(ts, '%H:%M:%S') for ts in timesteps]
        
        for prev_str, nxt_str, prev_dt, nxt_dt in zip(timesteps, timesteps[1:], dtimes, dtimes[1:]):
            if nxt_dt - prev_dt == timedelta(seconds=2) or nxt_dt - prev_dt == timedelta(seconds=1):
                print('merging', nxt_str, '->', prev_str)
                mask = intable_tcorr['LC-TIME'] == nxt_str
                intable_tcorr['LC-TIME'][mask] = prev_str
        timesteps = np.unique(intable_tcorr['LC-TIME'])
        
    '''
    time_dict = {}
    for t in timesteps:
        types = np.unique(TimeTable[TimeTable['LC-TIME'] == t]['EXPTYPE'])
        time_dict[t] = types[0]
        #print(t, types[0])
    
    TimeTable = Table({'LC-TIME': list(time_dict.keys()),
                       'EXPTYPE': list(time_dict.values())})
    '''
    return intable_tcorr

def list_images(intable):
    '''
    Produces a list of images of the same type, taken at the same time
    Output: NIGHT_t(h-m-s)_EXPTYPE.list
    '''
    if len(intable)//8 != len(np.unique(intable['LC-TIME'])):
        print('Time alignment warning')
    
    night = str(np.unique(intable['NIGHT'])[0])
    list_of_exposures = []
    pathlist_list = []
    for en,t in enumerate(np.unique(intable['LC-TIME'])):    
        time_name = t.replace(':','-')
        type_name = str(np.unique(intable['EXPTYPE'][intable['LC-TIME']==t])[0])
        list_name = night+'_'+time_name+'_'+type_name+'.list'
        
        list_of_images = []
        pathlist_list.append(list_name)
        with open(list_name, 'w') as l:
            for r in intable[intable['LC-TIME']==t]:
                fname = str(r['filename_input'])
                l.write(fname + '\n')
                list_of_images.append(fname)
        list_of_exposures.append(list_of_images)
    return list_of_exposures, pathlist_list

def infiles_to_lists(infiles_path: str, extra_columns=[], merge_times=True):
    '''
    Complete sorting chain. From a list of fits images (infiles_path) this produces a master list of individual image sets (2 shoes x 4 opamps),
    determined by timestamp. Each of these sets is represented by a list, included in the master_list of set lists.
    timetable is a table of unique observing times and the exposure types as those times.
    image_table is a table of important columns and image file names. Columns included can be expanded by extra_columns argument.
    '''
    intable = construct_table_of_images(infiles_path, extra_columns=extra_columns)
    images_table = get_timeline(intable, merge_times=True)
    list_of_exposures, master_list = list_images(images_table)
    return images_table, list_of_exposures, master_list

def split_path(path_: str):
    dirpath, fname = os.path.split(path_)
    base, ext = os.path.splitext(fname)
    return [dirpath, base, ext]

def generate_output_list(list_of_list_paths: list, suffix: str):
    outlists = []
    for list_path in list_of_list_paths:
        with open(list_path) as inlist:
            filenames = []
            for line in inlist:
                filenames.append(line.strip())
        
        lpath = split_path(list_path)
        outlist_path = os.path.join(lpath[0], lpath[1] + '-' + suffix + lpath[2])
        with open(outlist_path, 'w') as outlist:
            for impath in filenames:
                outpath = split_path(impath.strip())
                name = outpath[1] + '-' + suffix + outpath[2]
                outlist.write(os.path.join(outpath[0], name) + '\n')
        outlists.append(outlist_path)
    return outlists

def table_to_list(intable: Table, listpath: str, suffix: str):
    """Write two lists derived from an image table.

    * ``listpath`` will contain the ``FILENAME`` values with ``.fits``
      appended.
    * The output list (returned as ``outpath``) will have the same
      names but with ``-<suffix>`` inserted before the ``.fits``
      extension.

    The function returns a tuple ``(listpath, outpath)``.
    """
    names = [str(n) for n in intable['FILENAME']]

    with open(listpath, 'w') as f:
        f.writelines(f"{name}.fits\n" for name in names)

    pre, fn, ext = split_path(listpath)
    outpath = os.path.join(pre, f"{fn}-{suffix}{ext}")

    with open(outpath, 'w') as f:
        f.writelines(f"{name}-{suffix}.fits\n" for name in names)

    return listpath, outpath
        
def get_images(path_to_dir, inlist = None):
    images = []
    for x in os.listdir(path_to_dir):
        if x.endswith(".fits"):
            images.append(x)
    return images

#asd = get_images('/home/oskar/faks/stockholm/test_data/')
