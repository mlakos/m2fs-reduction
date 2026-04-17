import os
from pathlib import Path
import sys
import types
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


if 'pyraf' not in sys.modules:
    class _IrafStub:
        yes = True
        no = False

        def __getattr__(self, _name):
            return self

        def __call__(self, *args, **kwargs):
            return None

        def unlearn(self, *args, **kwargs):
            return None

    sys.modules['pyraf'] = types.SimpleNamespace(iraf=_IrafStub())

from file_handler import (
    classify_image_type,
    construct_table_of_images,
    generate_output_list,
    normalize_header_value,
)
from image_processing import (
    build_resume_combined_images,
    resolve_processing_dirs as resolve_processing_dirs_image,
    stack_dark_frames,
    science_frame_stacking,
    step7_dark_subtract,
    step2_overscan_trim,
    step9_stack_science,
)
from reduce_echelle import (
    _assert_local_apertures,
    _build_local_aperture_mapping,
    _read_apertures_from_apnum_cards,
    _renumber_multispec_fits_apertures,
    auto_wavelength_propagation,
    default_affiliation_map_path,
    default_extraction_pairs_path,
    default_star_geometry_path,
    choose_preprocess_plan,
    classify_role,
    discover_inputs,
    extract_all_stars,
    inspect_preprocess_state,
    resolve_inputs,
    resolve_processing_dirs as resolve_processing_dirs_echelle,
    write_infiles_from_directory,
)


def _write_test_fits(path, **header_items):
    hdr = fits.Header()
    for key, value in header_items.items():
        hdr[key] = value
    hdr.setdefault('FILENAME', path.stem)
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(path)


def _write_multispec_test_fits(path, apertures):
    hdr = fits.Header()
    for idx, ap in enumerate(apertures, start=1):
        hdr[f'APNUM{idx}'] = f'{int(ap)} 1 0'

    specs = []
    for idx, ap in enumerate(apertures, start=1):
        specs.append(f'spec{idx} = "{int(ap)} 1 100 200"')
    payload = 'wtype=multispec ' + ' '.join(specs)
    chunks = [payload[i:i + 68] for i in range(0, len(payload), 68)] or ['']
    for idx, chunk in enumerate(chunks, start=1):
        hdr[f'WAT2_{idx:03d}'] = chunk

    fits.PrimaryHDU(data=np.zeros((2, 8), dtype=np.float32), header=hdr).writeto(path)


def test_normalize_header_value_edge_cases():
    assert normalize_header_value('  ConFig   14   Twilight  ') == 'config 14 twilight'
    assert normalize_header_value('') == ''
    assert normalize_header_value(None) == ''


def test_classification_precedence_exptype_over_object():
    image_type, reason = classify_image_type('bias', 'config 14 twilight')
    assert image_type == 'BIAS'
    assert reason.startswith('exptype_')

    image_type, reason = classify_image_type('dark master', 'jsimon h3')
    assert image_type == 'DARK_MASTER'
    assert reason == 'exptype_dark_master'


def test_classification_object_subclasses_and_unknown():
    assert classify_image_type('object', 'fibermap config 14')[0] == 'FIBERMAP'
    assert classify_image_type('lamp', 'jsimon h3 quartz')[0] == 'QUARTZ'

    image_type, reason = classify_image_type('object', 'twilight flat')
    assert image_type == 'UNKNOWN'
    assert reason == 'ambiguous_object_tokens'


def test_construct_table_applies_override_and_adds_columns(tmp_path):
    raw_dir = tmp_path / 'night'
    raw_dir.mkdir()

    p1 = raw_dir / 'b0001c1.fits'
    p2 = raw_dir / 'b0001c2.fits'
    common = {
        'OBJECT': 'Instrument Check',
        'OPAMP': '1',
        'SHOE': 'B',
        'EXPTYPE': 'Focus',
        'LC-TIME': '01:00:00',
        'NIGHT': '15Sep2014',
    }
    _write_test_fits(p1, **common)
    common2 = dict(common)
    common2['OPAMP'] = '2'
    _write_test_fits(p2, **common2)

    infiles = raw_dir / 'infiles'
    infiles.write_text('b0001c1.fits\nb0001c2.fits\n')

    override_path = tmp_path / 'overrides.json'
    override_path.write_text(
        '{\n'
        '  "version": 1,\n'
        '  "mappings": {\n'
        '    "exptype=focus|object=instrument check": "lamp"\n'
        '  }\n'
        '}\n'
    )

    table = construct_table_of_images(
        str(infiles),
        override_path=str(override_path),
        interactive=False,
    )

    for required_col in ['EXPTYPE_NORM', 'OBJECT_NORM', 'IMAGE_TYPE', 'CLASS_REASON']:
        assert required_col in table.colnames

    assert set(table['IMAGE_TYPE']) == {'LAMP'}
    assert all(str(reason).startswith('override_') for reason in table['CLASS_REASON'])


def test_construct_table_marks_unresolved_ambiguity_noninteractive(tmp_path):
    raw_dir = tmp_path / 'night'
    raw_dir.mkdir()

    p1 = raw_dir / 'b0002c1.fits'
    p2 = raw_dir / 'b0002c2.fits'
    h1 = {
        'OBJECT': 'Twilight Flat',
        'OPAMP': '1',
        'SHOE': 'B',
        'EXPTYPE': 'Object',
        'LC-TIME': '01:10:00',
        'NIGHT': '15Sep2014',
    }
    h2 = dict(h1)
    h2['OPAMP'] = '2'
    _write_test_fits(p1, **h1)
    _write_test_fits(p2, **h2)

    infiles = raw_dir / 'infiles'
    infiles.write_text('b0002c1.fits\nb0002c2.fits\n')

    table = construct_table_of_images(str(infiles), interactive=False)
    assert set(table['IMAGE_TYPE']) == {'UNKNOWN'}

    unresolved = table.meta.get('UNRESOLVED_AMBIGUOUS_SIGNATURES', [])
    assert unresolved
    assert unresolved[0] == 'exptype=object|object=twilight flat'


def test_construct_table_prompts_unresolved_classes_interactive(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'night'
    raw_dir.mkdir()

    rows = [
        ('b0010c1.fits', '1', 'a'),
        ('b0010c2.fits', '2', 'b'),
        ('b0010c3.fits', '3', 'c'),
    ]
    for fname, opamp, obj in rows:
        _write_test_fits(
            raw_dir / fname,
            OBJECT=obj,
            OPAMP=opamp,
            SHOE='B',
            EXPTYPE='Focus',
            **{'LC-TIME': '01:20:00', 'NIGHT': '15Sep2014'}
        )

    infiles = raw_dir / 'infiles'
    infiles.write_text('b0010c1.fits\nb0010c2.fits\nb0010c3.fits\n')

    # dark -> b, science/object -> a, flat -> c
    answers = iter(['2', '1', '1'])
    monkeypatch.setattr('builtins.input', lambda _prompt='': next(answers))

    override_path = tmp_path / 'overrides.json'
    table = construct_table_of_images(
        str(infiles),
        override_path=str(override_path),
        interactive=True,
    )

    out = {str(row['OBJECT']).strip(): str(row['IMAGE_TYPE']) for row in table}
    assert out['a'] == 'SCIENCE'
    assert out['b'] == 'DARK'
    assert out['c'] == 'QUARTZ'
    assert 'UNKNOWN' not in set(table['IMAGE_TYPE'])

    payload = json.loads(override_path.read_text())
    mappings = payload.get('mappings', {})
    assert mappings['exptype=focus|object=a'] == 'SCIENCE'
    assert mappings['exptype=focus|object=b'] == 'DARK'
    assert mappings['exptype=focus|object=c'] == 'QUARTZ'


def test_resolve_processing_dirs_in_image_and_echelle(tmp_path):
    raw_dir = tmp_path / 'night'
    proc_dir = raw_dir / 'proc'
    proc_dir.mkdir(parents=True)

    infiles = raw_dir / 'infiles'
    infiles.write_text('dummy.fits\n')

    raw_i, proc_i, _ = resolve_processing_dirs_image(str(infiles))
    assert raw_i == str(raw_dir)
    assert proc_i == str(proc_dir)

    raw_e, proc_e = resolve_processing_dirs_echelle(str(raw_dir))
    assert raw_e == str(raw_dir)
    assert proc_e == str(proc_dir)

    raw_e2, proc_e2 = resolve_processing_dirs_echelle(str(proc_dir))
    assert raw_e2 == str(raw_dir)
    assert proc_e2 == str(proc_dir)


def test_step2_reads_raw_and_writes_proc_lists(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'night'
    proc_dir = raw_dir / 'proc'
    proc_dir.mkdir(parents=True)

    master_list = proc_dir / '15Sep2014_01-00-00_Object.list'
    master_list.write_text('b1000c1.fits\nb1000c2.fits\n')

    calls = []

    def fake_ovefit(inlist, outlist):
        in_entries = Path(inlist).read_text().splitlines()
        out_entries = Path(outlist).read_text().splitlines()
        calls.append((inlist, outlist, in_entries, out_entries))

    import image_processing as img

    monkeypatch.setattr(img.pyraf_utils, 'run_ccdproc_ovefit_trim', fake_ovefit)

    table = Table(
        {
            'FILENAME': ['b1000c1', 'b1000c2'],
            'filename_input': ['b1000c1.fits', 'b1000c2.fits'],
        }
    )

    prev_cwd = os.getcwd()
    os.chdir(proc_dir)
    try:
        out = step2_overscan_trim(table, [str(master_list)], raw_dir=str(raw_dir))
    finally:
        os.chdir(prev_cwd)

    assert len(calls) == 1
    _, _, in_entries, out_entries = calls[0]

    assert in_entries == [
        str(raw_dir / 'b1000c1.fits'),
        str(raw_dir / 'b1000c2.fits'),
    ]
    assert out_entries == ['b1000c1-ot.fits', 'b1000c2-ot.fits']
    assert list(out['filename_input']) == ['b1000c1-ot.fits', 'b1000c2-ot.fits']


def test_step9_routing_uses_image_type_and_object_filter(monkeypatch):
    table = Table(
        {
            'IMAGE_TYPE': ['SCIENCE', 'SCIENCE', 'TWILIGHT', 'QUARTZ', 'LAMP', 'FIBERMAP'],
            'OBJECT_NORM': ['target one', 'target two', 'config 14 twilight', 'jsimon h3 quartz', 'thar', 'fibermap config'],
            'OBJECT': ['Target One', 'Target Two', 'Config 14 Twilight', 'JSimon H3 Quartz', 'ThAr', 'Fibermap Config'],
            'EXPTYPE': ['Object', 'Object', 'Object', 'Lamp', 'Lamp', 'Object'],
            'NIGHT': ['15Sep2014'] * 6,
            'SHOE': ['B'] * 6,
            'filename_input': ['a.fits', 'b.fits', 'c.fits', 'd.fits', 'e.fits', 'f.fits'],
            'FILENAME': ['a', 'b', 'c', 'd', 'e', 'f'],
        }
    )

    calls = []

    def fake_stack(comb_img_table, sci_mask, mode, scale='none', image_type_label='SCIENCE'):
        calls.append((image_type_label, mode, scale, int(np.sum(sci_mask))))
        return comb_img_table, []

    import image_processing as img

    monkeypatch.setattr(img, 'science_frame_stacking', fake_stack)

    step9_stack_science(table, object_name='Target   One')

    assert ('SCIENCE', 'sum', 'none', 1) in calls
    assert ('TWILIGHT', 'median', 'median', 1) in calls
    assert ('QUARTZ', 'median', 'median', 1) in calls
    assert ('LAMP', 'median', 'median', 1) in calls
    assert not any(call[0] == 'FIBERMAP' for call in calls)


def test_write_infiles_from_directory_prefers_raw_chip_pattern(tmp_path):
    raw_dir = tmp_path / 'night'
    raw_dir.mkdir()

    (raw_dir / 'b0001c1.fits').write_text('')
    (raw_dir / 'r0001c4.fits').write_text('')
    (raw_dir / 'Object-b0001-ot-full.fits').write_text('')

    out = write_infiles_from_directory(str(raw_dir), infiles_path=str(raw_dir / 'infiles'))
    lines = Path(out).read_text().splitlines()
    assert lines == ['b0001c1.fits', 'r0001c4.fits']


def test_write_infiles_from_directory_dark_reuse_is_cross_night_but_plate_constrained(tmp_path):
    raw_dir = tmp_path / 'night'
    raw_dir.mkdir()

    _write_test_fits(
        raw_dir / 'b1000c1.fits',
        OBJECT='Target',
        EXPTYPE='Object',
        IMAGE_TYPE='SCIENCE',
        NIGHT='15Sep2014',
        SHOE='B',
        PLATE='P1',
    )
    _write_test_fits(
        raw_dir / 'b1001c1.fits',
        OBJECT='Target',
        EXPTYPE='Object',
        IMAGE_TYPE='SCIENCE',
        NIGHT='16Sep2014',
        SHOE='B',
        PLATE='P1',
    )
    _write_test_fits(
        raw_dir / 'b1002c1.fits',
        OBJECT='Dark frame',
        EXPTYPE='Dark',
        IMAGE_TYPE='DARK',
        NIGHT='16Sep2014',
        SHOE='B',
        PLATE='P1',
    )
    _write_test_fits(
        raw_dir / 'b1003c1.fits',
        OBJECT='Dark frame',
        EXPTYPE='Dark',
        IMAGE_TYPE='DARK',
        NIGHT='16Sep2014',
        SHOE='B',
        PLATE='P2',
    )

    out = write_infiles_from_directory(
        str(raw_dir),
        infiles_path=str(raw_dir / 'infiles'),
        night='15Sep2014',
        shoe='B',
        plate='P1',
    )
    lines = Path(out).read_text().splitlines()
    assert lines == ['b1000c1.fits', 'b1002c1.fits']


def test_generate_output_list_writes_one_line_per_input(tmp_path):
    input_list = tmp_path / 'group.list'
    input_list.write_text('a.fits\nsub/b.fits\n')

    outlists = generate_output_list([str(input_list)], 'ot')
    assert len(outlists) == 1

    lines = Path(outlists[0]).read_text().splitlines()
    assert lines == ['a-ot.fits', 'sub/b-ot.fits']


def test_science_frame_stacking_separates_object_norm_and_plate(tmp_path, monkeypatch):
    rows = []
    exposures = [
        ('Target One', 'target one', 'P1', 'o1a'),
        ('Target One', 'target one', 'P1', 'o1b'),
        ('Target One', 'target one', 'P2', 'o1c'),
        ('Target One', 'target one', 'P2', 'o1d'),
        ('Target Two', 'target two', 'P1', 'o2a'),
        ('Target Two', 'target two', 'P1', 'o2b'),
    ]
    for obj, obj_norm, plate, stem in exposures:
        p = tmp_path / f'{stem}.fits'
        _write_test_fits(
            p,
            OBJECT=obj,
            EXPTYPE='Object',
            IMAGE_TYPE='SCIENCE',
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE=plate,
            EGAIN=1.0,
        )
        rows.append(
            {
                'FILENAME': stem,
                'filename_input': str(p),
                'OBJECT': obj,
                'OBJECT_NORM': obj_norm,
                'EXPTYPE': 'Object',
                'EXPTYPE_NORM': 'object',
                'IMAGE_TYPE': 'SCIENCE',
                'CLASS_REASON': 'test',
                'NIGHT': '15Sep2014',
                'SHOE': 'B',
                'PLATE': plate,
            }
        )

    table = Table(rows=rows)
    mask = np.ones(len(table), dtype=bool)
    calls = []

    def fake_stack_science_images(stack_listpath, out_filepath, mode='sum', scale='none'):
        calls.append((Path(stack_listpath).name, Path(stack_listpath).read_text().splitlines(), out_filepath))
        fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(out_filepath, overwrite=True)

    import image_processing as img

    monkeypatch.setattr(img.pyraf_utils, 'stack_science_images', fake_stack_science_images)

    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _, outputs = science_frame_stacking(table, mask, mode='sum', image_type_label='SCIENCE')
    finally:
        os.chdir(prev_cwd)

    assert len(calls) == 3
    assert len(outputs) == 3
    grouped_inputs = [sorted(Path(p.split('[', 1)[0]).name for p in list_lines) for _, list_lines, _ in calls]
    assert sorted(grouped_inputs) == sorted([
        ['o1a.fits', 'o1b.fits'],
        ['o1c.fits', 'o1d.fits'],
        ['o2a.fits', 'o2b.fits'],
    ])

    out_names = sorted(Path(p).name for p in outputs)
    assert out_names == sorted([
        'target_one_15Sep2014_B_P1-sstack.fits',
        'target_one_15Sep2014_B_P2-sstack.fits',
        'target_two_15Sep2014_B_P1-sstack.fits',
    ])

    list_names = sorted(name for name, _, _ in calls)
    assert list_names == sorted([
        'stack_science_15Sep2014_B_P1_target_one.list',
        'stack_science_15Sep2014_B_P2_target_one.list',
        'stack_science_15Sep2014_B_P1_target_two.list',
    ])


def test_reduce_echelle_classify_role_prefers_image_type():
    meta = {
        'IMAGE_TYPE': 'SCIENCE',
        'EXPTYPE': 'Lamp',
        'OBJECT': 'ThAr calibration',
    }
    assert classify_role(meta) == 'object'


def test_discover_inputs_constrains_calibrations_to_object_plate(tmp_path):
    def write_product(name, image_type, object_text, plate):
        path = tmp_path / name
        _write_test_fits(
            path,
            OBJECT=object_text,
            EXPTYPE='Object' if image_type == 'SCIENCE' else 'Lamp',
            IMAGE_TYPE=image_type,
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE=plate,
            STACKED=True,
            STACKTYP='sum' if image_type == 'SCIENCE' else 'median',
        )
        return str(path)

    object_p1 = write_product('object_p1_sstack.fits', 'SCIENCE', 'Target A', 'P1')
    quartz_p1 = write_product('quartz_p1_mstack.fits', 'QUARTZ', 'Quartz Lamp', 'P1')
    thar_p1 = write_product('thar_p1_mstack.fits', 'LAMP', 'ThAr', 'P1')
    twi_p1 = write_product('twilight_p1_mstack.fits', 'TWILIGHT', 'Twilight', 'P1')

    write_product('object_p2_mstack.fits', 'SCIENCE', 'Target A', 'P2')
    write_product('quartz_p2_mstack.fits', 'QUARTZ', 'Quartz Lamp', 'P2')
    write_product('thar_p2_mstack.fits', 'LAMP', 'ThAr', 'P2')
    write_product('twilight_p2_mstack.fits', 'TWILIGHT', 'Twilight', 'P2')

    selected = discover_inputs(
        str(tmp_path),
        night='15Sep2014',
        shoe='B',
        object_name='Target A',
        required_roles={'object', 'quartz', 'thar', 'twilight'},
    )

    assert selected['object'] == object_p1
    assert selected['quartz'] == quartz_p1
    assert selected['thar'] == thar_p1
    assert selected['twilight'] == twi_p1


def test_resolve_inputs_prompts_from_header_inventory(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'night'
    proc_dir = raw_dir / 'proc'
    proc_dir.mkdir(parents=True)

    role_paths = {}
    for name, exptype, obj in [
        ('cfg_lamp.fits', 'ConfigA', 'LabelA'),
        ('cfg_obj.fits', 'ConfigB', 'LabelB'),
        ('cfg_flat.fits', 'ConfigC', 'LabelC'),
    ]:
        path = raw_dir / name
        _write_test_fits(
            path,
            OBJECT=obj,
            EXPTYPE=exptype,
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE='P1',
        )
        role_paths[name] = str(path)

    args = types.SimpleNamespace(
        quartz=None,
        thar=None,
        object=None,
        twilight=None,
        input_dir=str(proc_dir),
        raw_input_dir=str(raw_dir),
        proc_dir=str(proc_dir),
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name=None,
        image_type_override_path=str(tmp_path / 'image_type_overrides.json'),
    )

    # required roles order controls prompt order in this test
    answers = iter(['2', '1', '3'])
    monkeypatch.setattr('builtins.input', lambda _prompt='': next(answers))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)

    resolved = resolve_inputs(args, required_roles=('object', 'thar', 'quartz'))

    assert resolved['object'] == role_paths['cfg_obj.fits']
    assert resolved['thar'] == role_paths['cfg_lamp.fits']
    assert resolved['quartz'] == role_paths['cfg_flat.fits']


def test_resolve_inputs_noninteractive_reports_discovered_labels(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'night'
    proc_dir = raw_dir / 'proc'
    proc_dir.mkdir(parents=True)

    for name, exptype, obj in [
        ('cfg_one.fits', 'ConfigA', 'LabelA'),
        ('cfg_two.fits', 'ConfigB', 'LabelB'),
    ]:
        _write_test_fits(
            raw_dir / name,
            OBJECT=obj,
            EXPTYPE=exptype,
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE='P1',
        )

    args = types.SimpleNamespace(
        quartz=None,
        thar=None,
        object=None,
        twilight=None,
        input_dir=str(proc_dir),
        raw_input_dir=str(raw_dir),
        proc_dir=str(proc_dir),
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name=None,
        image_type_override_path=str(tmp_path / 'image_type_overrides.json'),
    )

    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)

    with pytest.raises(RuntimeError) as excinfo:
        resolve_inputs(args, required_roles=('object', 'thar'))

    message = str(excinfo.value)
    assert 'Discovered unique EXPTYPE/OBJECT combinations:' in message
    assert "EXPTYPE='ConfigA' OBJECT='LabelA'" in message
    assert "EXPTYPE='ConfigB' OBJECT='LabelB'" in message
    assert 'Non-interactive mode cannot prompt for unresolved roles.' in message


def test_step7_dark_subtract_routes_all_processable_non_dark(monkeypatch):
    table = Table(
        {
            'IMAGE_TYPE': ['SCIENCE', 'TWILIGHT', 'QUARTZ', 'LAMP', 'UNKNOWN', 'DARK_MASTER'],
            'OBJECT_NORM': ['target', 'twilight', 'quartz', 'thar', 'unknown', 'dark master'],
            'NIGHT': ['15Sep2014', '15Sep2014', '15Sep2014', '15Sep2014', '15Sep2014', 'multi'],
            'SHOE': ['B', 'B', 'B', 'B', 'B', 'B'],
            'PLATE': ['P1', 'P1', 'P1', 'P1', 'P1', 'P1'],
            'filename_input': ['a.fits', 'b.fits', 'c.fits', 'd.fits', 'e.fits', 'dark_master.fits'],
            'FILENAME': ['a', 'b', 'c', 'd', 'e', 'dark_master'],
        }
    )

    calls = []

    def fake_subtract_dark_mask(intable, mask, master_dark_path, label='science'):
        selected_types = set(intable['IMAGE_TYPE'][mask])
        calls.append((selected_types, master_dark_path, label))
        return intable

    import image_processing as img

    monkeypatch.setattr(img, 'subtract_dark_mask', fake_subtract_dark_mask)

    _, non_dark = step7_dark_subtract(table)

    assert np.sum(non_dark) == 5
    assert len(calls) == 1
    selected_types, master_dark_path, _ = calls[0]
    assert selected_types == {'SCIENCE', 'TWILIGHT', 'QUARTZ', 'LAMP'}
    assert master_dark_path == 'dark_master.fits'


def test_step7_dark_subtract_uses_first_dark_master_for_same_shoe_and_plate(monkeypatch):
    table = Table(
        {
            'IMAGE_TYPE': ['SCIENCE', 'DARK_MASTER', 'DARK_MASTER', 'DARK_MASTER'],
            'OBJECT_NORM': ['target', 'dark master', 'dark master', 'dark master'],
            'NIGHT': ['15Sep2014', 'multi', 'multi', 'multi'],
            'SHOE': ['B', 'B', 'B', 'B'],
            'PLATE': ['P1', 'P1', 'P1', 'P2'],
            'filename_input': ['obj.fits', 'dm_first.fits', 'dm_second.fits', 'dm_other_plate.fits'],
            'FILENAME': ['obj', 'dm_first', 'dm_second', 'dm_other_plate'],
        }
    )

    calls = []

    def fake_subtract_dark_mask(intable, mask, master_dark_path, label='science'):
        selected_types = set(intable['IMAGE_TYPE'][mask])
        calls.append((selected_types, master_dark_path, label))
        return intable

    import image_processing as img

    monkeypatch.setattr(img, 'subtract_dark_mask', fake_subtract_dark_mask)

    _, non_dark = step7_dark_subtract(table)

    assert np.sum(non_dark) == 1
    assert len(calls) == 1
    selected_types, master_dark_path, _ = calls[0]
    assert selected_types == {'SCIENCE'}
    assert master_dark_path == 'dm_first.fits'


def test_stack_dark_frames_names_are_plate_safe(tmp_path, monkeypatch):
    rows = []
    for plate, stem in [('P1', 'd1'), ('P1', 'd2'), ('P2', 'd3'), ('P2', 'd4')]:
        p = tmp_path / f'{stem}.fits'
        _write_test_fits(
            p,
            OBJECT='Dark Frame',
            EXPTYPE='Dark',
            IMAGE_TYPE='DARK',
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE=plate,
        )
        rows.append(
            {
                'FILENAME': stem,
                'filename_input': str(p),
                'OBJECT': 'Dark Frame',
                'OBJECT_NORM': 'dark frame',
                'EXPTYPE': 'Dark',
                'EXPTYPE_NORM': 'dark',
                'IMAGE_TYPE': 'DARK',
                'CLASS_REASON': 'test',
                'NIGHT': '15Sep2014',
                'SHOE': 'B',
                'PLATE': plate,
                'STACKTYP': '',
            }
        )

    table = Table(rows=rows)
    calls = []

    def fake_stack_science_images(stack_listpath, out_filepath, mode='median', scale='none'):
        calls.append((Path(stack_listpath).name, Path(out_filepath).name))
        fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(out_filepath, overwrite=True)

    import image_processing as img

    monkeypatch.setattr(img.pyraf_utils, 'stack_science_images', fake_stack_science_images)

    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        out = stack_dark_frames(table, method='median')
    finally:
        os.chdir(prev_cwd)

    assert len(calls) == 2
    out_names = sorted(name for _, name in calls)
    assert out_names == sorted([
        '15Sep2014-Dark_master-B_P1me.fits',
        '15Sep2014-Dark_master-B_P2me.fits',
    ])
    list_names = sorted(name for name, _ in calls)
    assert list_names == sorted([
        'dark_stack_15Sep2014_B_P1me.list',
        'dark_stack_15Sep2014_B_P2me.list',
    ])
    assert np.sum(out['IMAGE_TYPE'] == 'DARK_MASTER') == 2


def test_inspect_preprocess_state_and_plan_reuses_complete_context(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()

    def write_stack(name, image_type, obj):
        _write_test_fits(
            proc / name,
            OBJECT=obj,
            EXPTYPE='Object_stack',
            IMAGE_TYPE=image_type,
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE='P1',
            STACKED=True,
            STACKTYP='sum' if image_type == 'SCIENCE' else 'median',
        )

    write_stack('object_sstack.fits', 'SCIENCE', 'Target A')
    write_stack('quartz_mstack.fits', 'QUARTZ', 'Quartz')
    write_stack('thar_mstack.fits', 'LAMP', 'ThAr')

    state = inspect_preprocess_state(
        str(proc),
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name='Target A',
        required_roles={'object', 'quartz', 'thar'},
    )
    plan = choose_preprocess_plan(state)

    assert state['missing_roles'] == ()
    assert plan['skip'] is True
    assert plan['reason'].startswith('all required stacked')


def test_choose_preprocess_plan_prefers_late_resume_when_mcrr_exists(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()

    for stem, image_type, obj in [
        ('obj1-full-mcrr.fits', 'SCIENCE', 'Target A'),
        ('quartz-full-mcrr.fits', 'QUARTZ', 'Quartz'),
        ('thar-full-mcrr.fits', 'LAMP', 'ThAr'),
    ]:
        _write_test_fits(
            proc / stem,
            OBJECT=obj,
            EXPTYPE='Object',
            IMAGE_TYPE=image_type,
            NIGHT='15Sep2014',
            SHOE='B',
            PLATE='P1',
            STACKED=False,
        )

    state = inspect_preprocess_state(
        str(proc),
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name='Target A',
        required_roles={'object', 'quartz', 'thar'},
    )
    plan = choose_preprocess_plan(state)

    assert plan['skip'] is False
    assert plan['start_step'] == 9
    assert plan['resume_from_proc'] is True


def test_build_resume_combined_images_prefers_best_variant_for_step(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()

    base_hdr = {
        'OBJECT': 'Target A',
        'EXPTYPE': 'Object',
        'IMAGE_TYPE': 'SCIENCE',
        'NIGHT': '15Sep2014',
        'SHOE': 'B',
        'PLATE': 'P1',
    }
    _write_test_fits(proc / 'target-full.fits', **base_hdr)
    _write_test_fits(proc / 'target-full-D.fits', **base_hdr)
    _write_test_fits(proc / 'target-full-D-mcrr.fits', **base_hdr)

    t9 = build_resume_combined_images(
        str(proc),
        start_step=9,
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name='Target A',
    )
    assert len(t9) == 1
    assert str(t9['filename_input'][0]).endswith('target-full-D-mcrr.fits')

    t7 = build_resume_combined_images(
        str(proc),
        start_step=7,
        night='15Sep2014',
        shoe='B',
        plate='P1',
        object_name='Target A',
    )
    assert len(t7) == 1
    assert str(t7['filename_input'][0]).endswith('target-full.fits')


def test_build_resume_combined_images_dark_master_is_plate_keyed_first_seen(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()

    _write_test_fits(
        proc / 'a_dark_master_p1.fits',
        OBJECT='Dark Frame',
        EXPTYPE='Dark_master',
        IMAGE_TYPE='DARK_MASTER',
        NIGHT='18Dec2014',
        SHOE='B',
        PLATE='P1',
        STACKED=True,
        STACKTYP='median',
    )
    _write_test_fits(
        proc / 'z_dark_master_p1.fits',
        OBJECT='Dark Frame',
        EXPTYPE='Dark_master',
        IMAGE_TYPE='DARK_MASTER',
        NIGHT='21Dec2014',
        SHOE='B',
        PLATE='P1',
        STACKED=True,
        STACKTYP='median',
    )
    _write_test_fits(
        proc / 'm_dark_master_p2.fits',
        OBJECT='Dark Frame',
        EXPTYPE='Dark_master',
        IMAGE_TYPE='DARK_MASTER',
        NIGHT='18Dec2014',
        SHOE='B',
        PLATE='P2',
        STACKED=True,
        STACKTYP='median',
    )

    out = build_resume_combined_images(
        str(proc),
        start_step=7,
        night='15Sep2014',
        shoe='B',
        plate='P1',
    )

    assert len(out) == 1
    assert str(out['IMAGE_TYPE'][0]) == 'DARK_MASTER'
    assert str(out['PLATE'][0]) == 'P1'
    assert str(out['filename_input'][0]).endswith('a_dark_master_p1.fits')


def test_choose_preprocess_plan_honors_force_and_explicit_step():
    state = {
        'missing_roles': ('object',),
        'role_stage_sets': {'object': {'mosaic'}},
        'dark_master_count': 0,
        'dark_mosaic_count': 0,
    }

    forced = choose_preprocess_plan(state, force_preprocess=True)
    assert forced['start_step'] == 1
    assert forced['resume_from_proc'] is False

    explicit = choose_preprocess_plan(state, preprocess_from_step=8)
    assert explicit['start_step'] == 8
    assert explicit['resume_from_proc'] is True


def test_reduce_echelle_default_helper_paths_include_context_tokens():
    meta_by_role = {
        'object': {
            'OBJECT': 'Target One',
            'NIGHT': '15Sep2014',
            'SHOE': 'B',
            'PLATE': 'P1',
        },
        'quartz': {
            'OBJECT': 'Quartz Lamp',
            'NIGHT': '15Sep2014',
            'SHOE': 'B',
            'PLATE': 'P1',
        },
    }

    assert default_affiliation_map_path(meta_by_role) == 'affiliation_15Sep2014_B_P1.json'
    assert default_extraction_pairs_path(meta_by_role) == 'extraction_pairs_15Sep2014_B_P1_target_one.csv'
    assert default_star_geometry_path(meta_by_role) == 'star_geometry_15Sep2014_B_P1_target_one.json'


def test_build_local_aperture_mapping_maps_global_to_local():
    assert _build_local_aperture_mapping([5, 6, 7, 8]) == {5: 1, 6: 2, 7: 3, 8: 4}


def test_renumber_multispec_fits_apertures_updates_apnum_and_wat2(tmp_path):
    spec_path = tmp_path / 'thar_star02_ec.fits'
    _write_multispec_test_fits(spec_path, [5, 6])

    _renumber_multispec_fits_apertures(str(spec_path), {5: 1, 6: 2})

    assert _read_apertures_from_apnum_cards(str(spec_path)) == [1, 2]

    hdr = fits.getheader(spec_path)
    wat_keys = sorted(
        [k for k in hdr.keys() if str(k).startswith('WAT2_')],
        key=lambda key: int(str(key).split('_')[1]),
    )
    payload = ''.join(str(hdr[key]) for key in wat_keys)
    assert 'spec1 = "1 1 100 200"' in payload
    assert 'spec2 = "2 1 100 200"' in payload


def test_renumber_multispec_fits_apertures_global_four_to_local_four(tmp_path):
    spec_path = tmp_path / 'thar_star14_ec.fits'
    _write_multispec_test_fits(spec_path, [53, 54, 55, 56])

    _renumber_multispec_fits_apertures(
        str(spec_path),
        {53: 1, 54: 2, 55: 3, 56: 4},
    )

    assert _read_apertures_from_apnum_cards(str(spec_path)) == [1, 2, 3, 4]


def test_extract_all_stars_renumbers_object_and_thar_outputs(monkeypatch, tmp_path):
    def _parse_apertures(aperture_text):
        out = []
        for token in str(aperture_text).split(','):
            chunk = token.strip()
            if not chunk:
                continue
            if '-' in chunk:
                lo, hi = chunk.split('-', 1)
                lo_i = int(lo)
                hi_i = int(hi)
                if lo_i <= hi_i:
                    out.extend(range(lo_i, hi_i + 1))
                else:
                    out.extend(range(hi_i, lo_i + 1))
            else:
                out.append(int(chunk))
        return out

    def fake_apall_extract_star(image, _quartz, aperture_string, star_number):
        apertures = _parse_apertures(aperture_string)
        prefix = 'obj' if 'obj' in image else 'thar'
        out_path = tmp_path / f'{prefix}_star{int(star_number):02d}_ec.fits'
        _write_multispec_test_fits(out_path, apertures)
        return str(out_path)

    monkeypatch.setattr('reduce_echelle._apall_extract_star', fake_apall_extract_star)
    monkeypatch.setattr(
        'reduce_echelle._step6_quartz_apertures_in_preview_order',
        lambda _quartz, expected_count: (list(range(1, int(expected_count) + 1)), 'db/ap.test'),
    )

    pattern = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        obj_outputs, thar_outputs = extract_all_stars(
            'obj_ff.fits',
            'thar_ff.fits',
            'quartz_ref.fits',
            pattern,
        )
    finally:
        os.chdir(prev_cwd)

    assert _read_apertures_from_apnum_cards(obj_outputs[2]) == [1, 2, 3, 4]
    assert _read_apertures_from_apnum_cards(thar_outputs[2]) == [1, 2, 3, 4]


def test_extract_all_stars_fails_when_apall_extracts_fewer_apertures(monkeypatch, tmp_path):
    def fake_apall_extract_star(image, _quartz, _aperture_string, star_number):
        if int(star_number) == 14:
            apertures = [53, 54, 55] if 'obj' in image else [53, 54, 55, 56]
        else:
            apertures = [1, 2, 3, 4]
        prefix = 'obj' if 'obj' in image else 'thar'
        out_path = tmp_path / f'{prefix}_star{int(star_number):02d}_ec.fits'
        _write_multispec_test_fits(out_path, apertures)
        return str(out_path)

    monkeypatch.setattr('reduce_echelle._apall_extract_star', fake_apall_extract_star)
    monkeypatch.setattr(
        'reduce_echelle._step6_quartz_apertures_in_preview_order',
        lambda _quartz, expected_count: (list(range(1, int(expected_count) + 1)), 'db/ap.test'),
    )

    pattern = np.array([13, 13, 13, 13, 14, 14, 14, 14], dtype=int)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='Step 6 extraction mismatch for star 14'):
            extract_all_stars(
                'obj_ff.fits',
                'thar_ff.fits',
                'quartz_ref.fits',
                pattern,
            )
    finally:
        os.chdir(prev_cwd)


def test_extract_all_stars_preflight_fails_on_dense_vs_quartz_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        'reduce_echelle._step6_quartz_apertures_in_preview_order',
        lambda _quartz, expected_count: ([1, 2, 3, 5], 'db/ap.test'),
    )

    pattern = np.array([1, 1, 1, 1], dtype=int)
    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='Step 6 preflight mismatch'):
            extract_all_stars(
                'obj_ff.fits',
                'thar_ff.fits',
                'quartz_ref.fits',
                pattern,
            )
    finally:
        os.chdir(prev_cwd)


def test_auto_wavelength_propagation_runs_directly_on_real_targets(monkeypatch, tmp_path):
    thar_ref = tmp_path / 'thar_star01_ec.fits'
    thar_tgt = tmp_path / 'thar_star02_ec.fits'
    _write_multispec_test_fits(thar_ref, [1, 2, 3, 4])
    _write_multispec_test_fits(thar_tgt, [1, 2, 3, 4])

    calls = {
        'reidentify': [],
        'review': [],
        'mark': [],
    }

    def fake_reidentify(target, reference, **_kwargs):
        calls['reidentify'].append((target, reference))
        return {'found_frac': 1.0, 'fit_frac': 1.0, 'rms': 0.01}

    def fake_review(target, coordlist):
        calls['review'].append((target, coordlist))

    def fake_mark(target, **_kwargs):
        calls['mark'].append(target)

    monkeypatch.setattr('reduce_echelle._ecreidentify_thar', fake_reidentify)
    monkeypatch.setattr('reduce_echelle._review_reidentified_lines', fake_review)
    monkeypatch.setattr('reduce_echelle.mark_spectrum_as_reference', fake_mark)
    monkeypatch.setattr(
        'reduce_echelle.star_order_by_distance_from_reference',
        lambda _geom, _ref, _stars: [1, 2],
    )

    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = auto_wavelength_propagation(
            crr2_outputs={1: 'obj_star01_ec-crr2.fits', 2: 'obj_star02_ec-crr2.fits'},
            thar_outputs={1: str(thar_ref), 2: str(thar_tgt)},
            ref_star=1,
            coordlist='linelists$thar.dat',
        )
    finally:
        os.chdir(prev_cwd)

    assert calls['reidentify'] == [(str(thar_tgt), str(thar_ref))]
    assert calls['review'] == [(str(thar_tgt), 'linelists$thar.dat')]
    assert str(thar_tgt) in calls['mark']
    assert result['reviewed_thar_outputs'][1] == str(thar_ref)
    assert result['reviewed_thar_outputs'][2] == str(thar_tgt)
    assert list(tmp_path.glob('.tmp_*')) == []


def test_assert_local_apertures_raises_for_global_numbering(tmp_path):
    legacy_path = tmp_path / 'legacy_star02_ec.fits'
    _write_multispec_test_fits(legacy_path, [5, 6, 7, 8])

    with pytest.raises(RuntimeError, match='Re-run step 6'):
        _assert_local_apertures(str(legacy_path), expected_count=4)
