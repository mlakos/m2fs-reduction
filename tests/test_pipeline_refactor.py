import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from file_handler import (
    classify_image_type,
    construct_table_of_images,
    normalize_header_value,
)
from image_processing import (
    resolve_processing_dirs as resolve_processing_dirs_image,
    step2_overscan_trim,
    step9_stack_science,
)
from reduce_echelle import (
    resolve_processing_dirs as resolve_processing_dirs_echelle,
    write_infiles_from_directory,
)


def _write_test_fits(path, **header_items):
    hdr = fits.Header()
    for key, value in header_items.items():
        hdr[key] = value
    hdr.setdefault('FILENAME', path.stem)
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(path)


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
