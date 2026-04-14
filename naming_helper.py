import os


def fits_stem(path_or_name):
    """Return basename without .fits extension."""
    text = str(path_or_name)
    base = os.path.basename(text)
    if base.lower().endswith('.fits'):
        return base[:-5]
    return os.path.splitext(base)[0]


def with_suffix(path_or_name, suffix):
    """Return <stem><suffix>.fits preserving current filename style."""
    return f"{fits_stem(path_or_name)}{suffix}.fits"


def with_suffix_stem(path_or_name, suffix):
    """Return <stem><suffix> without extension."""
    return f"{fits_stem(path_or_name)}{suffix}"


def step2_scattered(path_or_name):
    return with_suffix(path_or_name, '-sl')


def step3_median(path_or_name):
    return with_suffix(path_or_name, '_med')


def step3_master_flat(path_or_name):
    return with_suffix(path_or_name, '_nflat')


def step4_flat_corrected(path_or_name):
    return with_suffix(path_or_name, '-F')


def step6_star_extract(path_or_name, star_number):
    return with_suffix(path_or_name, f'_star{int(star_number):02d}_ec')


def step7_crr2(path_or_name):
    return with_suffix(path_or_name, '-crr2')


def step11_dispcor(path_or_name):
    return with_suffix(path_or_name, '-dc')


def normalize_step2_like_input(path):
    """Normalize to deterministic step-2 '*-sl.fits' artifact path."""
    if not path:
        return None
    raw = str(path)
    lower = raw.lower()
    if lower.endswith('-sl.fits'):
        return raw
    if lower.endswith('-sl-f.fits'):
        return raw[:-7] + '.fits'
    return step2_scattered(raw)
