from pyraf import iraf


def load_ccdred():
    """
    Load IRAF packages needed for zerocombine/ccdproc.
    """
    iraf.noao()
    iraf.imred()
    iraf.ccdred()


def load_crutil():
    """Load IRAF packages needed for crmedian and other crutil tasks."""
    iraf.noao()
    iraf.imred()
    iraf.crutil()


def load_images():
    """Load IRAF images package (needed before imtranspose, imjoin, etc.)."""
    iraf.images()


def load_imutil():
    """Load IRAF packages needed for hedit and other imutil tasks."""
    iraf.noao()
    iraf.images()
    iraf.imutil()


def hedit_ccdsec(filename, value='[1:2048,1:2056]'):
    """Update the CCDSEC keyword in a FITS header via IRAF hedit.

    verify=no and show=yes: updates are applied automatically without prompting
    but each edit is still printed so the user can see what changed.
    """
    iraf.hedit(
        images=filename,
        fields='CCDSEC',
        value=value,
        add=iraf.no,
        addonly=iraf.no,
        delete=iraf.no,
        verify=iraf.no,
        show=iraf.yes,
        update=iraf.yes,
        mode='ql',
    )


def run_ccdproc_ovefit_trim(images, output):
    iraf.ccdred.ccdproc.unlearn()
    # Use iraf.yes / iraf.no to mirror IRAF boolean semantics explicitly.
    # (PyRAF can coerce types in many cases, but this is the least ambiguous.)
    iraf.ccdred.ccdproc(
        images='@'+images,
        output='@'+output,
        ccdtype="",
        max_cache=0,
        noproc=iraf.no,

        fixpix=iraf.no,
        overscan=iraf.yes,
        trim=iraf.yes,
        zerocor=iraf.no,
        darkcor=iraf.no,
        flatcor=iraf.no,
        illumcor=iraf.no,
        fringecor=iraf.no,
        readcor=iraf.no,
        scancor=iraf.no,

        readaxis="line",
        fixfile="",
        biassec="image",
        trimsec="image",
        zero="",
        dark="",
        flat="",
        illum="",
        fringe="",
        minreplace=1,
        scantype="shortscan",
        nscan=1,

        interactive=iraf.no,
        function="chebyshev",
        order=3,
        sample="*",
        naverage=1,
        niterate=1,
        low_reject=3,
        high_reject=3,
        grow=0.0,
        mode="ql",
    )

def run_ccdproc_flat_corr(images, output, flat):
    iraf.ccdred.ccdproc.unlearn()
    # Use iraf.yes / iraf.no to mirror IRAF boolean semantics explicitly.
    # (PyRAF can coerce types in many cases, but this is the least ambiguous.)
    iraf.ccdred.ccdproc(
        images='@'+images,
        output='@'+output,
        ccdtype="",
        max_cache=0,
        noproc=iraf.no,

        fixpix=iraf.no,
        overscan=iraf.no,
        trim=iraf.no,
        zerocor=iraf.no,
        darkcor=iraf.no,
        flatcor=iraf.yes,
        illumcor=iraf.no,
        fringecor=iraf.no,
        readcor=iraf.no,
        scancor=iraf.no,

        readaxis="line",
        fixfile="",
        biassec="image",
        trimsec="image",
        zero="",
        dark="",
        flat=flat,
        illum="",
        fringe="",
        minreplace=1,
        scantype="shortscan",
        nscan=1,

        interactive=iraf.no,
        function="chebyshev",
        order=3,
        sample="*",
        naverage=1,
        niterate=1,
        low_reject=3,
        high_reject=3,
        grow=0.0,
        mode="ql",
    )
    
def run_ccdproc_subtract_dark(images, output, dark_image):
    iraf.ccdred.ccdproc.unlearn()
    # Use iraf.yes / iraf.no to mirror IRAF boolean semantics explicitly.
    # (PyRAF can coerce types in many cases, but this is the least ambiguous.)
    iraf.ccdred.ccdproc(
        images='@'+images,
        output='@'+output,
        ccdtype="",
        max_cache=0,
        noproc=iraf.no,

        fixpix=iraf.no,
        overscan=iraf.no,
        trim=iraf.no,
        zerocor=iraf.no,
        darkcor=iraf.yes,
        flatcor=iraf.no,
        illumcor=iraf.no,
        fringecor=iraf.no,
        readcor=iraf.no,
        scancor=iraf.no,

        readaxis="line",
        fixfile="",
        biassec="image",
        trimsec="image",
        zero="",
        dark=dark_image,
        flat="",
        illum="",
        fringe="",
        minreplace=1,
        scantype="shortscan",
        nscan=1,

        interactive=iraf.no,
        function="chebyshev",
        order=3,
        sample="*",
        naverage=1,
        niterate=1,
        low_reject=3,
        high_reject=3,
        grow=0.0,
        mode="ql",
    )
    
def run_zerocombine_masterbias(image_list_path, out_path):
    iraf.ccdred.zerocombine.unlearn()
    iraf.ccdred.zerocombine(
    input = f'@{image_list_path}',
    output = f'{out_path}',
    combine = "median",
    reject = "minmax",
    ccdtype = " ",
    process = iraf.no,
    delete = iraf.no,
    clobber = iraf.no,
    scale = "none",
    statsec = "",
    nlow = 0,
    nhigh = 1,
    nkeep = 1,
    mclip = iraf.yes,
    lsigma = 3,
    hsigma = 3,
    rdnoise = "ENOISE",
    gain = "EGAIN",
    snoise = 0,
    pclip = -0.5,
    blank = 0,
    mode = "ql",
    )


def run_ccdproc_bias_corr(images, output, zero_image='zeror4.fits', dark_image='dark.fits'):
    iraf.ccdred.ccdproc.unlearn()
    iraf.ccdred.ccdproc(
        images='@'+images,
        output='@'+output,
        ccdtype="",
        max_cache=0,
        noproc=iraf.no,

        fixpix=iraf.no,
        overscan=iraf.no,
        trim=iraf.no,
        zerocor=iraf.yes,
        darkcor=iraf.no,
        flatcor=iraf.no,
        illumcor=iraf.no,
        fringecor=iraf.no,
        readcor=iraf.no,
        scancor=iraf.no,
        readaxis="line",
        fixfile="",
        biassec="image",
        trimsec="image",
        zero=zero_image,
        dark=dark_image,
        flat="",
        illum="",
        fringe="",
        minreplace=1,
        scantype="shortscan",
        nscan=1,
        interactive=iraf.no,
        function="chebyshev",
        order=3,
        sample="*",
        naverage=1,
        niterate=1,
        low_reject=3,
        high_reject=3,
        grow=0.0,
        mode="ql",
    )
    
def assemble_mosaic(images, output_name):
    """
    Assemble 4 OPAMP images into a single mosaic.
    
    images: Table or list of 4 images sorted by OPAMP (1-4)
    output_name: filename for the final combined mosaic

    Only the imcopy line (quadrant 4 copy) is printed; all imtranspose and
    imjoin output is suppressed via Stdout capture and verbose=iraf.no.
    """
    # Sort by OPAMP to ensure correct order
    images_sorted = images.copy()
    images_sorted.sort('OPAMP')
    
    # Extract filenames for each quadrant
    im1 = str(images_sorted[0]['filename_input'])
    im2 = str(images_sorted[1]['filename_input'])
    im3 = str(images_sorted[2]['filename_input'])
    im4 = str(images_sorted[3]['filename_input'])
    
    # Create temporary filenames for intermediate steps
    temp1  = 'temp_c1.fits'
    temp2  = 'temp_c2.fits'
    temp3  = 'temp_c3.fits'
    temp4  = 'temp_c4.fits'
    comb12 = 'temp_comb12.fits'
    comb43 = 'temp_comb43.fits'
    
    # Transform quadrant 1: transpose twice with horizontal flip
    iraf.imtranspose(im1, temp1, len_blk=512, Stdout=1)
    iraf.imtranspose(f"{temp1}[-*,*]", temp1, len_blk=512, Stdout=1)
    
    # Transform quadrant 2: horizontal flip, transpose twice
    iraf.imtranspose(f"{im2}[-*,*]", temp2, len_blk=512, Stdout=1)
    iraf.imtranspose(f"{temp2}[-*,*]", temp2, len_blk=512, Stdout=1)
    
    # Transform quadrant 3: transpose, vertical flip and transpose
    iraf.imtranspose(im3, temp3, len_blk=512, Stdout=1)
    iraf.imtranspose(f"{temp3}[*,-*]", temp3, len_blk=512, Stdout=1)
    
    # Quadrant 4: copy as-is — this line is intentionally printed
    iraf.imcopy(im4, temp4)
    
    # Join quadrants horizontally: c1+c2 and c4+c3
    iraf.imjoin(f"{temp1},{temp2}", comb12, 1, pixtype="", verbose=iraf.no)
    iraf.imjoin(f"{temp4},{temp3}", comb43, 1, pixtype="", verbose=iraf.no)
    
    # Join the two rows vertically
    iraf.imjoin(f"{comb43},{comb12}", output_name, 2, pixtype="", verbose=iraf.no)
    
    iraf.imdelete(f"{temp1},{temp2},{temp3},{temp4},{comb12},{comb43}", verify=iraf.no)
    
    return None

def stack_science_images(inpath_list, output_path, mode="median", scale="none"):
    iraf.immatch.imcombine.unlearn()
    iraf.immatch.imcombine(
        input   = '@'+inpath_list,# List of images to combine
        output  = output_path,# List of output images
        headers = "",# List of header files (optional)
        bpmasks = "",# List of bad pixel masks (optional)
        rejmask = "",# List of rejection masks (optional)
        nrejmas = "",# List of number rejected masks (optional)
        expmask = "",# List of exposure masks (optional)
        sigmas  = "",# List of sigma images (optional)
        imcmb   = "$I",# Keyword for IMCMB keywords
        logfile = "STDOUT",# Log file
        combine = mode,# Type of combine operation
        reject  = "none",# Type of rejection
        project = iraf.no,# Project highest dimension of input images?
        outtype = "real",# Output image pixel datatype
        outlimi = "",# Output limits (x1 x2 y1 y2 ...)
        offsets = "none",# Input image offsets
        masktyp = "none",# Mask type
        maskval = 0,# Mask value
        blank   = 0.,# Value if there are no pixels
        scale   = scale,# Image scaling
        zero    = "none",# Image zero point offset
        weight  = "none",
        statsec = "",
        expname = "",
        lthresh = "INDEF",
        hthresh = "INDEF",
        nlow    = 1,
        nhigh   = 1,
        nkeep   = 1,
        mclip   = iraf.yes,
        lsigma  = 3.,
        hsigma  = 3.,
        rdnoise = "ENOISE",
        gain    = "EGAIN",
        snoise  = 0.,
        sigscal = 0.1,
        pclip   = -0.5,
    )
    return None

def median_crr_single(input_image, output_image):
    '''Run crmedian on a single image. crmedian only accepts one image at a time (type f, not *f).'''
    iraf.noao.imred.crutil.crmedian.unlearn()
    iraf.noao.imred.crutil.crmedian(
        input   = input_image,
        output  = output_image,
        crmask  = "",
        median  = "",
        sigma   = "",
        residua = "",
        var0    =  0.,
        var1    =  0.,
        var2    =  0.,
        lsigma  = 10.,
        hsigma  =  3.,
        ncmed   =   5,
        nlmed   =   5,
        ncsig   =  25,
        nlsig   =  25,
        mode    =  "ql"
    )
    return None


def do_apflatten(inlist_path, outlist_path):
    iraf.noao.imred.echelle.apflatten.unlearn()
    iraf.noao.imred.echelle.apflatten(
        input   = '@'+inlist_path,  # List of images to flatten
        output  = '@'+outlist_path, # List of output flatten images
        apertur = "",               # Apertures
        referen = "",               # List of reference images
        interac = iraf.yes,         # Run task interactively?
        find    = iraf.no,         # Find apertures?
        recente = iraf.yes,         # Recenter apertures?
        resize  = iraf.yes,         # Resize apertures?
        edit    = iraf.yes,         # Edit apertures?
        trace   = iraf.no,         # Trace apertures?
        fittrac = iraf.no,         # Fit traced points interactively?
        flatten = iraf.yes,         # Flatten spectra?
        fitspec = iraf.yes,         # Fit normalization spectra interactively?
        line    = "INDEF",          # Dispersion line
        nsum    = 10,               # Number of dispersion lines to sum or median
        thresho = 10,               # Threshold for flattening spectra
        pfit    = "fit2d",          # Profile fitting type (fit1d|fit2d)
        clean   = iraf.no,          # Detect and replace bad pixels?
        saturat = "INDEF",          # Saturation level
        readnoi = "ENOISE",                # Read out noise sigma (photons)
        gain    = "EGAIN",                # Photon gain (photons/data number)
        lsigma  = 4,                # Lower rejection threshold
        usigma  = 4,                # Upper rejection threshold
        functio = "chebyshev",       # Fitting function for normalization spectra
        order   = 1,                # Fitting function order
        sample  = "*",              # Sample regions
        naverag = 1,                # Average or median
        niterat = 0,                # Number of rejection iterations
        low_rej = 3,                # Lower rejection sigma
        high_re = 3,                # High upper rejection sigma
        grow    = 0,                # Rejection growing radius
        mode    = "ql"
    )