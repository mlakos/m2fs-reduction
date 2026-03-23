from astropy.table import Table
from astropy.io import fits

from datetime import datetime, timedelta

import numpy as np
import os
from os.path import isfile, join
import sys

def construct_table_of_images(path_to_list, extra_columns=[]):
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
    table_out['filename_input'] = table_out['filename_input'].astype('U32')
    
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
        outlist_path = lpath[0]+lpath[1]+'-'+suffix+lpath[2]
        with open(outlist_path, 'w') as outlist:
            for impath in filenames:
                outpath = split_path(impath.strip())
                outlist.write(outpath[0]+outpath[1]+'-'+suffix+outpath[2]+'\n')
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
