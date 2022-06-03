# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:25:19 2020

@author: Cherif
"""

import argparse
import click
from zipfile import ZipFile 
import os
import shutil
from LxGeoPyLibs.vendor.gdal import gdal_merge as gm

def extract_single_zip(zip_file_path, extract_path):
    """
    Given a zip file extract required files in the defined extract path
    Required files: PAN & MS tif files, xml metadata file
    This function is based on airbus products zipfiles.

    Return dict of output file paths for panchromatic, multispectral and xml file
    """
    
    TIF_files = []
    xml_file = []
    opened_zipfile = ZipFile(zip_file_path, 'r')
    # Selecting files
    filenames = opened_zipfile.namelist()
    for filename in filenames:
        lower_filename = filename.lower()
        if lower_filename.endswith(".tif"):
            TIF_files.append(filename)
        elif ("/dim" in lower_filename) and (lower_filename.endswith("xml")):
            xml_file =filename
    
    # Parts count check
    pan_parts_count=len( list(filter(lambda x : "_P_" in os.path.basename(x), TIF_files) ))
    pan_parts_paths=[]
    ms_parts_count=len( list(filter(lambda x : "_MS_" in os.path.basename(x), TIF_files) ))
    ms_parts_paths=[]
    
    # Extracting files tifs
    RAW_DATA_FOLDER = os.path.join(extract_path, "orthos")
    if not os.path.isdir(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)
    
    for TIF_file in TIF_files:
        tif_base_name = os.path.basename(TIF_file)
        tif_part_name = tif_base_name.split("_")[-1]
        output_path = os.path.join(RAW_DATA_FOLDER, "_P_{}.tif".format(tif_part_name) if "_P_" in tif_base_name else "_MS_{}.tif".format(tif_part_name))
        with open(output_path, 'wb') as out_f:
            out_f.write(opened_zipfile.read(TIF_file))
        pan_parts_paths.append(output_path) if "_P_" in os.path.basename(output_path) else ms_parts_paths.append(output_path)
    
    ## Merging tif files parts
    assert len(pan_parts_paths) == pan_parts_count
    p_merged_output = os.path.join(RAW_DATA_FOLDER, "_P_.tif")
    pan_parts_paths = list(set(pan_parts_paths))
    if pan_parts_count > 0:
        gm.main(['', '-o', p_merged_output] + pan_parts_paths )
        [os.remove(file) for file in pan_parts_paths]
    else:
        print("Wrong number of pan tif files!")
        raise    
    
    assert len(ms_parts_paths) == ms_parts_count
    ms_merged_output = os.path.join(RAW_DATA_FOLDER, "_MS_.tif")
    ms_parts_paths = list(set(ms_parts_paths))
    if pan_parts_count > 0:
        gm.main(['', '-o', ms_merged_output] + ms_parts_paths )
        [os.remove(file) for file in ms_parts_paths]
    else:
        raise("Wrong number of pan tif files!")    
    
    # Extracting files xml
    output_xml_path = os.path.join(RAW_DATA_FOLDER, os.path.basename(xml_file))
    with open(output_xml_path, 'wb') as out_f:
        out_f.write(opened_zipfile.read(xml_file))
    
    return {
        "P":p_merged_output,
        "MS":ms_merged_output,
        "XML":output_xml_path
    }


@click.command()
@click.argument('zipfile_path', type=click.Path(exists=True))
@click.option('--extract_path',type=click.Path(), help="Output extraction folder")
@click.option('--out_image', type=click.Path(), help="Path to output pansharpened image")
@click.option('--out_imd', type=click.Path(), help="Path to output imd file")
@click.option('--keep_raw_data', default=False, help="Keep extracted panchromatic and multispectral images and xml data")
def main(zipfile_path, extract_path, out_image, out_imd, keep_raw_data):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("zipfile_path",
                        help="Path to downlaoded airbus zipfile")
    parser.add_argument("extract_path",
                        help="path of extraction")
    
    args = parser.parse_args()
    extract_single_zip(args.zipfile_path, args.extract_path)
    