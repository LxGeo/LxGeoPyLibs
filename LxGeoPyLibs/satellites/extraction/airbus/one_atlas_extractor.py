# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:25:19 2020

@author: Cherif
"""

import logging
import click
from zipfile import ZipFile 
import shutil
import os
from pathlib import Path
from osgeo_utils import gdal_merge as gm
from LxGeoPyLibs.satellites.imd import IMetaData
from LxGeoPyLibs.satellites.transformation.pansharpen import pansharpen_image
from LxGeoPyLibs import configure_logging, _logger

class ZipExtractionError(BaseException):
    pass
class PartsMergeError(BaseException):
    pass
class PansharpeningError(BaseException):
    pass

def extract_single_zip(zip_file_path, extract_path):
    """
    Given a zip file extract required files in the defined extract path
    Required files: PAN & MS tif files, xml metadata file
    This function is based on airbus products zipfiles.

    Return dict of output file paths for panchromatic, multispectral and xml file
    """

    RAW_DATA_FOLDER = Path(extract_path) / "dezipped"
    if not RAW_DATA_FOLDER.is_dir():
        os.makedirs(RAW_DATA_FOLDER)
    

    info_dict = {
        "RAW_DATA_FOLDER":RAW_DATA_FOLDER
    }


    TIF_files = []
    xml_file = None
    opened_zipfile = ZipFile(zip_file_path, 'r')
    # Selecting files
    filenames = opened_zipfile.namelist()
    for filename in filenames:
        if Path(filename).suffix.lower() ==".tif":
            TIF_files.append(filename)
        elif not xml_file and ("/dim" in str(filename).lower()) and (str(filename).lower().endswith("xml")):
            xml_file =filename
    
    # Extracting metdata xml
    if xml_file:
        output_xml_path = RAW_DATA_FOLDER / f"imd{Path(xml_file).suffix}"
        with open(str(output_xml_path.resolve()), 'wb') as out_f:
            out_f.write(opened_zipfile.read(str(xml_file)))
        out_imd_file = os.path.join(RAW_DATA_FOLDER, "ortho.imd")
        info_dict["XML"]=output_xml_path
        with open( out_imd_file, "w" ) as imd_file:
            iMetaData = IMetaData( str(output_xml_path) )
            imd_file.write(iMetaData.IMD_geo())
    else:
        _logger.warning("Cannot find xml metadata in zipfile!")

    # Parts count check
    pan_parts_count=len( list(filter(lambda x : "_P_" in Path(x).stem, TIF_files) ))
    pan_parts_paths=[]
    ms_parts_count=len( list(filter(lambda x : "_MS_" in Path(x).stem, TIF_files) ))
    ms_parts_paths=[]
    pms_parts_count =  len( list(filter(lambda x : "_PMS_" in Path(x).stem, TIF_files) ))
    pms_parts_paths=[]
    
    # Extracting files tifs    
    for TIF_file in TIF_files:
        tif_base_name = Path(TIF_file).stem
        tif_part_name = tif_base_name.split("_")[-1]
        if "_P_" in tif_base_name:
            output_path = RAW_DATA_FOLDER / f"_P_{tif_part_name}.tif"
            pan_parts_paths.append(output_path)
        elif "_MS_" in tif_base_name:
            output_path = RAW_DATA_FOLDER / f"_MS_{tif_part_name}.tif"
            ms_parts_paths.append(output_path)
        elif "_PMS_" in tif_base_name:
            output_path = RAW_DATA_FOLDER / f"_PMS_{tif_part_name}.tif"
            pms_parts_paths.append(output_path)
        else:
            raise Exception("Unkown tif file file representation!")
        with open(output_path, 'wb') as out_f:
            out_f.write(opened_zipfile.read(str(TIF_file)))
        
    
    ## Merging tif files parts
    if pan_parts_count>0:
        assert len(pan_parts_paths) == pan_parts_count
        p_merged_output = RAW_DATA_FOLDER / "_P_.tif"
        pan_parts_paths = list(map(lambda x:str(x) ,set(pan_parts_paths)))
        if pan_parts_count > 0:
            gm.main(['', '-o', str(p_merged_output)] + pan_parts_paths )
            info_dict["P"]=p_merged_output
            [os.remove(file) for file in pan_parts_paths]
        else:
            err_msg = "Wrong number of pan tif files!"
            _logger.error(err_msg)
            raise PartsMergeError(err_msg)
    
    if ms_parts_count>0:
        assert len(ms_parts_paths) == ms_parts_count
        ms_merged_output = RAW_DATA_FOLDER / "_MS_.tif"
        ms_parts_paths = list(map(lambda x:str(x) ,set(ms_parts_paths)))
        if ms_parts_count > 0:
            gm.main(['', '-o', str(ms_merged_output)] + ms_parts_paths )
            info_dict["MS"]=ms_merged_output
            [os.remove(file) for file in ms_parts_paths]
        else:
            err_msg = "Wrong number of ms tif files!"
            _logger.error(err_msg)
            raise PartsMergeError(err_msg)

    if pms_parts_count>0:
        assert len(pms_parts_paths) == pms_parts_count
        pms_merged_output = RAW_DATA_FOLDER / "_PMS_.tif"
        pms_parts_paths = list(map(lambda x:str(x) ,set(pms_parts_paths)))
        if pms_parts_count > 0:
            gm.main(['', '-o', str(pms_merged_output)] + pms_parts_paths )
            info_dict["PMS"]=pms_merged_output
            [os.remove(file) for file in pms_parts_paths]
        else:
            err_msg = "Wrong number of pms tif files!"
            _logger.error(err_msg)
            raise PartsMergeError(err_msg)

    return info_dict


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path',type=click.Path(), help="Output extraction folder")
@click.option('-dd', '--delete_dezipped', is_flag=True, help="Delete extracted panchromatic and multispectral images and xml data")
@click.option('-v', '--verbose', is_flag=True, help="Verbosity flag for debug level record printing")
def main(input_path, output_path, delete_dezipped, verbose):
    """ input_path: Path to zipfile or folder containing zipfiles """
    
    input_path=Path(input_path)
    output_path=Path(output_path)

    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    configure_logging(log_level=log_level)
    
    is_zip_file_predicate = lambda x:str(x).lower().endswith(".zip")
    if input_path.is_dir():
        c_zipfile_paths = list(filter(is_zip_file_predicate,
        map(lambda x: input_path / x, os.listdir(input_path))
        ))
        out_zipfile_paths = list(map(lambda x: output_path/x.stem, c_zipfile_paths))
    else:
        c_zipfile_paths = [input_path]
        out_zipfile_paths = [output_path]
    
    for in_file, outdir in zip(c_zipfile_paths, out_zipfile_paths):
        
        try:
            ext_dict=extract_single_zip(in_file, outdir)
        except Exception as e:
            _logger.error(f"Failed extracting of zipfile at: {in_file}")
            _logger.error(str(e))
            shutil.rmtree(outdir)
            continue
        
        if not ext_dict.get("PMS"):
            try:            
                # pansharpen image
                out_pansharpened_path = os.path.join(outdir, "ortho.tif")
                exec_code = pansharpen_image(str(ext_dict["P"]), str(ext_dict["MS"]), out_pansharpened_path)
                if exec_code == 1:
                    err_msg = f"Failed pansharpening to {out_pansharpened_path}"
                    raise PansharpeningError(err_msg)
                
                # pansharpen deletes all files with same basename as its output
                
            except Exception as e:
                _logger.error(f"Failed postprocessing of extracted zipfile at: {outdir}")
                _logger.error(str(e))
                continue

        if delete_dezipped:
            try:
                shutil.rmtree(ext_dict["RAW_DATA_FOLDER"])
            except Exception as e:
                _logger.error("Error deleting dezipped folder!")
                _logger.error(str(e))
    


if __name__ == "__main__":
    main()
    