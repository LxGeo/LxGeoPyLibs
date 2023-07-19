
import os
import subprocess
import json
from LxGeoPyLibs.satellites.transformation.epipolar_gen import main as epipolar_cmd
from LxGeoPyLibs.satellites.transformation.disparity_to_dsm import main as disparity_cmd
from LxGeoPyLibs.trials.assing_height_perfectGt import run

import click
def call_click_command(cmd, *args, **kwargs):
    # Get positional arguments from args
    arg_values = {c.name: a for a, c in zip(args, cmd.params)}
    args_needed = {c.name: c for c in cmd.params
                   if c.name not in arg_values}

    # build and check opts list from kwargs
    opts = {a.name: a for a in cmd.params if isinstance(a, click.Option)}
    for name in kwargs:
        if name in opts:
            arg_values[name] = kwargs[name]
        else:
            if name in args_needed:
                arg_values[name] = kwargs[name]
                del args_needed[name]
            else:
                raise click.BadParameter(
                    "Unknown keyword argument '{}'".format(name))
    # check positional arguments list
    for arg in (a for a in cmd.params if isinstance(a, click.Argument)):
        if arg.name not in arg_values:
            raise click.BadParameter("Missing required positional"
                                     "parameter '{}'".format(arg.name))
    # build parameter lists
    opts_list = sum(
        [[o.opts[0], str(arg_values[n])] for n, o in opts.items()], [])
    args_list = [str(v) for n, v in arg_values.items() if n not in opts]

    # call the command
    #print( " ".join(opts_list))
    cmd(opts_list + args_list, standalone_mode=False)

def get_full_path_from_id(conf_path, file_id):
    with open(conf_path) as f:
        conf_content = json.load(f)    
    return os.path.join(os.path.dirname(conf_path), conf_content[file_id])

gpi = get_full_path_from_id

def single_run(conf_file, top_out_dir, v1tov2):

    with open(conf_file) as f:
        conf_dict = json.load(f)
    out_dir = os.path.join(top_out_dir, conf_dict["test_name"] + ("_v1tov2" if v1tov2 else "_v2tov1"))
    os.makedirs(out_dir, exist_ok=True)

    ### loading views conf
    v1_conf_path = os.path.join(conf_dict["top_dir"], conf_dict["views"]["v1"])
    with open(v1_conf_path) as f:
        v1_conf = json.load(f)
    v2_conf_path = os.path.join(conf_dict["top_dir"], conf_dict["views"]["v2"])
    with open(v2_conf_path) as f:
        v2_conf = json.load(f)
    
    if v1tov2:
        ref_conf = v2_conf_path
        tar_conf = v1_conf_path
    else:
        ref_conf = v1_conf_path
        tar_conf = v2_conf_path
    
    ####
    epi1_path = os.path.join(out_dir, "epi1.tif")
    epi2_path = os.path.join(out_dir, "epi2.tif")
    cf_path = os.path.join(out_dir, "couple.json")
    disp_path  = os.path.join(out_dir, "disp.tif")
    dhm_path  = os.path.join(out_dir, "dhm.tif")
    ####

    """call_click_command(epipolar_cmd, **{
        "input_alternative_image_1":gpi(ref_conf, "ortho"),
        "input_alternative_image_2":gpi(tar_conf, "ortho"),
        "input_reference_image_1":gpi(ref_conf, "ortho"),
        "input_imd_1":gpi(ref_conf, "imd"),
        "input_reference_image_2":gpi(tar_conf, "ortho"),
        "input_imd_2":gpi(tar_conf, "imd"),
        "output_epi_image_1":epi1_path,
        "output_epi_image_2":epi2_path,
        "couple_file":cf_path
        })
    return"""
    
    
    b_vector1 = gpi(ref_conf, "buildings_vector")
    b_vector2 = gpi(tar_conf, "buildings_vector")
    imd1 = gpi(ref_conf, "imd")
    imd2 = gpi(tar_conf, "imd")
    b_vector_aligned = os.path.join(out_dir, "aligned.shp")
    print(f"Output path: {b_vector_aligned}")
    command8_to_run = f"C:/SANDBOX/LxGeo/lxProximityAlign/build/out/Release/lxProximityAlign.exe --ishp {b_vector1} --rshp {b_vector2} -o {b_vector_aligned} --i_imd {imd1} --r_imd {imd2} --overwrite_output --max_disparity 100 --ndbv 0.001 1 10 20 40 100"
    print(command8_to_run)
    #os.system(command8_to_run)
    subprocess.Popen(command8_to_run.split(" "))

    """call_click_command(disparity_cmd, **{
        "input_disparity":disp_path,
        "output_path":dhm_path,
        "couple_file":cf_path,
        "scale_factor":1/10
    })"""

    """disparity_cmd( [
        "--input_disparity", disp_path, "--output_path", dhm_path, "--couple_file", cf_path
    ])"""

    """vector1_out = gpi(ref_conf, "buildings_vector"); vector1_out = os.path.join(os.path.dirname(vector1_out), "h_"+os.path.basename(vector1_out))
    vector2_out = gpi(tar_conf, "buildings_vector"); vector2_out = os.path.join(os.path.dirname(vector2_out), "h_"+os.path.basename(vector2_out))
    run(gpi(ref_conf, "buildings_vector"), gpi(ref_conf, "imd"), gpi(tar_conf, "buildings_vector"), gpi(tar_conf, "imd"), vector1_out, vector2_out)
    """
    


if __name__ == "__main__":
    top_conf="C:/DATA_SANDBOX/Alignment_Project/alignment_results/data_configs"
    configs_paths = os.listdir(top_conf)
    top_out_dir = "C:/DATA_SANDBOX/Alignment_Project/alignment_results/lxProximityAlign1D/"
    for conf in configs_paths:
        #if ("gt" not in conf.lower()):
        #    continue
        conf=os.path.join(top_conf, conf)
        single_run(conf, top_out_dir, True)
        single_run(conf, top_out_dir, False)
    a=0