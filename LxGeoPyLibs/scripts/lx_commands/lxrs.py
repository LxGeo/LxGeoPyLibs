import os
config_path = os.path.join(os.path.expanduser('~'), 'config.json')

def cmd_lxrs(in_image, opt_dict):
    """
    Run lxRemoteSegmentation on system.
    Returns command execution success
    """
    command_template = f"lxRemoteSegmentation.py -i {in_image} -s {config_path}"
    for k,v in opt_dict.items():
        command_template+= f" {k} {v}"
    
    response = os.system(command_template)
    if response:
        print("lxRemoteSegmentation execution error!")
    return response
