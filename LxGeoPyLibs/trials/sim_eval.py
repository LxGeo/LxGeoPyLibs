from LxGeoPyLibs.trials.similarity import map_similarities
from LxGeoPyLibs.geometry.metrics.distances import polygon_distances
import os, json, yaml
import unsync

fix_path = lambda x: x.replace("//got/mcherif_home", "/home/mcherif")
distances_set = set([polygon_distances.chamfer_distance, polygon_distances.hausdorff_distance, polygon_distances.polis_distance])

@unsync.unsync(cpu_bound=True)
def dataset_task(dataset_path):
    print("Running for :"+dataset_path)
    cmd_file_p = os.path.join(dataset_path, "commands.txt")
    #return os.path.isfile(cmd_file_p)

    with open(cmd_file_p) as f:
        conf = yaml.load(f, yaml.Loader)

    ref_data_path = fix_path(conf["ref"])

    maps_sims = {}

    for tar_name, tar_obj in conf.items():
        if "ref" in tar_name: continue

        c_tar_misalgined_path = fix_path(tar_obj["in_path"])
        c_tar_misalgined_sim = map_similarities(ref_data_path, c_tar_misalgined_path, distances_set)
        
        algined_sims={}
        for approach_name, aligned_path in tar_obj["out_paths"].items():
            aligned_path = os.path.dirname(fix_path(aligned_path))
            algined_sim = map_similarities(ref_data_path, aligned_path, distances_set)
            algined_sims[approach_name]=algined_sim

        maps_sims[tar_name] = {"misaligned_sims":c_tar_misalgined_sim, "algined_sims":algined_sims}
    
    print(maps_sims)
    with open(os.path.join(dataset_path, "eval.txt"), "w") as f:
        f.write(str(maps_sims))    
    return maps_sims
        
    





if __name__ == "__main__":

    datasets_path = [
        "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/Brazil/belford_east/alignment",
        "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/BRAGANCA/alignment",
        "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/Funchal/alignment",
        "/home/mcherif/Documents/DATA_SANDBOX/Alignment_Project/DATA_SANDBOX/paris_tristereo_dhm_plus_external/alignment",
    ]
    
    """for dst_path in datasets_path:
        try:
            c_map_sims = dataset_task(dst_path)
        except Exception as e:
            #print("error running for "+dst_path)
            print(e)"""
    
    tasks = [dataset_task(dst_path) for dst_path in datasets_path ]
    coverage_result = [task.result() for task in tasks]
    