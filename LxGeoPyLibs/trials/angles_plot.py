import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from LxGeoPyLibs.satellites.formulas import base_to_height_ratio, compute_rotation_angle
import math

if __name__ == "":

    # Define initial parameters
    sat_az1 = np.radians(60)
    sat_az2 = np.radians(0)
    sat_el1_deg = 30
    sat_el2_deg = 30

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    p1 = ax.scatter(sat_az1, sat_el1_deg)
    p2 = ax.scatter(sat_az2, sat_el2_deg)
    ax.set_rmax(90)

    ax_az1 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    ax_az2 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    az1_slider = Slider(
        ax=ax_az1,
        label='Sat1 azimuth',
        valmin=0,
        valmax=360,
        valinit=sat_az1,
    )
    az2_slider = Slider(
        ax=ax_az2,
        label='Sat2 azimuth',
        valmin=0,
        valmax=360,
        valinit=sat_az2,
    )

    ax_el1 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    ax_el2 = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    el1_slider = Slider(
        ax=ax_el1,
        label='Sat1 eleveation',
        valmin=0,
        valmax=90,
        valinit=sat_el1_deg,
        valstep=1,
        orientation="vertical"
    )
    el2_slider = Slider(
        ax=ax_el2,
        label='Sat2 eleveation',
        valmin=0,
        valmax=90,
        valinit=sat_el2_deg,
        valstep=1,
        orientation="vertical"
    )

    axbox = fig.add_axes([0.1, 0.9, 0.3, 0.075])
    text_box = TextBox(axbox, "", textalignment="center")

    def update(val):
        p1.set_offsets((np.radians(az1_slider.val),el1_slider.val))
        p2.set_offsets((np.radians(az2_slider.val), el2_slider.val))
        b_h_value = base_to_height_ratio(np.radians(az1_slider.val), np.radians(el1_slider.val),np.radians(az2_slider.val), np.radians(el2_slider.val) )
        
        rotation_angle = math.degrees(compute_rotation_angle(
            math.radians(az1_slider.val),
            el1_slider.val,
            math.radians(az2_slider.val),
            el2_slider.val)
            )
        
        text_box.set_val(f" Rotation angle: {rotation_angle} || B\H: {b_h_value}")
        fig.canvas.draw_idle()

    az1_slider.on_changed(update)
    az2_slider.on_changed(update)
    el1_slider.on_changed(update)
    el2_slider.on_changed(update)

    plt.show()


def coupled_angles_plot(list_of_couples_of_views, list_of_couple_names):
    """
    Arg1: a list of tuple where each tuple is composed of two tuples of views, namely, a view is composed of two angles (az, el)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    for couple_view, couple_name in zip(list_of_couples_of_views,list_of_couple_names):
        v1 = couple_view[0]
        v2 = couple_view[1]
        az1,el1 = v1
        az2,el2 = v2
        ax.plot([az1, az2], [math.degrees(el1), math.degrees(el2)], linestyle='--', marker='o', label=couple_name)
    fig.legend(loc="upper right")
    return fig


def get_full_path_from_id(conf_path, file_id):
    with open(conf_path) as f:
        conf_content = json.load(f)    
    return os.path.join(os.path.dirname(conf_path), conf_content[file_id])

gpi = get_full_path_from_id

from LxGeoPyLibs.satellites.imd import IMetaData
import os, json

if __name__ == "__main__":

    top_conf="C:/DATA_SANDBOX/Alignment_Project/alignment_results/data_configs"
    views_names = [
        "Brazil Vila Velha",
        "Ethiopia Addis Ababa",
        "India Mumbai",
        "Pakistan Rawalpindi",
        "Qatar Doha",
        "Kuwait Kuwait",
        "Sweden Stockholm"]
    configs_paths = [
        "Brazil_Vila_Velha_GT.json",
        "Ethiopia_Addis_Ababa_GT.json",
        "India_Mumbai_GT.json",
        "Pakistan_Rawalpindi_GT.json",
        "Qatar_Doha_GT.json",
        "Kuwait_Kuwait_GT.json",
        "Sweden_Stockholm_GT.json",]
    configs_paths = map(lambda x: os.path.join(top_conf, x), configs_paths)

    list_of_couples_of_views = []

    for c_couple_conf in configs_paths:
        with open(c_couple_conf) as f:
            conf_dict = json.load(f)
        
        v1_conf_path = os.path.join(conf_dict["top_dir"], conf_dict["views"]["v1"])
        v2_conf_path = os.path.join(conf_dict["top_dir"], conf_dict["views"]["v2"])

        imd_path1 = gpi(v1_conf_path, "imd"); imd1=IMetaData(imd_path1)
        imd_path2 = gpi(v2_conf_path, "imd"); imd2=IMetaData(imd_path2)

        az1 = math.radians(imd1.satAzimuth()); el1 = math.radians(imd1.satElevation())
        az2 = math.radians(imd2.satAzimuth()); el2 = math.radians(imd2.satElevation())

        list_of_couples_of_views.append(
            ( (az1, el1), (az2, el2) )
        )

    fig = coupled_angles_plot(list_of_couples_of_views, views_names)
    plt.show()

    a=0
