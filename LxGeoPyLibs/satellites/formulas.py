### module of useful formulas related to satellite acquisation metadata

import math
# cotangent lambda function
cot = lambda x : 1/math.tan(x)

def compute_rotation_angle(satAz1, satEl1, satAz2, satEl2):
    """
    Computes rotation angle to be applied on images for epipolar alignment
    """    
    rot_angle = ( cot(satEl1) * math.cos(satAz1) - math.cos(satAz2) * cot(satEl2) ) / \
        ( cot(satEl1) * math.sin(satAz1) - math.sin(satAz2) * cot(satEl2) )
    return math.atan(rot_angle)

def compute_axis_displacement_ratios(angle):
    """
    Computes displacement on X & Y axis using angle value.
    Used for decoupling disparity or elevation values to X & Y axis
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    return dx,dy

