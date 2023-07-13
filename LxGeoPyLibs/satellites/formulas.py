### module of useful formulas related to satellite acquisation metadata

import math
# cotangent lambda function
cot = lambda x : 1/math.tan(x)

def compute_rotation_angle(satAz1, satEl1, satAz2, satEl2):
    """
    Computes rotation angle to be applied on images for epipolar alignment
    """    
    dx, dy = compute_roof2roof_constants(satAz1, satEl1, satAz2, satEl2)
    return math.atan2(dy, dx)

def compute_roof2roof_constants(satAz1_r, satEl1_r, satAz2_r, satEl2_r):
    """
    Computes roof to roof translation coeficients between two orthos using acquisation angles
    """
    dX = math.sin(satAz1_r)/math.tan(satEl1_r) - math.sin(satAz2_r)/math.tan(satEl2_r)
    dY = math.cos(satAz1_r)/math.tan(satEl1_r) - math.cos(satAz2_r)/math.tan(satEl2_r)
    return dX, dY

def nicolas_roof2roof(satAz1_r, satEl1_r, satAz2_r, satEl2_r):
    num = math.tan(satEl1_r) * math.tan(satEl2_r)
    denum = math.sqrt( math.tan(satEl1_r)**2 + math.tan(satEl2_r)**2 - 2*math.tan(satEl1_r)*math.tan(satEl2_r) *math.cos(satAz1_r-satAz2_r) )
    return num / denum



