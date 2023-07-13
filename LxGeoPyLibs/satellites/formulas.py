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



if __name__=="__main__":
    from LxGeoPyLibs.satellites.imd import IMetaData
    imd1_path = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_A_Neo/4d8738a2-816a-4924-b132-05fd48d011bb.IMD"
    imd2_path = "C:/DATA_SANDBOX/Alignment_Project/PerfectGT/Pakistan_Rawalpindi_B_Neo/5347f144-8bd0-4b5f-9b33-6aa1e7b469ed.IMD"
    imd1 = IMetaData(imd1_path)
    imd2 = IMetaData(imd2_path)

    dx1, dy1 = compute_roof2roof_constants(math.radians(imd1.satAzimuth()), math.radians(imd1.satElevation()),
                                         math.radians(imd2.satAzimuth()), math.radians(imd2.satElevation()))
    
    dx2, dy2 = compute_roof2roof_constants(math.radians(imd2.satAzimuth()), math.radians(imd2.satElevation()),
                                         math.radians(imd1.satAzimuth()), math.radians(imd1.satElevation()))
    
    rot_angle1 = compute_rotation_angle(math.radians(imd1.satAzimuth()), math.radians(imd1.satElevation()),
                                         math.radians(imd2.satAzimuth()), math.radians(imd2.satElevation()))
    
    rot_angle2 = compute_rotation_angle(math.radians(imd2.satAzimuth()), math.radians(imd2.satElevation()),
                                         math.radians(imd1.satAzimuth()), math.radians(imd1.satElevation()))

    pass