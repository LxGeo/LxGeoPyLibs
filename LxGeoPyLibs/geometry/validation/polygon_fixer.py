import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString

class linked_ring_coords(list):
    def __init__(self, coordinates=None):
        super().__init__(coordinates)
    
    def __getitem__(self, idx):
        if idx < -len(self):
            return self.__getitem__(idx +len(self))
        elif idx > len(self):
            return self.__getitem__(idx-len(self))
        else:
            return super().__getitem__(idx)

def pts_collinear(p1, p2, p3):
    """
    """
    vector_1 = [p1[0]-p2[0], p1[1]-p2[1]]
    vector_2 = [p3[0]-p2[0], p3[1]-p2[1]]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if np.isnan(angle):
        return True
    angle = min(angle, np.pi-angle)
    pts_collinear = angle < (np.pi / 180)
    return pts_collinear

def poly_remove_lines(ring):
    """
    """
    if not (ring.coords):
        return ring
    #exterior_coords=poly.exterior.coords[:-1]
    exterior_coords=linked_ring_coords(ring.coords[:-1])
    simplified_R=[]
    turn_m2_idx = len(exterior_coords) -2
    turn_m1_idx = len(exterior_coords) -1
    
    c_pt_idx = 0
    deleted_pt_cnt=0
    while (len(simplified_R) <= len(exterior_coords)):
        
        if not pts_collinear( exterior_coords[c_pt_idx], exterior_coords[turn_m1_idx], exterior_coords[turn_m2_idx] ):
            simplified_R.append(exterior_coords[turn_m1_idx])                        
            turn_m2_idx=turn_m1_idx
            turn_m1_idx=c_pt_idx
            c_pt_idx+=1
            c_pt_idx%=len(exterior_coords)
        
        else:
            exterior_coords.pop(turn_m1_idx)
            deleted_pt_cnt+=1
            if simplified_R: simplified_R.pop()
            c_pt_idx-=1
            #turn_m1_idx-=1
            #turn_m2_idx-=1
            turn_m1_idx=c_pt_idx-1
            turn_m2_idx=turn_m1_idx-1
            
            continue
        
    simplified_R.append(simplified_R[0])
    return simplified_R