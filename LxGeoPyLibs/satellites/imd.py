
### Module for IMD files loading

from pathlib import Path
import yaml
import click
import xml.etree.ElementTree as ET
from math import atan, pi, cos, sin, tan, atan2, radians, degrees, fabs

class MissingImdFileParser(BaseException):
    def __init__(self, extension:str=None):        
        self.extension = extension
        super(MissingImdFileParser, self).__init__()        
        
    def __str__(self):
        if not self.extension:
            return "File extension is not provided!"        
        return f"File extension [{self.extension}] is not supported"
    pass

def mean(values_list):
    return float(sum(values_list))/len(values_list)

def parse_GeoImd(imd_xml_file, imd_metadata_object):

    dSatElv, dSatAzi = -999.9, -999.9
    sSatElv = 'meanSatEl'
    sSatAzi = 'meanSatAz'

    oFin = open( imd_xml_file, 'r')
    for sLine in oFin:
        sLine = sLine.replace(' ','').replace('\n', '').replace('\t', '').replace(';', '')
        try:
            aParts = sLine.split('=')
            if len(aParts) >= 2:
                if aParts[0] == sSatElv:
                    dSatElv = radians( float( aParts[1] ) )
                elif aParts[0] == sSatAzi :
                    dSatAzi = radians( float( aParts[1] ) )
        except:
            raise IOError
    imd_metadata_object.dSatElv = dSatElv
    imd_metadata_object.dSatAzi = dSatAzi

def parse_XmlImd(imd_xml_file, imd_metadata_object):
    tree = ET.parse(imd_xml_file)
    root = tree.getroot()
    ######
    imd_metadata_object.satelliteID = list(map( lambda x: str(x.text), root.findall(".//MISSION") ))[0]
    
    #####
    imd_metadata_object.meanSunAz = meanSunAz = float(mean(list(map( lambda x: float(x.text), root.findall(".//SUN_AZIMUTH") ))))
    imd_metadata_object.meanSunEl = meanSunEl = float(mean(list(map( lambda x: float(x.text), root.findall(".//SUN_ELEVATION") ))))
    azimA = mean(list(map( lambda x: float(x.text), root.findall(".//AZIMUTH_ANGLE") )))
    IncAAccross = mean(list(map( lambda x: float(x.text), root.findall(".//INCIDENCE_ANGLE_ACROSS_TRACK") )))
    IncAAlong = mean(list(map( lambda x: float(x.text), root.findall(".//INCIDENCE_ANGLE_ALONG_TRACK") )))
    IncAngle = mean(list(map( lambda x: float(x.text), root.findall(".//INCIDENCE_ANGLE") )))
    if None in [ meanSunAz, meanSunEl, azimA, IncAAccross, IncAAlong, IncAngle ]:
        imd_metadata_object.dSatAzi = None
        imd_metadata_object.dSatElv = None            
        return
    
    IncAAccrossRAD = IncAAccross * pi / 180
    IncAAlongRAD = IncAAlong * pi / 180
    
    meanSatAz = (degrees(atan2( tan(IncAAlongRAD), tan(IncAAccrossRAD))) ) % 360 +90
    
    if fabs( azimA - meanSatAz ) < 1.0 :
        finalAzi = azimA
    else:
        if fabs( azimA - 90.0 ) < 1.0 :
            finalAzi = meanSatAz + azimA - 90.0
        elif fabs( azimA - 180.0 ) < 1.0 :
            finalAzi = meanSatAz + azimA - 180.0            
        else:
            finalAzi = meanSatAz
    
    meanSatEl = -IncAngle + 90
    
    imd_metadata_object.dSatAzi = float(finalAzi)
    imd_metadata_object.dSatElv = float(meanSatEl)


class IMetaData:

    parsers_map = {".xml":parse_XmlImd, ".imd": parse_GeoImd}

    def __init__(self, imd_xml_file):
        """
        Get information about a scene from an IMF file
        :param oScene: LxScene: the sat scene.
        :return:
        """
        file_extension = Path(imd_xml_file).suffix.lower()
        if file_extension not in self.parsers_map: raise MissingImdFileParser(file_extension)
        parser = self.parsers_map.get(file_extension.lower())
        parser(imd_xml_file, self)
        

    def satID(self):
        return self.satelliteID
    
    def satAzimuth(self):
        return self.dSatAzi

    def satElevation(self):
        return self.dSatElv
    
    def sunAzimuth(self):
        return self.meanSunAz
        
    def sunElevation(self):
        return self.meanSunEl
    
    def IMD_dict(self):
        imd_dict = {"image": {}}
        imd_dict["image"]["satAzimuth"] = self.satAzimuth()
        imd_dict["image"]["satElevation"] = self.satElevation()
        imd_dict["image"]["sunAzimuth"] = self.sunAzimuth()
        imd_dict["image"]["sunElevation"] = self.sunElevation()
        return imd_dict
    
    def IMD_geo(self):
        """
        Returns imd data in predfined format
        """
        imd_string = 'BEGIN_GROUP = IMAGE_1\n\tsatId = "{}";\n\tmeanSunAz = {} ;\n\tmeanSunEl = {} ;\n\tmeanSatAz = {};\n\tmeanSatEl = {};\nEND_GROUP = IMAGE_1\nEND;'
        return imd_string.format(self.satID(), self.sunAzimuth(), self.sunElevation(), self.satAzimuth(), self.satElevation())


    def get_constants(self):
        """
        Returns a dict of X & Y constants
        """

        def getCstX(satAz, satEl):
            dSinSatAz = sin(satAz)
            return dSinSatAz / tan( satEl )
        def getCstY(satAz, satEl):
            dCosSatAz = cos(satAz)
            return dCosSatAz / tan( satEl )
        
        satAz = radians(self.satAzimuth())
        satEl = radians(self.satElevation())
        
        coefsDict = { "coefX": getCstX(satAz, satEl), "coefY": getCstY(satAz, satEl) }
        return coefsDict