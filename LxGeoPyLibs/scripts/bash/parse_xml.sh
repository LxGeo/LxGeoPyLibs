xml_file=$1
output_geo_imd_file=$2

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters!"
    echo "Provide input xml and output geo imd in order!"
    exit 1
fi

# check file
if [[ ! -e $xml_file ]]
then
    echo "file not found"
    exit 1
fi

get_attr_mean(){
local attr_name=$1
local file_name=$2

local mean_value=$( 
python - << EOF
import xml.etree.ElementTree as ET
def mean(values_list):
    return float(sum(values_list))/len(values_list)

tree = ET.parse(r"$file_name")
root = tree.getroot()
mean_value=float(mean(list(map( lambda x: float(x.text), root.findall(".//$attr_name") ))))
import sys
sys.stdout.write(str(mean_value))
EOF
)
echo "$mean_value"
}

get_attr_first_value(){
attr_name=$1
file_name=$2
echo $(xmllint --xpath 'string(//'${attr_name}')' ${file_name})
}

# Pi value
Pi=$(echo "4*a(1)" | bc -l)
# sat name
satelliteID=$(get_attr_first_value MISSION $xml_file)
# sun azimuth
meanSunAz=$(get_attr_mean "SUN_AZIMUTH" $xml_file)
# sun elevation
meanSunEl=$(get_attr_mean "SUN_ELEVATION" $xml_file)
# azimuth
azimA=$(get_attr_mean "AZIMUTH_ANGLE" $xml_file)
# Incidence angle accross track
IncAAccross=$(get_attr_mean "INCIDENCE_ANGLE_ACROSS_TRACK" $xml_file)
# Incidence angle along track
IncAAlong=$(get_attr_mean "INCIDENCE_ANGLE_ALONG_TRACK" $xml_file)
# Incidence angle
IncAngle=$(get_attr_mean "INCIDENCE_ANGLE" $xml_file)


## Transform
IncAAccrossRAD=$(bc -l <<< "$IncAAccross * $Pi / 180")
IncAAlongRAD=$(bc -l <<< "$IncAAlong * $Pi / 180")
atanSat=$(echo "a( ( s( $IncAAccrossRAD ) / c( $IncAAccrossRAD ) ) / ( s( $IncAAlongRAD ) / c( $IncAAlongRAD ) ) )" | bc -l )

meanSatAz=$(eval "awk \"BEGIN{ print (  (180 / $Pi) * ( atan2( sin($IncAAlongRAD)/cos($IncAAlongRAD), sin($IncAAccrossRAD)/cos($IncAAccrossRAD) ) ) ) % 360 + 90" }\"  )

if (( $(bc <<< "sqrt(($azimA - $meanSatAz)^2)<1") ))
then
    finalAzi=$azimA
else
    if (( $(bc <<< "sqrt(($azimA - 90)^2)<1") ))
    then
        finalAzi=$(bc <<< "($meanSatAz + $azimA - 90)")
    else
        if (( $(bc <<< "sqrt(($azimA - 180)^2)<1") ))
        then
            finalAzi=$(bc <<< "($meanSatAz + $azimA - 180)")
        else
            finalAzi=$meanSatAz
        fi
    fi
fi

meanSatEl=$(echo "-$IncAngle+90" | bc -l)

geo_imd_content="BEGIN_GROUP = IMAGE_1\n\tsatId = \"${satelliteID}\";\n\tmeanSunAz = ${meanSunAz} ;\n\tmeanSunEl = ${meanSunEl} ;\n\tmeanSatAz = ${finalAzi};\n\tmeanSatEl = ${meanSatEl};\nEND_GROUP = IMAGE_1\nEND;"
echo -e $geo_imd_content > $output_geo_imd_file