import xml.etree.ElementTree as ET
import re
from pyproj import CRS, Transformer
import numpy as np
import pickle as pkl


def extract_speed_and_convert(unit_str):
    # 使用正则表达式匹配数字部分
    match = re.match(r'(\d+(\.\d+)?)([^\d]+)?', unit_str)
    if match:
        # 提取数字部分并转换为浮点数
        speed_mph = float(match.group(1))
        # 转换为公里每小时
        speed_kph = speed_mph * 0.621371
        return speed_mph, speed_kph
    else:
        raise ValueError("Unable to resolve speed limit values")



wgs84 = CRS('EPSG:4326')
# Defining the UTM coordinate system, using EPSG code
# The zone is 50 and the hemisphere is 'north', which can be adjusted according to actual needs
utm = CRS(f'EPSG:3405')


def wgs_to_utm(lat_lons):

    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    lats, lons = np.array(lat_lons).T
    utm_coords = transformer.transform(lons, lats)

    polyline_xy = np.column_stack((utm_coords[0], utm_coords[1]))

    zeros_z = np.zeros_like(polyline_xy[:, 0])
    polyline = np.column_stack((polyline_xy, zeros_z))
    return polyline



osm_file = '/your_osm_file.osm'


tree = ET.parse(osm_file)
root = tree.getroot()


map_features = {}
nodes = {}
# ways = []


for node in root.findall('.//node'):
    node_id = node.attrib['id']
    lat = float(node.attrib['lat'])
    lon = float(node.attrib['lon'])
    nodes[node_id] = (lat, lon)


for way in root.findall('way'):
    way_id = way.attrib['id']
    way_nodes = []
    for nd in way.findall('./nd'):
        ref = nd.attrib['ref']
        way_nodes.append(nodes[ref])
    polyline = wgs_to_utm(way_nodes)


    maxspeed_tag = way.find('.//tag[@k="SpeedLimit"]')
    if maxspeed_tag is not None:
        speed_limit_str = maxspeed_tag.get('v')
        try:
            speed_limit_mph, speed_limit_kph = extract_speed_and_convert(speed_limit_str)
        except ValueError as e:

            speed_limit_mph = speed_limit_kph = 0
    else:
        speed_limit_mph = speed_limit_kph = 0

    road_type = None
    # Check the road type and only deal with Lane and LaneBoundary
    highway_tag = way.find('.//tag[@k="Type"]')
    if highway_tag is not None:
        road_type_value = highway_tag.get('v')
        if road_type_value == 'LaneBoundary':
            road_type = 'ROAD_EDGE_BOUNDARY'
        if road_type_value == 'Lane':
            road_type = 'LANE_SURFACE_STREET'

    map_features[way_id] = {
        "speed_limit_mph": speed_limit_mph,
        'speed_limit_kph': speed_limit_kph,
        'type': road_type,
        "polyline": polyline
        # "left_neighbor":
        # "right_neighbor":

    }

# print(map_features)
# print(list(map_features.items())[:2])

pickle_data = pkl.dumps(map_features)

# 保存到文件
with open('map_features.pkl', 'wb') as f:
    f.write(pickle_data)
