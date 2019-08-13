# small Austin Area
import anuga
import numpy

cache = False
verbose = True

# Define Scenario
# the final goal is to define a rainfall scenario
# Here, in order to test, we follow the example to first have a fixed wave scenario
scenario = 'fixed_wave'
name_stem = 'aw_small'
meshname = name_stem + '.msh'

gage_file_name = 'aw_small_gauges.csv'

# bounding polygon for study area
bounding_polygon = anuga.read_polygon('aw_small_extent.csv')
A = anuga.polygon_area(bounding_polygon) / 1000000.0
print 'Area of bounding polygon = %.2f km^2' % A

# Read interior polygons
poly_river = anuga.read_polygon('aw_small_river.csv')

# the greater the base_scale is the less the triangle it will be divided into
just_fitting = False
base_scale = 5000
background_res = 10 * base_scale
river_res = base_scale
city_res = base_scale

interior_regions = [[poly_river,river_res]]

tide = 0.0
