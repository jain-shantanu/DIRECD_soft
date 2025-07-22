import streamlit as st
import time
import numpy as np
from functions import map_calibration as mp
import sunpy.map
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label
from skimage.measure import label, regionprops, regionprops_table
import skimage.morphology
import astropy.units as u
from sympy import Plane, Point, Point3D



def create_sphere(cx,cy,cz, r, resolution=20):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    #return np.stack([x,y,z])
    return x,y,z

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def area_mask(smap):
# Input Binary map where 1 is dimming
    amap = np.zeros(smap.data.shape)
    rad, R_km, cenX, cenY, nanleft, dist_top = mp.map_info(smap)
    points_coords =  np.argwhere(smap.data==1)
    for x in range(len(points_coords)):
        pixel_coords = points_coords[x]
        area_2 = mp.calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top)
        amap[pixel_coords[0]][pixel_coords[1]] = area_2
    area_map = sunpy.map.Map(amap,smap.meta)
    return area_map    


def full_disk_mask1(segmented_img): 
    # BD_full_img : full-disk processed image that we create 2048x2048
    # segmented_img : corresponding cropped segmented image 
    data_full_img = np.arange(0, 2048*2048).reshape(2048, 2048)
    data_full_img = data_full_img*np.nan
    header = segmented_img.meta
    BD_full_img = sunpy.map.Map(data_full_img, header)
    BD_full_img.meta['crpix1'] = 1024
    BD_full_img.meta['crpix2'] = 1024
    BD_full_img.meta['naxis1'] = 2048
    BD_full_img.meta['naxis2'] = 2048
    
     
    final_size = BD_full_img.data.shape # [rows, columns] 
    crpix_x_full = round(BD_full_img.meta['crpix1']) 
    crpix_y_full = round(BD_full_img.meta['crpix2']) 
    crpix_x_crop = round(segmented_img.meta['crpix1']) 
    crpix_y_crop = round(segmented_img.meta['crpix2']) 
    # coordinate of the bottom left pixel 
    xpix= crpix_x_full - crpix_x_crop 
    ypix= crpix_y_full - crpix_y_crop 

     
    # Create final mask 
    data = segmented_img.data 
    height, length= data.shape 
#     print('Height: ', height,'Length: ', length) 
    fulldisk_mask= np.zeros(final_size)*np.nan
     
    ## Using PIL library 
    image_full=Image.fromarray(fulldisk_mask) 
    image_crop=Image.fromarray(data)     
    
    # box: a 2-tuple is used instead, it’s treated as the upper left corner: (xpix, ypix). 
    # Note: Image coordinate system starts from the upper left corner 
    box= (xpix, ypix) 
    Image.Image.paste(image_full, image_crop, box=box) 
    fulldisk_data=np.array(image_full) 
   
    fulldisk_mask= sunpy.map.Map(fulldisk_data, BD_full_img.meta) 

    return fulldisk_mask


def full_disk_mask2(segmented_img): 
    # BD_full_img : full-disk processed image that we create 2048x2048
    # segmented_img : corresponding cropped segmented image 
    data_full_img = np.arange(0, 1024*1024).reshape(1024, 1024)
    data_full_img = data_full_img*np.nan
    header = segmented_img.meta
    BD_full_img = sunpy.map.Map(data_full_img, header)
    BD_full_img.meta['crpix1'] = 1024/2
    BD_full_img.meta['crpix2'] = 1024/2
    BD_full_img.meta['naxis1'] = 2048/2
    BD_full_img.meta['naxis2'] = 2048/2
    
     
    final_size = BD_full_img.data.shape # [rows, columns] 
    crpix_x_full = round(BD_full_img.meta['crpix1']) 
    crpix_y_full = round(BD_full_img.meta['crpix2']) 
    crpix_x_crop = round(segmented_img.meta['crpix1']) 
    crpix_y_crop = round(segmented_img.meta['crpix2']) 
    # coordinate of the bottom left pixel 
    xpix= crpix_x_full - crpix_x_crop 
    ypix= crpix_y_full - crpix_y_crop 

     
    # Create final mask 
    data = segmented_img.data 
    height, length= data.shape 
#     print('Height: ', height,'Length: ', length) 
    fulldisk_mask= np.zeros(final_size)*np.nan
     
    ## Using PIL library 
    image_full=Image.fromarray(fulldisk_mask) 
    image_crop=Image.fromarray(data)     
    
    # box: a 2-tuple is used instead, it’s treated as the upper left corner: (xpix, ypix). 
    # Note: Image coordinate system starts from the upper left corner 
    box= (xpix, ypix) 
    Image.Image.paste(image_full, image_crop, box=box) 
    fulldisk_data=np.array(image_full) 
   
    fulldisk_mask= sunpy.map.Map(fulldisk_data, BD_full_img.meta) 

    return fulldisk_mask

def resize_full_disk(full_disk,shape): 
     # BD_full_img : full-disk processed image that we create 2048x2048
    # segmented_img : corresponding cropped segmented image 
    data_full_img = np.arange(0, shape[0]*shape[1]).reshape(shape[0]*shape[1])
    data_full_img = data_full_img*np.nan
    header = full_disk.meta
    BD_full_img = sunpy.map.Map(data_full_img, header)
    BD_full_img.meta['crpix1'] = int(shape[0]/2)
    BD_full_img.meta['crpix2'] = int(shape[1]/2)
    try:
        BD_full_img.meta['rsun'] = BD_full_img.meta['rsun_ref']
    except:
        BD_full_img.meta['rsun'] = BD_full_img.meta['rsun']
    BD_full_img.meta['cdelt1'] = BD_full_img.meta['cdelt1']*(header['crpix1']/BD_full_img.meta['crpix1'])
    BD_full_img.meta['cdelt2'] = BD_full_img.meta['cdelt2']*(header['crpix1']/BD_full_img.meta['crpix1'])
    
    data_rescaled = sunpy.image.resample.resample(full_disk.data,(shape[0],shape[1]),method='nearest')

    fulldisk_scaled = sunpy.map.Map(data_rescaled,BD_full_img.meta)
    return fulldisk_scaled


def eliminate_small_regions(binary):
#     flipped_binary = np.flipud(binary)
    label_im = label(binary)
    region_area=[]
    for region in regionprops(label_im):
        region_area.append(region.area)

    if not region_area: 
        modified_mask = binary
    else:
        im = skimage.morphology.remove_small_objects(label_im, min_size=0.05*np.max(region_area))
        
        modified_mask = np.where(im==0,False,True)
    return modified_mask

def choosing_sectors(h) :   
 

    sector7 = h[(h.lon > -15*u.deg) & (h.lon <= 15*u.deg)]
    sector8 = h[(h.lon > 15*u.deg) & (h.lon <= 45*u.deg)]
    sector9 = h[(h.lon > 45*u.deg) & (h.lon <= 75*u.deg)]
    sector10 = h[(h.lon > 75*u.deg) & (h.lon <= 105*u.deg)]
    sector11 = h[(h.lon > 105*u.deg) & (h.lon <= 135*u.deg)]
    sector12 = h[(h.lon > 135*u.deg) & (h.lon <= 165*u.deg)]
    
    sector1 = h[((h.lon > 165*u.deg) & (h.lon <= 180*u.deg)) | ((h.lon >= -180*u.deg)&(h.lon <= -165*u.deg))]
    sector2 = h[(h.lon > -165*u.deg) & (h.lon <= -135*u.deg)]
    sector3 = h[(h.lon > -135*u.deg) & (h.lon <= -105*u.deg)]
    sector4 = h[(h.lon > -105*u.deg) & (h.lon <= -75*u.deg)]
    sector5 = h[(h.lon > -75*u.deg) & (h.lon <= -45*u.deg)]
    sector6 = h[(h.lon > -45*u.deg) & (h.lon <= -15*u.deg)]
    
    return sector1,sector2,sector3,sector4,sector5,sector6,sector7,sector8,sector9,sector10,sector11,sector12

def sectors_to_pix_and_data(sector_n,smap,area_map=False,intensity_map=False):
    #dim_mask = np.argwhere(sector_n == sector_n)
    pixels = np.asarray(np.rint(smap.world_to_pixel(sector_n)), dtype=int)
    x = pixels[0, :]
    y = pixels[1, :]
    dim = smap.data[y,x]
    if area_map !=False:
        area = area_map.data[y,x]
    else:
        area = 0
    if intensity_map !=False:
        intensity = intensity_map.data[y,x]
    else: 
        intensity =0
    return area,intensity,dim,x,

def get_plane(txtt,points): 
    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = uu = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = vv = [x1-x2, y1-y2, z1-z2]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point_initial  = np.array(p1)
    normal = np.array(u_cross_v)

    normalized_normal = normal/np.sqrt(normal[0]**2 + normal[1]**2 +normal[2]**2) 
    
# Normal to C
    N_C_x = p1[0]+normalized_normal[0]
    N_C_y = p1[1]+normalized_normal[1]
    N_C_z = p1[2]+normalized_normal[2]

    vec_normal = [N_C_x,N_C_y,N_C_z]

    d = -point_initial.dot(normal)
    if txtt=='red':
            xx, zz = np.meshgrid(np.ceil(range(int(x0-2e8), int(x1+4e8),int(1e8))), np.ceil(range(int(z0-3e8),int(z1+7e8),int(1e8))))
            n = (-normal[0] * xx - normal[2] * zz - d) * 1. / normal[1]
    elif txtt=='green':
            xx, zz = np.meshgrid(range(int(1e8), int(15e8),int(1e8)), range(int(-5e8),int(7e8),int(1e8)))
            n = (-normal[0] * xx - normal[1] * zz - d) * 1. / normal[2]
    
    return (xx,n,zz,normal)


def proj_plane(P_x, P_y,P_z,C_x,C_y,C_z,normal):
    #2 points of the line and the normal of the plane
    
    p = Point3D(P_x, P_y,P_z)
    
    # using Plane()
    p1 = Plane(Point3D(C_x,C_y,C_z), normal_vector =(normal[0], normal[1], normal[2]))

    # using projection()
    projectionPoint = p1.projection(p)

    projectionPoint.evalf()[0]

    xp = projectionPoint.evalf()[0]
    yp = projectionPoint.evalf()[1]
    zp = projectionPoint.evalf()[2]
    
    return [float(xp),float(yp),float(zp)]

def angle_pos_neg(vec1,vec2):
# Calculate the dot product of the vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes (norms) of the vectors
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Calculate the angle in degrees
    angle_deg = np.degrees(angle_rad)

    # Check if the angle is positive or negative based on vector direction
    # cross_product = np.cross(vec1, vec2)
    # if np.dot(cross_product, np.array([0, 0, 1])) < 0:
    #     angle_deg = -angle_deg
    return angle_deg

def km_to_pixel_coords(km_coords, rad, Rsun, cenX, cenY, map_shape):
    """Convert km coordinates to pixel coordinates"""
    px_yz = km_coords[1:] * (rad / Rsun)  # Scale to pixels
    px_yz[1] *= -1  # Flip y-axis
    coord = np.round(px_yz + np.array([cenX, cenY])).astype('int')
    coord[1] = map_shape[1] - coord[1]  # Adjust y-coordinate to image coordinates
    return coord

