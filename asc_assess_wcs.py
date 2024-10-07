#!/usr/bin/env python

# Read the WCS from an image (1st parameter) and then read
# x,y positions from file (2nd parameter) and calculate RA and
# Dec corresponding to those x,y positions using the WCS.

# Example usage:
# ./calc_radec_from_WCS.py image coord_list
#
# where coord file is a headerless file of this format:
# 1 21:35:08.107 +52:11:32.42 starname1
# 2 21:35:20.316 +52:10:06.43 starname2
# 3 21:34:51.150 +52:06:02.12 starname3
#
# Example:
# ./calc_radec_from_WCS.py r478577-4.fits positions.dat
#
# Had to install scikit
#  pip install scikit-image
#
# This program does not seem to like ZPX transforms, error is:
# 'File "/home/rahm/anaconda3/lib/python3.8/site-packages/astropy/wcs/wcs.py", # line 448, in __init__
#    tmp_wcsprm = _wcs.Wcsprm(header=tmp_header_bytes, key=key,
# ValueError: Internal error in wcslib header parser:


import sys
#import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy import units as u
#import PIL
#from PIL import Image, ImageOps
from skimage import exposure
from astropy import coordinates as coord
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)
import math

# Check input parameters
if ( len(sys.argv) != 3):
    print("There is a problem with your arguments.")
    print("Argument 1 should be an existing fits file.")
    print("Argument 2 should be a file containing coordinates.")
    sys.exit('Parameters are wrong')

# If all OK, then first argument is the file to open
imagefile = sys.argv[1]
print("Opening ",imagefile)

# Open image to get image parameters
# image_data = fits.getdata(imagefile)

hdu = fits.open(imagefile)[0]
wcs = WCS(hdu.header)

# Print WCS
print("WCS is:\n",wcs)

# Read in list of coordinates ra dec from file specified on command line.
coordfile = sys.argv[2]
print("Opening coordinates file ",coordfile,"\n")
fd = open(coordfile, 'r') 
lines = fd.readlines()

# Plot image
print("Plotting image")
plt.figure(figsize=(12, 12))
#plt.draw()

lowval = np.percentile(hdu.data, 50)
highval = np.percentile(hdu.data, 90)
plt.imshow(hdu.data, cmap='hot', origin='lower', vmin=lowval, vmax=highval)
#plt.draw()

# Intialise variable for cumulative sum of distances
distsum = 0.0

# Loop through each line reading the coordinates and calculating
# World coordinates (RA and Dec)
for line in lines:
    xycoords = line.split()
    coordstr = xycoords[0] + ' ' + xycoords[1]
    print("coordstr is ",coordstr)
    star_coord = SkyCoord(coordstr, unit=(u.hourangle,u.deg) )
    
    coordarray = np.array([[star_coord.ra.deg, star_coord.dec.deg ]], dtype=np.float64)
    # Convert ra, dec to pixels using WCS
    pixarr = wcs.wcs_world2pix(coordarray, 0)
    print("pixarr is ", pixarr)
    # Now find peaks in data by masking other areas except for a box 
    # around our region of interest.
    # Set box size
    subarr_size = 60
    xpos = pixarr[0][0]
    ypos = pixarr[0][1]
    print("xpos = ",xpos,"   ypos = ",ypos,"\n")
    plt.scatter(xpos,ypos,s=300,edgecolor='red',facecolor='none')
    plt.text(xpos,ypos,'target radec',color='red')
    # Calculate box corners
    # Need to use numpy coordinates row,column (y,x) instead of data coords (x,y)
    startx = int(round(xpos)-subarr_size/2)
    endx = int(round(xpos)+subarr_size/2)
    starty = int(round(ypos)-subarr_size/2)
    endy = int(round(ypos)+subarr_size/2)
    # Need 5 points to give complete rectangle
    rectanglex = [startx, endx,endx,startx,startx]
    rectangley = [starty,starty,endy,endy,starty]
    plt.plot(rectanglex, rectangley, color='green')
    # Extract a subarray hopefully including star
    # Need to use rows and columns here?
    startrow = starty
    endrow = endy
    startcolumn = startx
    endcolumn = endx
    subarr = hdu.data[startrow:endrow,startcolumn:endcolumn]
    # Get stats of subarray
    mean, median, std = sigma_clipped_stats(subarr, sigma=3.0)
    # define daofind command
    daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
    # Create mask array wth all values set to True
    mask = np.ones(hdu.data.shape, dtype=bool)
    mask2 = np.zeros(hdu.data.shape, dtype=int)
    mask2[startrow:endrow,startcolumn:endcolumn] = 1
    maskimage = mask2 * hdu.data
    lowval = np.percentile(subarr, 50)
    highval = np.percentile(subarr, 90)
#    plt.imshow(maskimage, cmap='hot', origin='lower', vmin=lowval, vmax=highval)
#    plt.waitforbuttonpress()
    # Set the subarray to false (no masking)
    mask[startrow:endrow,startcolumn:endcolumn] = False
    # Run the DAO starfinder software
    # Creates an astropy table called sources
    sources = daofind(hdu.data - median, mask=mask)
    # Set floating point numbers to be 8 significant figures?
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
    print(sources)
    sources.sort('flux', reverse=True)
    brightest_source = sources[:1]
    print("Brightest source:")
    print(brightest_source)
    daorow = brightest_source[0][2]
    daocolumn = brightest_source[0][1]
    daox = daocolumn
    daoy = daorow
    print("Daophot x,y are ",daorow, daocolumn)
    plt.scatter(daox,daoy,s=300,edgecolor='orange',facecolor='none')
    # plt.text uses data coordinates by default
    plt.text(daox, daoy, 'Daophot',color='white')

    # Try some other centroiding algorithms
    # Centre of mass centroids
    # Returns coords in pixel (x,y) order not numpy axis order (row,column)
    xycen1 = centroid_com(hdu.data - median,mask=mask)
    print("centre of mass centroids are ",xycen1)
    # These coords are in x,y not row, column
    comrow = xycen1[1]
    comcolumn = xycen1[0]
    comx = xycen1[0]
    comy = xycen1[1]
    plt.scatter(comx,comy,s=300,edgecolor='blue',facecolor='none')
    plt.text(comx,comy,'COM',color='blue')
    # Again outputs centroid coords in x,y not row, column (I think)
    xycen2 = centroid_quadratic(hdu.data - median, mask=mask)
    print("quadratic fit are ",xycen2)
    quadx = xycen2[0]
    quady = xycen2[1]
    plt.scatter(quadx,quady,s=300,edgecolor='pink',facecolor='none')
    plt.text(quadx,quady,'Quadratic',color='pink')
    # Again outputs centroid coords in x,y not row, column (I think)
    xycen3 = centroid_1dg(hdu.data - median, mask=mask)
    print("1 dimension fitting centroids are ",xycen3)
    fit1dx = xycen3[0]
    fit1dy = xycen3[1]
    plt.scatter(fit1dx,fit1dy,s=300,edgecolor='brown',facecolor='none')
    plt.text(fit1dx,fit1dy, '1D fit', color='brown')
    # The 2D centroid took a long time and printed crazy numbers
#    xycen4 = centroid_2dg(hdu.data - median, mask=mask)
#    print("2 dimension fitting centroids are ",xycen4)
#    plt.scatter(xycen1[0],xycen1[1])

    # Use 1d gauss fit as best guess for centroid`
    # Calculate distance in pixels
    dist = math.sqrt( (fit1dx-xpos)**2 + (fit1dy-ypos)**2 )
    print("distance in pixels is ",dist)
    distsum = distsum + dist
    plt.draw()
#    plt.waitforbuttonpress()
    print(" ")

plt.draw()
print("Cumulative sum of distances is ",distsum)
plt.show()
fd.close() 
            

