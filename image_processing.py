from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


####################
# HELPER FUNCTIONS #
####################

def div_by_ref(images):
    """Divide images by the reference"""
    return {k: v / images['ref'] for k,v in images.items() if k != 'ref'}


def frame_image(images, bright_threshold):
    """Find the center of the object and define a clipping frame around it"""
    """Return the center of the object and the x and y limits of the frame"""
    ref_img = images['ref']
    ref_max = np.unravel_index(np.argmax(ref_img), ref_img.shape)
    # Average column and row value
    ref_mean_x = np.mean(ref_img, axis=0)
    ref_mean_y = np.mean(ref_img, axis=1)
    # Index of the row and the column with max mean value
    ref_max_x = np.argmax(ref_mean_x)
    ref_max_y = np.argmax(ref_mean_y)
    # Filter out pixels with brightness below threshold
    ref_bright_x = np.where(ref_img[ref_max_y] > bright_threshold)[0]
    ref_bright_y = np.where(ref_img[:,ref_max_x] > bright_threshold)[0]
    # Determine diameter and center
    ref_d_x = len(ref_bright_x)
    ref_d_y = len(ref_bright_y)
    ref_center_x = int(ref_bright_x[0]+ref_d_x/2)
    ref_center_y = int(ref_bright_y[0]+ref_d_y/2)
    # Radius
    ref_r = (ref_d_x + ref_d_y) / 4
    # Frame with 20% spacing
    x_limits = ref_center_x - int(ref_r * 1.2), ref_center_x + int(ref_r * 1.2)
    y_limits = ref_center_y - int(ref_r * 1.2), ref_center_y + int(ref_r * 1.2)
    return {
        'center': (ref_center_x, ref_center_y),
        'xlims': x_limits,
        'ylims': y_limits,
    }


def clip(image, frame):
    """Given an image and a frame, do the clipping"""
    xlims = frame['xlims']
    ylims = frame['ylims']
    return image[ylims[0]:ylims[1], xlims[0]:xlims[1]]


def clip_images(images, frame):
    """Given a frame and a set of images, clip the images"""
    return {k: clip(v, frame) for k, v in images.items()}


# Definition of the disposition of the images in the side by side plot
disposition = {
    (0,0): 'left',
    (0,1): 'right',
    (1,0): 'top',
    (1,1): 'bottom',
    (2,0): 'clockwise',
    (2,1): 'counterclockwise'
}


def side_by_side_plot(images, title):
    """Plot the images of various angles and positions side by side"""
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    fig.suptitle(title)
    for k, v in disposition.items():
        axes[k].imshow(images[v], cmap=colormap)
        axes[k].set_title(v.title())


def clip_bin(image, center, bins):
    """Given an image, its center and bin format, extract the bin data"""
    bindata = []
    for b in bins:
        bindata.append(image[(center[1]+b['y0']):(center[1]+b['y1']),
                        (center[0]+b['x0']):(center[0]+b['x1'])])
    return np.array(bindata)


def take_image_bins(images, frame, bins):
    """Extract bin data from the image and calculate its average, for all bins"""
    imagebins = {k: np.mean(clip_bin(img, frame['center'], bins), axis=(1,2)) 
                 for k, img in images.items()
                }
    return imagebins


def add_ratios(image_bins):
    for k, v in image_bins.items():
        # Append radial and azimuthal ratios
        image_bins[k] = np.append(v,
                                    [abs((v[0]+v[3])/(v[1]+v[2])-1),
                                     abs((v[0]+v[1])/(v[2]+v[3])-1)
                                    ]
                                 )


def get_image_bins_and_ratios(images, frame, bins):
    """Calculate average bin values and scrambling rations"""
    ib = take_image_bins(images, frame, bins)
    add_ratios(ib)
    return ib


def print_bin_values(title, image_bins):
    """Function that prints the image bins"""
    print(title)
    print('input, bin 1, bin 2, bin 3, bin 4, radial scrambling, aximuthal scrambling')
    for k, values in image_bins.items():
        print(f"{k}, {','.join([str(v) for v in values])}")


############
# SETTINGS #
############

# PLOTS
colormap = "gray_r"

# Parameters
nf_bright_threshold = 500
ff_bright_threshold = 250
nf_ds_bright_threshold = 250
ff_ds_bright_threshold = 250

# We are taking 20px by 20px bin sizes, which is an appropriate size
# for the smaller images
binsize = (20, 20)

# ... and the bin defining vertices are
bins = [{'x0': (k-2)*binsize[0], 'y0': int(-binsize[1]/2), 'x1': (k-1)*binsize[0], 'y1': int(binsize[1]/2)} for k in range(0,4)]


###################
# LOAD IMAGE DATA #
###################

# NEAR FIELD IMAGES
nf_images = {
    'ref': fits.getdata("near field reference.fits"),
    'left': fits.getdata("near field left.fits"),
    'right': fits.getdata("near field right.fits"),
    'top': fits.getdata("near field top.fits"),
    'bottom': fits.getdata("near field bottom.fits"),
    'clockwise': fits.getdata("near field angle 1.fits"),
    'counterclockwise': fits.getdata("near field angle 2.fits"),
}
# NEAR FIELD OPERATIONS
nf_div_images = div_by_ref(nf_images)

# FAR FIELD IMAGES
ff_images = {
    'ref': fits.getdata("far field reference.fits"),
    'left': fits.getdata("far field left.fits"),
    'right': fits.getdata("far field right.fits"),
    'top': fits.getdata("far field top.fits"),
    'bottom': fits.getdata("far field bottom.fits"),
    'clockwise': fits.getdata("far field angle 1.fits"),
    'counterclockwise': fits.getdata("far field angle 2.fits"),
}
# FAR FIELD OPERATIONS
ff_div_images = div_by_ref(ff_images)

# NEAR FIELD IMAGES, DOUBLE SCRAMBLER
nf_ds_images = {
    'ref': fits.getdata("double scrambler near field reference.fits"),
    'left': fits.getdata("double scrambler near field left.fits"),
    'right': fits.getdata("double scrambler near field right.fits"),
    'top': fits.getdata("double scrambler near field top.fits"),
    'bottom': fits.getdata("double scrambler near field bottom.fits"),
    'clockwise': fits.getdata("double scrambler near field angle 1.fits"),
    'counterclockwise': fits.getdata("double scrambler near field angle 2.fits"),
}
# NEAR FIELD OPERATIONS
nf_ds_div_images = div_by_ref(nf_ds_images)

# FR FIELD IMAGES, DOUBLE SCRAMBLER
ff_ds_images = {
    'ref': fits.getdata("double scrambler far field reference.fits"),
    'left': fits.getdata("double scrambler far field left.fits"),
    'right': fits.getdata("double scrambler far field right.fits"),
    'top': fits.getdata("double scrambler far field top.fits"),
    'bottom': fits.getdata("double scrambler far field bottom.fits"),
    'clockwise': fits.getdata("double scrambler far field angle 1.fits"),
    'counterclockwise': fits.getdata("double scrambler far field angle 2.fits"),
}
# NEAR FIELD OPERATIONS
ff_ds_div_images = div_by_ref(ff_ds_images)

##################
# PROCESS IMAGES #
##################

# CLIP AND PLOT NEAR FIELD
nf_frame = frame_image(nf_images, nf_bright_threshold)
nf_cliped_div_images = clip_images(nf_div_images, nf_frame)

# CLIP AND PLOT FAR FIELD
ff_frame = frame_image(ff_images, ff_bright_threshold)
ff_cliped_div_images = clip_images(ff_div_images, ff_frame)

# CLIP AND PLOT DOUBLE SCRAMBLER NEAR FIELD
nf_ds_frame = frame_image(nf_ds_images, nf_ds_bright_threshold)
nf_ds_cliped_div_images = clip_images(nf_ds_div_images, nf_ds_frame)

# CLIP AND PLOT DOUBLE SCRAMBLER FAR FIELD
ff_ds_frame = frame_image(ff_ds_images, ff_ds_bright_threshold)
ff_ds_cliped_div_images = clip_images(ff_ds_div_images, ff_ds_frame)

# BINS
nf_image_bins = get_image_bins_and_ratios(nf_div_images, nf_frame, bins)
ff_image_bins = get_image_bins_and_ratios(ff_div_images, ff_frame, bins)
nf_ds_image_bins = get_image_bins_and_ratios(nf_ds_div_images, nf_ds_frame, bins)
ff_ds_image_bins = get_image_bins_and_ratios(ff_ds_div_images, ff_ds_frame, bins)


##################
# PRINT AND PLOT #
##################

print_bin_values('Near Field', nf_image_bins)
print_bin_values('Far Field', ff_image_bins)
print_bin_values('Near Field, Double Scrambler', nf_ds_image_bins)
print_bin_values('Far Field, Double Scrambler', ff_ds_image_bins)

side_by_side_plot(nf_cliped_div_images, 'Near Field Divided Views')
side_by_side_plot(ff_cliped_div_images, 'Far Field Divided Views')
side_by_side_plot(nf_ds_cliped_div_images, 'Near Field Divided Views With Double Scrambler')
side_by_side_plot(ff_ds_cliped_div_images, 'Far Field Divided Views With Double Scrambler')

plt.show()