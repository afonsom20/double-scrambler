from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def div_by_ref(images):
    """Divide images by the reference"""
    return {k: v / images['ref'] for k,v in images.items() if k != 'ref'}

def frame_image(images, bright_threshold):
    """Find the center of the object and clip a frame around it"""
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
    xlims = frame['xlims']
    ylims = frame['ylims']
    return image[ylims[0]:ylims[1], xlims[0]:xlims[1]]

def clip_images(images, frame):
    """Given a frame, clip the images"""
    return {k: clip(v, frame) for k, v in images.items()}

disposition = {
    (0,0): 'left',
    (0,1): 'right',
    (1,0): 'top',
    (1,1): 'bottom',
    (2,0): 'clockwise',
    (2,1): 'counterclockwise'
}

def side_by_side_plot(images, title):
    """Plot the images of various angles side by side"""
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    fig.suptitle(title)
    for k, v in disposition.items():
        axes[k].imshow(images[v], cmap=colormap)
        axes[k].set_title(v.title())

# PLOTS
colormap = "gray_r"

# Parameters
nf_bright_threshold = 500
ff_bright_threshold = 250
nf_ds_bright_threshold = 250
ff_ds_bright_threshold = 250

# NEAR FIELD IMAGES
nf_images = {
    'ref': fits.getdata("near field reference.fits"),
    'left': fits.getdata("near field left.fits"),
    'right': fits.getdata("near field right.fits"),
    'top': fits.getdata("near field top.fits"),
    'bottom': fits.getdata("near field bottom.fits"),
    'clockwise': fits.getdata("near field bottom.fits"),
    'counterclockwise': fits.getdata("near field bottom.fits"),
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

# CLIP AND PLOT NEAR FIELD
nf_frame = frame_image(nf_images, nf_bright_threshold)
nf_cliped_div_images = clip_images(nf_div_images, nf_frame)
side_by_side_plot(nf_cliped_div_images, 'Near Field Divided Views')

# CLIP AND PLOT FAR FIELD
ff_frame = frame_image(ff_images, ff_bright_threshold)
ff_cliped_div_images = clip_images(ff_div_images, ff_frame)
side_by_side_plot(ff_cliped_div_images, 'Far Field Divided Views')

# CLIP AND PLOT DOUBLE SCRAMBLER NEAR FIELD
nf_ds_frame = frame_image(nf_ds_images, nf_ds_bright_threshold)
nf_ds_cliped_div_images = clip_images(nf_ds_div_images, nf_ds_frame)
side_by_side_plot(nf_ds_cliped_div_images, 'Near Field Divided Views With Double Scrambler')

# CLIP AND PLOT DOUBLE SCRAMBLER FAR FIELD
ff_ds_frame = frame_image(ff_ds_images, ff_ds_bright_threshold)
ff_ds_cliped_div_images = clip_images(ff_ds_div_images, ff_ds_frame)
side_by_side_plot(ff_ds_cliped_div_images, 'Far Field Divided Views With Double Scrambler')

plt.show()
