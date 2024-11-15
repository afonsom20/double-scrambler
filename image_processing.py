from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# NEAR FIELD IMAGES
near_field_ref = fits.getdata("near field reference.fits")
near_field_left = fits.getdata("near field left.fits")
near_field_right = fits.getdata("near field right.fits")
near_field_top = fits.getdata("near field top.fits")
near_field_bottom = fits.getdata("near field bottom.fits")
near_field_clockwise = fits.getdata("near field bottom.fits")
near_field_counterclockwise = fits.getdata("near field bottom.fits")

# NEAR FIELD OPERATIONS
near_field_left_divided = np.divide(near_field_left, near_field_ref)
near_field_right_divided = np.divide(near_field_right, near_field_ref)
near_field_top_divided = np.divide(near_field_top, near_field_ref)
near_field_bottom_divided = np.divide(near_field_bottom, near_field_ref)
near_field_clockwise_divided = np.divide(near_field_clockwise, near_field_ref)
near_field_counterclockwise_divided = np.divide(near_field_counterclockwise, near_field_ref)

# FAR FIELD IMAGES
far_field_ref = fits.getdata("far field reference.fits")
far_field_left = fits.getdata("far field left.fits")
far_field_right = fits.getdata("far field right.fits")
far_field_top = fits.getdata("far field top.fits")
far_field_bottom = fits.getdata("far field bottom.fits")
far_field_clockwise = fits.getdata("far field angle 1.fits")
far_field_counterclockwise = fits.getdata("far field angle 2.fits")

# FAR FIELD OPERATIONS
far_field_left_divided = np.divide(far_field_left, far_field_ref)
far_field_right_divided = np.divide(far_field_right, far_field_ref)
far_field_top_divided = np.divide(far_field_top, far_field_ref)
far_field_bottom_divided = np.divide(far_field_bottom, far_field_ref)
far_field_clockwise_divided = np.divide(far_field_clockwise, far_field_ref)
far_field_counterclockwise_divided = np.divide(far_field_counterclockwise, far_field_ref)

# PLOTS
colormap = "viridis"

# Near Field Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Near Field Divided Views")

axes[0, 0].imshow(near_field_left_divided, cmap=colormap)
axes[0, 0].set_title("Left")

axes[0, 1].imshow(near_field_right_divided, cmap=colormap)
axes[0, 1].set_title("Right")

axes[0, 2].imshow(near_field_top_divided, cmap=colormap)
axes[0, 2].set_title("Top")

axes[1, 0].imshow(near_field_bottom_divided, cmap=colormap)
axes[1, 0].set_title("Bottom")

axes[1, 1].imshow(near_field_clockwise_divided, cmap=colormap)
axes[1, 1].set_title("Clockwise")

axes[1, 2].imshow(near_field_counterclockwise_divided, cmap=colormap)
axes[1, 2].set_title("Counterclockwise")



# Far Field Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Far Field Divided Views")

axes[0, 0].imshow(far_field_left_divided, cmap=colormap)
axes[0, 0].set_title("Left")

axes[0, 1].imshow(far_field_right_divided, cmap=colormap)
axes[0, 1].set_title("Right")

axes[0, 2].imshow(far_field_top_divided, cmap=colormap)
axes[0, 2].set_title("Top")

axes[1, 0].imshow(far_field_bottom_divided, cmap=colormap)
axes[1, 0].set_title("Bottom")

axes[1, 1].imshow(far_field_clockwise_divided, cmap=colormap)
axes[1, 1].set_title("Clockwise")

axes[1, 2].imshow(far_field_counterclockwise_divided, cmap=colormap)
axes[1, 2].set_title("Counterclockwise")



plt.show()