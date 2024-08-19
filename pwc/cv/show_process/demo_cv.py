import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from os import path
from image_processes import *
from shape_analysis import *
import matplotlib.cm as cm

contour_colour = (64, 35, 12) # BGR tuple for "#0c2340"
line_colour = (112, 90, 73) # BGR tuple for "#495a70" # dark gray used in poster
line_colour = (13, 68, 212) # BGR tuple for orange
linewidth = 5
dash_length = 20
gap_length = 15
line_offset = 1

image = 'ISIC_0062770.JPG'
image = path.join(path.expanduser("~"), "win_home", "Documents", "conferences", "SAPRF", "poster-images", image)
image = cv2.imread(image)
font = FontProperties(fname=None)
font_colour = "#0c2340" # the hex code for contour_colour
font_size = 24

plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
# 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

# trim edges -- some images have icons in corners
h, w, _ = image.shape
trim_pct = 10
trim_pixels = int(min(w, h) * (trim_pct /100))
image = image[trim_pixels:h-trim_pixels, trim_pixels:w-trim_pixels]

original = image.copy()
image_sym = image.copy()
image_bor = image.copy()
image_col = image.copy()
image_dia = image.copy()
image_con = image.copy() # original with contour outlined

image = hair_rm(image)
image = colour_cluster(image, n_clusters=5)
image = my_clahe(image)
mask, contour, contoured_image = segment(image)


# Apply mask
bool_mask = mask.astype(bool)
# Set the pixels specified by the mask to white (255, 255, 255 for RGB)
image_col[~bool_mask] = [255, 255, 255]  # You may adjust the color as needed
image_col = colour_cluster(image_col, 5)
image_col[~bool_mask] = [255, 255, 255]  # You may adjust the color as needed

# Calculate the perimeter of the lesion contour
perimeter = cv2.arcLength(contour, True)
# Calculate the area of the lesion contour
area = cv2.contourArea(contour)
# Calculate the radius of the circle with the same perimeter as the contour
radius = int(perimeter / (2 * np.pi))

# get the central point of the contour
M = cv2.moments(contour)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
# Find the maximum and minimum y positions of the contour
minY = min(contour[:, 0, 1])
maxY = max(contour[:, 0, 1])
minX = min(contour[:, 0, 0])
maxX = max(contour[:, 0, 0])

newCenterY = (minY + maxY) // 2
newCenterX = (minX + maxX) // 2

# Draw the lesion contour
cv2.drawContours(image_sym, [contour], 0, contour_colour, linewidth)
cv2.drawContours(image_bor, [contour], 0, contour_colour, linewidth)
cv2.drawContours(image_col, [contour], 0, contour_colour, linewidth)
cv2.drawContours(image_dia, [contour], 0, contour_colour, linewidth)
cv2.drawContours(image_con, [contour], 0, contour_colour, linewidth)

# Draw the circle with the same perimeter from the (self-identified) center of the contour
cv2.circle(image_bor, (newCenterX, newCenterY), radius-line_offset, contour_colour, linewidth)
cv2.circle(image_bor, (newCenterX, newCenterY), radius+line_offset, contour_colour, linewidth)
cv2.circle(image_bor, (newCenterX, newCenterY), radius, line_colour, linewidth)

######## Diameter
center, (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
angle_rad = np.deg2rad(angle)
start_point= (int(center[0] - 0.5 * major_axis * np.cos(angle_rad)),
              int(center[1] - 0.5 * major_axis * np.sin(angle_rad)))
end_point = (int(center[0] + 0.5 * major_axis * np.cos(angle_rad)),
              int(center[1] + 0.5 * major_axis * np.sin(angle_rad)))
cv2.line(image_dia, start_point, end_point, contour_colour, linewidth*2) 
cv2.line(image_dia, start_point, end_point, line_colour, linewidth)


# Crop image to center the lesion and circle
# Define the buffer size as a percentage of the circle radius
buffer_percentage = 0.2  # 10% buffer
# Calculate the buffer size in pixels
buffer_size = int(radius * buffer_percentage)

# Calculate the new width and height of the cropped region with the buffer
newWidth = newHeight = 2 * (radius + buffer_size)

# Calculate the top-left corner of the cropped region
cropX = cX - (radius + buffer_size)
cropY = newCenterY - (radius + buffer_size)

# Crop the image to the specified region
cropped_sym = image_sym[cropY:cropY+newHeight, cropX:cropX+newWidth]
cropped_bor = image_bor[cropY:cropY+newHeight, cropX:cropX+newWidth]
cropped_col = image_col[cropY:cropY+newHeight, cropX:cropX+newWidth]
cropped_dia = image_dia[cropY:cropY+newHeight, cropX:cropX+newWidth]
cropped_og = original[cropY:cropY+newHeight, cropX:cropX+newWidth]
cropped_con = image_con[cropY:cropY+newHeight, cropX:cropX+newWidth]

# compactness = np.divide(4 * np.pi * area, perimeter**2) 
minY = min(contour[:, 0, 1]) - cropY
maxY = max(contour[:, 0, 1]) - cropY
minX = min(contour[:, 0, 0]) - cropX
maxX = max(contour[:, 0, 0]) - cropX
newCenterY = (minY + maxY) // 2
newCenterX = (minX + maxX) // 2


####### asymmetry
# Draw the dashed vertical line
current_length = 0
while current_length < (maxY - minY):
    if current_length % (dash_length + gap_length) < dash_length:
        cv2.line(cropped_sym, (newCenterX-line_offset, minY + current_length - line_offset), (newCenterX+line_offset, minY + current_length + line_offset), contour_colour, linewidth+(line_offset*2))
    current_length += 1

current_length = 0
while current_length < (maxY - minY):
    if current_length % (dash_length + gap_length) < dash_length:
        cv2.line(cropped_sym, (newCenterX, minY + current_length), (newCenterX, minY + current_length), line_colour, linewidth)
    current_length += 1

# mask_center = mask.shape[0] // 2

# centered = center_segment(mask, contour)
# # rotated = align_major_axis(mask, contour)

# # L = rotated
# # Lx = cv2.flip(L, 0)
# # dLx = (L + Lx) % 2

# # Ly = cv2.flip(L, 1)
# # dLy = (L + Ly) % 2

# # # symmetry is the ratio of the summed difference to the total mask area: 1 = perfect symmetry
# # x_symmetry = np.round(np.divide(np.sum(dLx), np.sum(Lx)), 4)
# # y_symmetry = np.round(np.divide(np.sum(dLy), np.sum(Ly)), 4)

# flipped = cv2.flip(centered, 0)
# overlayed = cv2.bitwise_xor(centered, flipped)
# # overlayed[mask_center:, :] = 0

# ## center image
# M = cv2.moments(centered)
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])
# image_center = (overlayed.shape[1] // 2, overlayed.shape[0] // 2)
# displacement = (image_center[0] - cX, image_center[1] - cY)
# # Use the displacement to recenter the image
# centered_sym = np.roll(overlayed, displacement, axis=(0, 1))
# centered_sym = centered_sym[80:-80, 80:-80]

######## Colour


########## plotting

def format_axes(axes):
    for ax in axes:
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.axis('off')

def format_axis(ax):
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.axis('off')


fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(cropped_sym, cv2.COLOR_BGR2RGB))
axs[0].set_title('Asymmetry', color=font_colour, fontproperties=font, fontsize=font_size)

axs[1].imshow(cv2.cvtColor(cropped_bor, cv2.COLOR_BGR2RGB))
axs[1].set_title('Border Irregularity', color=font_colour, fontproperties=font, fontsize=font_size)

axs[2].imshow(cv2.cvtColor(cropped_col, cv2.COLOR_BGR2RGB))
axs[2].set_title('Colour Variance', color=font_colour, fontproperties=font, fontsize=font_size)

axs[3].imshow(cv2.cvtColor(cropped_dia, cv2.COLOR_BGR2RGB))
axs[3].set_title('Diameter', color=font_colour, fontproperties=font, fontsize=font_size)

format_axes(axs)
plt.tight_layout()
plt.savefig('ABCD-features.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


show_title = False
######## Individual images
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_og, cv2.COLOR_BGR2RGB))
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-original.pdf', format='pdf', dpi=600, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_con, cv2.COLOR_BGR2RGB))
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-contour.pdf', format='pdf', dpi=600, bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_sym, cv2.COLOR_BGR2RGB))
if show_title:
    axs.set_title('Asymmetry', color=font_colour, fontproperties=font, fontsize=font_size)
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-symmetry.pdf', format='pdf', dpi=600, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_bor, cv2.COLOR_BGR2RGB))
if show_title:
    axs.set_title('Border Irregularity', color=font_colour, fontproperties=font, fontsize=font_size)
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-border.pdf', format='pdf', dpi=600, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_col, cv2.COLOR_BGR2RGB))
if show_title:
    axs.set_title('Colour Variance', color=font_colour, fontproperties=font, fontsize=font_size)
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-colour.pdf', format='pdf', dpi=600, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(cv2.cvtColor(cropped_dia, cv2.COLOR_BGR2RGB))
if show_title:
    axs.set_title('Diameter', color=font_colour, fontproperties=font, fontsize=font_size)
format_axis(axs)
plt.tight_layout()
plt.savefig('demo-diameter.pdf', format='pdf', dpi=600, bbox_inches='tight')

