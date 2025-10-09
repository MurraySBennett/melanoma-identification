from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..config import (FILES, PATHS)


# font = FontProperties(fname="Garamond BoldCondensed.ttf")
font = FontProperties()
FONT_COLOUR = "black" #"#0c2340" # the hex code for contour_colour
FONT_SIZE   = 14
LINE_COLOUR = "black"#"#D4440D"
AXIS_LABEL_FONT_SIZE = 18
plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
plt.rcParams['pdf.fonttype'] = 42

data_path   = FILES['btl_cv']
fig_path    = PATHS['figures']
data = pd.read_csv(data_path)

features    = ['pi_sym', 'pi_bor', 'pi_col']
# data        = data[['id'] + features]
data = data[features].dropna()
feature_labels = ["Asymmetry", "Border\nIrregularity", "Colour\nVariance"]


plt.figure(figsize=(8, 6))
bp = plt.boxplot(data, tick_labels=feature_labels, patch_artist=True, widths=0.6)
for p in bp['boxes']:
    p.set(facecolor='lightblue', edgecolor=LINE_COLOUR, linewidth=1.5)
for whisker in bp['whiskers']:
    whisker.set(color=LINE_COLOUR, linewidth=1.5)
for cap in bp['caps']:
    cap.set(color=LINE_COLOUR, linewidth=1.5)
for med in bp['medians']:
    med.set(color=LINE_COLOUR, linewidth=2)

plt.ylabel("BTL estimate", fontsize=AXIS_LABEL_FONT_SIZE)
plt.xticks(fontsize=AXIS_LABEL_FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.tight_layout()

plt.savefig(
    fig_path / "btl_boxPlot.pdf",
    format='pdf', dpi=600, bbox_inches='tight'
)
# plt.show()
plt.close()


fig, axs = plt.subplots(1, 3, figsize=(18,6))
for i, feature in enumerate(features):
    axs[i].scatter(range(len(data[feature])), data[feature])
    axs[i].set_title(feature_labels[i], fontsize=AXIS_LABEL_FONT_SIZE)
    if i == 0:
        axs[i].set_ylabel("BTL estimate", fontsize=AXIS_LABEL_FONT_SIZE)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_xlim(-3000, len(data[feature]))
    axs[i].set_xticks([])

    ax_inset = inset_axes(axs[i], width="15%", height="99%", loc='center left')
    ax_inset.boxplot(data[feature], patch_artist=True, widths=0.6)
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['left'].set_visible(False)
    ax_inset.spines['bottom'].set_visible(False)
    ax_inset.yaxis.set_visible(False)
    ax_inset.xaxis.set_visible(False)

# plt.tight_layout()
plt.show()
