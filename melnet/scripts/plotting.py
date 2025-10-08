import matplotlib.pyplot as plt


hex_colours = dict(
    UTSA_BLUE   = "#0C2340",
    UTSA_ORANGE = "#D3A80C",
    PINK        = "#D04D92",
)

def pltParams():
    plt.rcParams['text.antialiased'] = True
    #plt.rcParams['font.family'] = font.get_name()
    # 0 is no compression = max quality/max size, 9 is max compression = low quality/min size
    plt.rcParams['pdf.compression'] = 3 # (embed all fonts and images)
    plt.rcParams['pdf.fonttype'] = 42

def rmSpines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



