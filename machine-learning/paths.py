from os import path
import pandas as pd

home = path.join(path.expanduser('~'), 'win_home', 'melanoma-identification')
paths = dict(
    home        = home,
    images      = path.join(home, "images", "resized"),
    masks       = path.join(home, "images", "segmented", "masks"),
    data        = path.join(home, "machine-learning", "data")
)

# def load_data(**data):
#     df = 
#     for k, v in data:
#         df = pd.csv_read(v, header=1)

