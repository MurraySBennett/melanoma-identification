import numpy as np

# workflow
# 1. data= pd.read_csv(data_path) 
# 2. data = ABC_aligned(data)
# 3. data = cv_btl_scale(data)

def ABC_aligned(data):
    data["sym"] = data[["x_sym", "y_sym"]].mean(axis=1)
    data["bor"] = 1 - data["compact"] 
    data["col"] = data["rms"]
    return data


def scale_x_to_y(x, y):
    x = np.array(x)
    y = np.array(y)
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    min_x, max_x = np.min(x[valid_indices]), np.max(x[valid_indices])
    min_y, max_y = np.min(y[valid_indices]), np.max(y[valid_indices])
    
    scaled_x = (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y
    return scaled_x


def cv_btl_scale(data, replace=True):
    if replace:
        data["sym"] = scale_x_to_y(data["sym"], data["pi_sym"])
        data["bor"] = scale_x_to_y(data["bor"], data["pi_bor"])
        data["col"] = scale_x_to_y(data["col"], data["pi_col"])
    else:
        data["sym_scaled"] = scale_x_to_y(data["sym"], data["pi_sym"])
        data["bor_scaled"] = scale_x_to_y(data["bor"], data["pi_bor"])
        data["col_scaled"] = scale_x_to_y(data["col"], data["pi_col"])
    return data



