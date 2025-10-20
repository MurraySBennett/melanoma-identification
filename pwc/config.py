import os
from pathlib import Path


# Melanoma identification directory
HOME_PATH = Path(__file__).resolve().parent.parent

PATHS = dict(
    home=HOME_PATH,
    images=HOME_PATH / "images" / "resized",
    masks=HOME_PATH / "images" / "masks",
    segmented=HOME_PATH / "images" / "segmented",
    cv=HOME_PATH / "pwc" / "cv",
    estimates=HOME_PATH / "pwc" / "data" / "estimates",
    raw_data = HOME_PATH / "pwc" / "data" / "raw",
    clean_data=HOME_PATH / "pwc" / "data" / "cleaned",
    figures = HOME_PATH / "pwc" / "figures",
    svm_models=HOME_PATH / "pwc" / "models",

)

FILES = dict(
    metadata=HOME_PATH / "images" / "ISIC-database" / "metadata.csv",
    malignant_ids = PATHS['cv'] / "01_feature-analysis" / "malignant_ids.txt",
    cv_shape=PATHS['cv'] / "01_feature-analysis" / f"cv_shape{'_reproduced' if REPRODUCE else ''}.txt",
    cv_colour=PATHS['cv'] / "01_feature-analysis" / f"cv_colour{'_reproduced' if REPRODUCE else ''}.txt",
    cv_data=PATHS['estimates'] / "cv-data.csv",
    btl_data=PATHS['estimates'] / "btl_data.csv",
    btl_cv = PATHS['estimates'] / "btl_cv_data.csv"
)


for key in ['masks', 'segmented', 'estimates', 'raw_data', 'clean_data', 'figures', 'svm_models']:
    path = PATHS[key]
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating {path.name}: {e}")
    
for key in ['metadata', 'malignant_ids']:
    file = FILES[key]
    if not file.exists():
        print(f"I can't find the metadata file: {file}. This is an unhelpful message, but we will need it.")

if not PATHS['images'].is_dir():
    print(f"I can't find the images directory: {PATHS['images']}. We definitely need it.")

