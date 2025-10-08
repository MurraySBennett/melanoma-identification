# %% Load libs
import os
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e) 
else:
    print("No GPUs available, using CPU.")
    
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


# %% Hyper parameters
    
model_name = 'B0'
WEIGHTS = None # or 'imagenet'
img_sizes  = dict(
    B0=224, B1=240, B2=260, B3=300, B4=380, B5=456, B6=528, B7=600
)
# IMG_SIZE   = img_sizes[model_name]
IMG_SIZE   = 224
BATCH_SIZE = 256#128
LR         = 1e-6#5

EPOCHS     = 100
patience   = 40

N_IMAGES = None # None = use all

# model accuracy saving thresholds
acc_thresholds = np.arange(0.55, 0.91, 0.05)
loss_threshold = 0.7
tolerance = 0.025


# Set Paths, read Data
base_dir = Path(os.getcwd()).parent.parent
image_dir= base_dir / 'images' / 'resized'
csv_path = base_dir / 'pwc' / 'data' / 'estimates' / 'btl_cv_data.csv'
fig_dir  = base_dir / 'melnet' / 'figures'
data_dir = base_dir / 'melnet' / 'data'
model_dir= base_dir / 'melnet' / 'models'

data = pd.read_csv(csv_path)
data['id'] = data['id'].apply(lambda x: x + '.JPG')
data['malignant'] = data['malignant'].astype(str)
if N_IMAGES is not None:
    data = data.head(min(N_IMAGES, len(data)))

dca_images = pd.read_csv(data_dir / "dca_images.txt")
dca_images = {img + ".JPG" for img in dca_images['id']}
is_dca = data['id'].isin(dca_images)


X = data['id']
y = data['malignant']


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed_value, stratify=y  # 80% train_val, 20% test
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=seed_value, stratify=y_train_val  # 75% of 80% is 60% train, 25% of 80% is 20% val
)

train_mask = ~data.iloc[data.index.isin(X_train.index)]['id'].isin(dca_images)
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

train_data = pd.DataFrame({'id': X_train_filtered, 'malignant': y_train_filtered}).reset_index()
val_data = pd.DataFrame({'id': X_val, 'malignant': y_val}).reset_index()
test_data = pd.DataFrame({'id': X_test, 'malignant': y_test}).reset_index()

data['split'] = 'unknown'
data.loc[data['id'].isin(X_train_filtered), 'split'] = 'train'
data.loc[data['id'].isin(X_val), 'split'] = 'val'
data.loc[data['id'].isin(X_test), 'split'] = 'test'
save_split = data[['id', 'split']]
save_split.to_csv(data_dir / "image_splits.csv", index=False)


# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Training data")
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=image_dir,
    x_col='id',
    y_col='malignant',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    drop_remainder=True
)

print("Validation data")
validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=image_dir,
    x_col='id',
    y_col='malignant',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    drop_remainder=True
)

print("Test data")
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=image_dir,
    x_col='id',
    y_col='malignant',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    drop_remainder=True
)

# %% Display training images
# print("Train data shape: ", train_data.shape)
# print(val_data.head())
# print(sum(val_data.malignant == '1'), sum(val_data.malignant == '0'))
def display_images(generator, num_images=5, title=""):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # 1 row, num_images columns

    for i in range(num_images):
        try: # Try except block to handle potential StopIteration error if the generator is exhausted.
            x, y = next(generator)  # Get a batch
            image = x[0]  # Get the first image from the batch
            label = y[0]  # Get the label for the first image

            ax = axes[i]  # Get the current subplot axes
            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        except StopIteration:
            print("Generator is exhausted. Fewer images displayed.")
            break # Exit the loop if the generator is exhausted.
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
display_images(train_generator, num_images=6, title="Training")
display_images(validation_generator, num_images=6, title="Validation")
display_images(test_generator, num_images=6, title="Test")

# %% Custom callbacks
# Save model at accuracy

def list_to_float(var):
    if isinstance(var, list):
        return var[-1]
    else:
        return var
    
    
class SaveModelAtThresholds(Callback):
    def __init__(self, acc_thresholds, loss_threshold, tolerance=0.02, save_dir='saved_models', accuracy_gap=0.1):
        super(SaveModelAtThresholds, self).__init__()
        self.acc_thresholds = sorted(list(acc_thresholds))
        self.loss_threshold = loss_threshold
        self.tolerance = tolerance
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.saved_thresholds = {}
        self.accuracy_gap = accuracy_gap

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        train_accuracy=logs.get('accuracy')

        if val_loss is not None and val_accuracy is not None and train_accuracy is not None:
            val_loss = list_to_float(val_loss)
            val_accuracy = list_to_float(val_accuracy)
            train_accuracy = list_to_float(train_accuracy)

            if val_loss < self.loss_threshold:
                if abs(train_accuracy - val_accuracy) <= self.accuracy_gap:
                    for acc_threshold in self.acc_thresholds:
                        acc_threshold = np.round(acc_threshold, 3)
                        if acc_threshold - self.tolerance <= val_accuracy <= acc_threshold + self.tolerance:
                            if acc_threshold not in self.saved_thresholds or val_loss < self.saved_thresholds.get(acc_threshold, {}).get('loss', float('inf')):
                                old_model_path = None
                                if acc_threshold in self.saved_thresholds:
                                    old_model_path = os.path.join(self.save_dir, f"weights_acc_{int(self.saved_thresholds[acc_threshold]['accuracy'] * 1000)}.h5")

                                self.saved_thresholds[acc_threshold] = {
                                    'accuracy': val_accuracy,
                                    'loss': val_loss
                                }
                                model_path = os.path.join(self.save_dir, f"weights_acc_{int(val_accuracy * 1000)}.h5")

                                if old_model_path and os.path.exists(old_model_path):
                                    os.remove(old_model_path)

                                self.model.save_weights(model_path)

                                logs['saved_acc'] = acc_threshold
                                print(f"\nSaved/updated model at {int(val_accuracy * 1000) / 10}% accuracy and loss < {self.loss_threshold}.")
                else:
                    print(f"\nSkipping save: Accuracy gap ({abs(train_accuracy - val_accuracy):.3f}) exceeds threshold ({self.accuracy_gap}).")



reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=patience//10,
    min_lr=1e-6
)
# early_stopping has no effect when running the EPOCH loop the way I am
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)


save_model_acc = SaveModelAtThresholds(
    acc_thresholds=acc_thresholds,
    loss_threshold=loss_threshold,
    tolerance=tolerance,
    save_dir=model_dir,
    accuracy_gap=0.1
)

callbacks = [save_model_acc]


# %% Build model

# model_mapping = {
#     "B0": EfficientNetB0,
#     "B5": EfficientNetB5
# }

def build_model(num_classes=1, trainable=False):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(
        include_top=False,
        input_tensor=inputs,
        weights='imagenet'
    )
    base_model.trainable = trainable

    if ~trainable:
        for layer in base_model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="pred")(x)

    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")]
    )
    return model


# def unfreeze_model(model):
#     for layer in model.layers[-20:]:
#         if not isinstance(layer, layers.BatchNormalization):
#             layer.trainable = True
#     optimizer = keras.optimizers.Adam(learning_rate=1e-5)
#     model.compile(
#         optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")]
#     )


#     if model_name in model_mapping:
#         model = model_mapping[model_name](
#             weights=WEIGHTS,
#             include_top = True,
#             classes=2,
#             input_shape=(IMG_SIZE, IMG_SIZE, 3)
#         )
#     else:
#         print("No valid model!")

# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=LR),
#     loss='binary_crossentropy',
#     metrics=['accuracy', AUC(name='auc')],
#     experimental_run_tf_function=False
# )



# %% Training

best_val_loss = float('inf')
best_model_weights = None

all_metrics = []

model = build_model(trainable=True)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=1,
        callbacks=callbacks
    )
    
    saved_acc = None;
    for callback in callbacks:
        if isinstance(callback, SaveModelAtThresholds):
            callback.on_epoch_end(0, history.history)
            saved_acc = history.history.get('saved_acc', None)
   
    all_metrics.append({
        'epoch': epoch + 1,
        'train_acc': np.round(history.history['accuracy'][0], 3),
        'val_acc': np.round(history.history['val_accuracy'][0], 3),
        'train_loss': np.round(history.history['loss'][0], 3),
        'val_loss': np.round(history.history['val_loss'][0], 3),
        'train_auc': np.round(history.history['auc'][0], 3),
        'val_auc': np.round(history.history['val_auc'][0], 3),
        'saved_acc': saved_acc
    })
    
    val_loss = history.history['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.get_weights()
        model.save_weights(model_dir / 'best_loss_model.h5')
        print(f"Epoch {epoch+1}: New best val_loss = {best_val_loss:.3f}. Model weights saved.")

model.set_weights(best_model_weights)
model.save(model_dir / 'melnet')


# %% Save metrics

metrics = pd.DataFrame(all_metrics)
metrics.to_csv(data_dir / 'training_metrics.csv', index=False)
averages = pd.DataFrame({
    'epoch': metrics['epoch'].unique(),
    'acc_train': metrics.groupby('epoch')['train_acc'].mean().values,
    'acc_val': metrics.groupby('epoch')['val_acc'].mean().values,
    'loss_train': metrics.groupby('epoch')['train_loss'].mean().values,
    'loss_val': metrics.groupby('epoch')['val_loss'].mean().values,
    'auc_train': metrics.groupby('epoch')['train_auc'].mean().values,
    'auc_val': metrics.groupby('epoch')['val_auc'].mean().values
})


# %% Plot training and validation metrics

try:
    metrics
except NameError:
    metrics = pd.read_csv(data_dir / 'training_metrics.csv')

def convert_to_numeric(metrics):
    columns_to_convert = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'train_auc', 'val_auc']
    for col in columns_to_convert:
        metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
    saved_acc_numeric = []
    for val in metrics['saved_acc']:
        if isinstance(val, str):
            try:
                numeric_val = float(val.split('[')[-1].split(']')[0])
                saved_acc_numeric.append(numeric_val)
            except (ValueError, IndexError):
                saved_acc_numeric.append(None)
        else:
            saved_acc_numeric.append(val)
    metrics['saved_acc'] = saved_acc_numeric
    return metrics


def plot_saved_acc(ax, metrics, key):
    saved_acc_values = metrics['saved_acc'][pd.notna(metrics['saved_acc'])].unique()
    for acc_val_num in saved_acc_values:
        filtered_metrics = metrics[metrics['saved_acc'] == acc_val_num]
        if not filtered_metrics.empty:
            best_epoch = filtered_metrics['epoch'].iloc[-1]
            best_val_acc = filtered_metrics[key].iloc[-1]
            ax.plot(best_epoch, best_val_acc, 'ko')
        
metrics = convert_to_numeric(metrics)
epochs_range = metrics['epoch'].values

plt.figure(figsize=(12, 8))

# Accuracy
ax1 = plt.subplot(2, 2, 1)
ax1.plot(epochs_range, metrics['train_acc'], label='Train')
ax1.plot(epochs_range, metrics['val_acc'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')
ax1.set_xlim(1, epochs_range.max())
ax1.set_ylim(0, 1)
plot_saved_acc(ax1, metrics, 'val_acc')

# Loss
ax2 = plt.subplot(2, 2, 2)
ax2.plot(epochs_range, metrics['train_loss'], label='Train')
ax2.plot(epochs_range, metrics['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_xlim(1, epochs_range.max())
plot_saved_acc(ax2, metrics, 'val_loss')

# AUC
ax3 = plt.subplot(2, 2, (3, 4))
ax3.plot(epochs_range, metrics['train_auc'], label='Train')
ax3.plot(epochs_range, metrics['val_auc'], label='Validation')
ax3.set_title('Model ROC-AUC')
ax3.set_ylabel('ROC-AUC')
ax3.set_xlabel('Epoch')
ax3.set_xlim(1, epochs_range.max())
ax3.set_ylim(0, 1)
plot_saved_acc(ax3, metrics, 'val_auc')

plt.tight_layout()
plt.savefig(fig_dir / "melnet_training_metrics.png")
plt.show()


# %% Test models

try:
    model
except NameError:
    model = tf.keras.models.load_model(model_dir / 'melnet')
    print("Model loaded.")

model.load_weights(model_dir / 'best_loss_model.h5')


def evaluate_model(model, test_generator):
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator)
    y_pred = (y_pred_proba > 0.5).astype(int)

    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'auc': test_auc,
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp)
    }

results_df = pd.DataFrame(columns=['model', 'loss', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'specificity'])

best_model_metrics = evaluate_model(model, test_generator)
best_model_metrics['model'] = 'best_model'
results_df = pd.concat([results_df, pd.DataFrame([best_model_metrics])], ignore_index=True)

weights = list(model_dir.glob("weights*.h5"))[:-1]
for w in weights:
    model.load_weights(w)
    sub_optimal_metrics = evaluate_model(model, test_generator)
    sub_optimal_metrics['model'] = w.stem
    results_df = pd.concat([results_df, pd.DataFrame([sub_optimal_metrics])], ignore_index=True)
results_df.to_csv(data_dir / 'test_metrics.csv', index=False)


metrics_to_plot = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'specificity']
models = results_df['model'].tolist()
num_models = len(models)
num_metrics = len(metrics_to_plot)

metric_values = {metric: results_df[metric].tolist() for metric in metrics_to_plot}
bar_width = 0.1
x = range(num_models)

plt.figure(figsize=(15, 8))
for i, metric in enumerate(metrics_to_plot):
    positions = [pos + i * bar_width for pos in x]
    label = metric.upper() if metric == 'auc' else metric.capitalize()
    plt.bar(positions, metric_values[metric], width=bar_width, label=label, edgecolor='black', linewidth=1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks([pos + bar_width * (num_metrics - 1) / 2 for pos in x], models, rotation=0, ha='right')
plt.xlabel('Model calcWeights')
plt.ylabel('Metric Value')
plt.title('Model Performance Comparison')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(fig_dir / 'test_metrics_modelgrp.png')
plt.show()


x = range(num_metrics)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(figsize=(15, 8))
for i, m in enumerate(models):
    model_metrics = results_df[results_df['model'] == m]
    positions = [pos + i * bar_width for pos in x]
    for j, metric in enumerate(metrics_to_plot):
        label = m if j == 0 else ""
        plt.bar(positions[j], model_metrics[metric], width=bar_width, label=label, color=colors[i % len(colors)], edgecolor='black', linewidth=1)

plt.xticks([pos + bar_width * (num_models - 1) / 2 for pos in x], [metric.upper() if metric == 'auc' else metric.capitalize() for metric in metrics_to_plot])
plt.xlabel('Metric')
plt.ylabel('Metric Value')
plt.title('Model Performance Comparison')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(fig_dir / 'test_metrics_metgrp.png')
plt.show()

