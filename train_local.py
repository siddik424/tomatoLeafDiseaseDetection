"""
Tomato Leaf Disease Detection — VGG16 Transfer Learning
========================================================
LOCAL MAC VERSION — paths hardcoded for your machine
Dataset : train/ and val/ already on Desktop
Output  : models/ folder (h5 + class_labels.json)
"""

import os, json, shutil, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for Mac M1
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS  — your exact local paths
# ═══════════════════════════════════════════════════════════════════════════════
BASE         = Path('/Users/shoebakternafiz/Desktop/tomato-project')
ORIG_TRAIN   = BASE / 'train'
ORIG_VAL     = BASE / 'val'
SPLIT_DIR    = BASE / 'split_dataset'   # will be created automatically
MODEL_DIR    = BASE / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / 'tomato_vgg16_best.keras'
FINAL_H5_PATH   = MODEL_DIR / 'tomato_disease_vgg16.h5'
LABELS_PATH     = MODEL_DIR / 'class_labels.json'
LOG_PATH        = MODEL_DIR / 'training_log.csv'

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16       # reduced from 32 — safer for Mac RAM
EPOCHS_P1   = 15
EPOCHS_P2   = 20
LR_P1       = 1e-4
LR_P2       = 1e-5
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

print(f'TensorFlow  : {tf.__version__}')
print(f'Train dir   : {ORIG_TRAIN}')
print(f'Val dir     : {ORIG_VAL}')
print(f'Model dir   : {MODEL_DIR}')
print(f'Batch size  : {BATCH_SIZE}')
print(f'Image size  : {IMAGE_SIZE}')
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Collect all images
# ═══════════════════════════════════════════════════════════════════════════════
exts = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG')
class_dirs  = sorted([d for d in ORIG_TRAIN.iterdir() if d.is_dir()])
CLASS_NAMES = [d.name for d in class_dirs]
NUM_CLASSES = len(CLASS_NAMES)

print(f'Found {NUM_CLASSES} classes:')
all_images_by_class = {}
for cdir in class_dirs:
    imgs = []
    for ext in exts:
        imgs += list(cdir.glob(ext))
    val_cdir = ORIG_VAL / cdir.name
    if val_cdir.exists():
        for ext in exts:
            imgs += list(val_cdir.glob(ext))
    all_images_by_class[cdir.name] = [str(p) for p in imgs]
    print(f'  {cdir.name:<55} {len(imgs):>6,} images')

total = sum(len(v) for v in all_images_by_class.values())
print(f'\nTotal images : {total:,}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Stratified 70/15/15 split
# ═══════════════════════════════════════════════════════════════════════════════
print('\nBuilding 70/15/15 split...')

# Clear old split if exists
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for split in ('train', 'val', 'test'):
    for cls in CLASS_NAMES:
        (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

split_counts = {'train': {}, 'val': {}, 'test': {}}
for cls in CLASS_NAMES:
    imgs = all_images_by_class[cls]
    random.Random(SEED).shuffle(imgs)
    train_imgs, temp = train_test_split(
        imgs, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED
    )
    val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=SEED)
    for split_name, split_imgs in [
        ('train', train_imgs), ('val', val_imgs), ('test', test_imgs)
    ]:
        dest = SPLIT_DIR / split_name / cls
        for src in split_imgs:
            shutil.copy2(src, dest / Path(src).name)
        split_counts[split_name][cls] = len(split_imgs)

df = pd.DataFrame(split_counts).T
df['TOTAL'] = df.sum(axis=1)
print(f"Train : {df.loc['train','TOTAL']:,}")
print(f"Val   : {df.loc['val','TOTAL']:,}")
print(f"Test  : {df.loc['test','TOTAL']:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data generators
# ═══════════════════════════════════════════════════════════════════════════════
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
eval_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_set = train_gen.flow_from_directory(
    SPLIT_DIR / 'train', target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED
)
val_set = eval_gen.flow_from_directory(
    SPLIT_DIR / 'val', target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_set = eval_gen.flow_from_directory(
    SPLIT_DIR / 'test', target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

NUM_CLASSES  = len(train_set.class_indices)
CLASS_LABELS = list(train_set.class_indices.keys())
SHORT_LABELS = [c.replace('Tomato___', '').replace('_', ' ') for c in CLASS_LABELS]

print(f'\nTrain : {train_set.n:,} images  |  Val : {val_set.n:,}  |  Test : {test_set.n:,}')
print(f'Classes : {NUM_CLASSES}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Class weights
# ═══════════════════════════════════════════════════════════════════════════════
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_set.classes),
    y=train_set.classes
)
class_weight_dict = dict(zip(np.unique(train_set.classes), weights))
print('\nClass weights computed.')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Build model
# ═══════════════════════════════════════════════════════════════════════════════
print('\nBuilding VGG16 model...')

base_model = VGG16(
    input_shape=IMAGE_SIZE + (3,),
    weights='imagenet',
    include_top=False
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x);          x = BatchNormalization()(x); x = layers.Activation('relu')(x); x = Dropout(0.5)(x)
x = Dense(256)(x);          x = BatchNormalization()(x); x = layers.Activation('relu')(x); x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

trainable_p     = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_p = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f'Trainable params     : {trainable_p:,}')
print(f'Non-trainable params : {non_trainable_p:,}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Phase 1: Train head only
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' PHASE 1 — Training head (VGG16 frozen)')
print('='*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_P1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    ModelCheckpoint(str(BEST_MODEL_PATH), monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-8, verbose=1),
    CSVLogger(str(LOG_PATH), append=False)
]

history_p1 = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS_P1,
    callbacks=callbacks_p1,
    class_weight=class_weight_dict,
    verbose=1
)
print(f'Phase 1 best val_accuracy: {max(history_p1.history["val_accuracy"]):.4f}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Phase 2: Fine-tune block5
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print(' PHASE 2 — Fine-tuning VGG16 block5')
print('='*60)

unfreeze = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        unfreeze = True
    layer.trainable = unfreeze

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_P2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(str(BEST_MODEL_PATH), monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-9, verbose=1),
    CSVLogger(str(LOG_PATH), append=True)
]

history_p2 = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS_P2,
    callbacks=callbacks_p2,
    class_weight=class_weight_dict,
    verbose=1
)
print(f'Phase 2 best val_accuracy: {max(history_p2.history["val_accuracy"]):.4f}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Evaluate on test set
# ═══════════════════════════════════════════════════════════════════════════════
print('\nLoading best model and evaluating on test set...')
best_model = load_model(str(BEST_MODEL_PATH))

test_set.reset()
test_loss, test_acc = best_model.evaluate(test_set, verbose=1)
print(f'\nTest Accuracy : {test_acc*100:.2f}%')
print(f'Test Loss     : {test_loss:.4f}')

test_set.reset()
y_pred_probs = best_model.predict(test_set, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_set.classes
y_true_bin   = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

kappa    = cohen_kappa_score(y_true, y_pred)
mcc      = matthews_corrcoef(y_true, y_pred)
f1_mac   = f1_score(y_true, y_pred, average='macro')
f1_wt    = f1_score(y_true, y_pred, average='weighted')
auc_macro = roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')
top3_acc = sum(
    y_true[i] in np.argsort(y_pred_probs[i])[-3:]
    for i in range(len(y_true))
) / len(y_true)

print('\n' + '='*60)
print('CLASSIFICATION REPORT')
print('='*60)
print(classification_report(y_true, y_pred, target_names=SHORT_LABELS, digits=4))

metrics_summary = {
    'Test Accuracy'       : f'{test_acc*100:.2f}%',
    'Top-3 Accuracy'      : f'{top3_acc*100:.2f}%',
    'Macro F1-Score'      : f'{f1_mac:.4f}',
    'Weighted F1-Score'   : f'{f1_wt:.4f}',
    'AUC-ROC (macro OvR)' : f'{auc_macro:.4f}',
    "Cohen's Kappa"       : f'{kappa:.4f}',
    'MCC'                 : f'{mcc:.4f}',
    'Test Loss'           : f'{test_loss:.4f}',
}
print('\nSUMMARY METRICS')
for k, v in metrics_summary.items():
    print(f'  {k:<30} {v}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Save everything
# ═══════════════════════════════════════════════════════════════════════════════
best_model.save(str(FINAL_H5_PATH))
print(f'\nH5 model saved  : {FINAL_H5_PATH}')

labels_payload = {
    'class_indices' : train_set.class_indices,
    'idx_to_class'  : {str(v): k for k, v in train_set.class_indices.items()},
    'num_classes'   : NUM_CLASSES,
    'image_size'    : list(IMAGE_SIZE),
    'preprocessing' : 'vgg16_preprocess_input',
    'test_metrics'  : metrics_summary,
}
with open(str(LABELS_PATH), 'w') as f:
    json.dump(labels_payload, f, indent=2)
print(f'Labels saved    : {LABELS_PATH}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Save training plots
# ═══════════════════════════════════════════════════════════════════════════════
full_hist = {}
for key in history_p1.history:
    full_hist[key] = history_p1.history[key] + history_p2.history[key]
n_p1 = len(history_p1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in [(axes[0],'loss','Loss'), (axes[1],'accuracy','Accuracy')]:
    ax.plot(full_hist[metric],          label=f'Train', color='#e74c3c', lw=2)
    ax.plot(full_hist[f'val_{metric}'], label=f'Val',   color='#2ecc71', lw=2)
    ax.axvline(x=n_p1-0.5, color='#3498db', linestyle='--', lw=1.5, label='Phase 1→2')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('VGG16 Training History', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
print(f'Plot saved      : {MODEL_DIR}/training_history.png')

# Confusion matrix
cm_mat  = confusion_matrix(y_true, y_pred)
cm_norm = cm_mat.astype('float') / cm_mat.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(cm_mat,  annot=True, fmt='d',   cmap='YlOrRd',
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=axes[0])
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=axes[1])
axes[0].set_title('Confusion Matrix (Counts)',     fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalised)', fontweight='bold')
for ax in axes:
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f'Plot saved      : {MODEL_DIR}/confusion_matrix.png')

with open(MODEL_DIR / 'test_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print('\n' + '='*60)
print(' TRAINING COMPLETE')
print(f' Model  → {FINAL_H5_PATH}')
print(f' Labels → {LABELS_PATH}')
print('='*60)
