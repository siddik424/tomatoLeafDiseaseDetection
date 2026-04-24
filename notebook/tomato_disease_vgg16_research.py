"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Tomato Leaf Disease Detection Using VGG16 Transfer Learning                ║
║  Dataset : PlantVillage — Tomato (kaustubhb999/tomatoleaf, Kaggle)          ║
║  Model   : VGG16 (ImageNet) — Two-Phase Transfer Learning                   ║
║  Classes : 10 (9 diseases + Healthy)  |  Split: 70/15/15                   ║
║  Target  : Q1 Journal Publication                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE (Kaggle):
    1. Add dataset: Settings → Add data → kaustubhb999/tomatoleaf
    2. Enable GPU accelerator
    3. Run all cells

USAGE (Local):
    Place dataset at  ../tomato/train  and  ../tomato/val
    pip install tensorflow scikit-learn matplotlib seaborn pillow pandas
    python tomato_disease_vgg16_research.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
import os, json, shutil, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from glob import glob
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, load_img
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f'TensorFlow version : {tf.__version__}')
print(f'GPU available      : {len(tf.config.list_physical_devices("GPU")) > 0}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU device         : {gpus[0].name}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION & PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
CANDIDATE_ROOTS = [
    Path('/kaggle/input/tomatoleaf'),
    Path('/kaggle/input/tomato-leaf-disease-dataset'),
    Path.cwd().parent / 'tomato',
    Path.cwd().parent / 'data',
    Path.cwd() / 'data',
]

SPLIT_NAMES = {
    'train': ['train', 'Train', 'training'],
    'val'  : ['val', 'valid', 'validation', 'test'],
}

def locate_dataset():
    for root in CANDIDATE_ROOTS:
        for t in SPLIT_NAMES['train']:
            td = root / t
            if td.exists():
                for v in SPLIT_NAMES['val']:
                    vd = root / v
                    if vd.exists():
                        return td.resolve(), vd.resolve(), root.resolve()
    return None, None, None

ORIG_TRAIN_DIR, ORIG_VAL_DIR, DATA_ROOT = locate_dataset()
if ORIG_TRAIN_DIR is None:
    raise FileNotFoundError(
        'Dataset not found. Add kaustubhb999/tomatoleaf via Kaggle Settings → Add data.'
    )

WORKING_DIR      = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
SPLIT_DIR        = WORKING_DIR / 'split_dataset'
MODEL_DIR        = WORKING_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH  = MODEL_DIR / 'tomato_vgg16_best.keras'
FINAL_H5_PATH    = MODEL_DIR / 'tomato_disease_vgg16.h5'
LABELS_PATH      = MODEL_DIR / 'class_labels.json'
LOG_PATH         = MODEL_DIR / 'training_log.csv'

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
EPOCHS_P1    = 15      # Phase 1: head training with frozen VGG16
EPOCHS_P2    = 20      # Phase 2: fine-tune block5 of VGG16
LR_P1        = 1e-4
LR_P2        = 1e-5
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
# ─────────────────────────────────────────────────────────────────────────────

print(f'Data root      : {DATA_ROOT}')
print(f'Original train : {ORIG_TRAIN_DIR}')
print(f'Original val   : {ORIG_VAL_DIR}')
print(f'Split output   : {SPLIT_DIR}')
print(f'Model output   : {MODEL_DIR}')
print(f'Image size     : {IMAGE_SIZE}')
print(f'Batch size     : {BATCH_SIZE}')
print(f'Phase 1 epochs : {EPOCHS_P1}  lr={LR_P1}')
print(f'Phase 2 epochs : {EPOCHS_P2}  lr={LR_P2}')
print(f'Split ratio    : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} (train/val/test)')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
exts = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG')
class_dirs  = sorted([d for d in ORIG_TRAIN_DIR.iterdir() if d.is_dir()])
CLASS_NAMES = [d.name for d in class_dirs]
NUM_CLASSES = len(CLASS_NAMES)

all_images_by_class = {}
for cdir in class_dirs:
    imgs = []
    for ext in exts:
        imgs += list(cdir.glob(ext))
    val_cdir = ORIG_VAL_DIR / cdir.name
    if val_cdir.exists():
        for ext in exts:
            imgs += list(val_cdir.glob(ext))
    all_images_by_class[cdir.name] = [str(p) for p in imgs]

total_images = sum(len(v) for v in all_images_by_class.values())
print(f'Number of classes : {NUM_CLASSES}')
print(f'Total images      : {total_images:,}\n')
print(f'{"Class":<55} {"Images":>8}')
print('-' * 65)
for i, cls in enumerate(CLASS_NAMES):
    n = len(all_images_by_class[cls])
    print(f'  {i:2d}. {cls:<50} {n:>6,}')

# Class distribution bar chart
counts      = [len(all_images_by_class[c]) for c in CLASS_NAMES]
short_names = [c.replace('Tomato___', '').replace('Two-spotted_', '') for c in CLASS_NAMES]

plt.figure(figsize=(14, 5))
bars = plt.bar(short_names, counts, color='#2ecc71', edgecolor='#27ae60', linewidth=0.8)
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             str(cnt), ha='center', va='bottom', fontsize=9)
plt.title('Class Distribution (Full Dataset)', fontsize=13, fontweight='bold')
plt.xlabel('Disease Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'eda_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: eda_distribution.png')

# Sample images grid
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for ax, cls in zip(axes, CLASS_NAMES):
    imgs = all_images_by_class[cls]
    if imgs:
        img = plt.imread(random.choice(imgs))
        ax.imshow(img)
    short = cls.replace('Tomato___', '').replace('_', ' ')
    ax.set_title(short, fontsize=9, fontweight='bold')
    ax.axis('off')
plt.suptitle('Sample Images — All 10 Classes', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'eda_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: eda_samples.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRATIFIED 70/15/15 DATASET SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
# Build split directory structure
for split in ('train', 'val', 'test'):
    for cls in CLASS_NAMES:
        (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

split_counts = {'train': {}, 'val': {}, 'test': {}}

for cls in CLASS_NAMES:
    imgs = all_images_by_class[cls]
    random.Random(SEED).shuffle(imgs)
    train_imgs, temp_imgs = train_test_split(
        imgs, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=SEED
    )
    for split_name, split_imgs in [
        ('train', train_imgs), ('val', val_imgs), ('test', test_imgs)
    ]:
        dest_dir = SPLIT_DIR / split_name / cls
        for src in split_imgs:
            shutil.copy2(src, dest_dir / Path(src).name)
        split_counts[split_name][cls] = len(split_imgs)

df_split = pd.DataFrame(split_counts).T
df_split['TOTAL'] = df_split.sum(axis=1)
print(df_split.to_string())
print(f"\nTrain : {df_split.loc['train','TOTAL']:,}")
print(f"Val   : {df_split.loc['val','TOTAL']:,}")
print(f"Test  : {df_split.loc['test','TOTAL']:,}")
print(f"Total : {df_split['TOTAL'].sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA GENERATORS WITH VGG16 PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
train_gen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range         = 40,
    width_shift_range      = 0.2,
    height_shift_range     = 0.2,
    shear_range            = 0.2,
    zoom_range             = 0.2,
    horizontal_flip        = True,
    vertical_flip          = True,
    brightness_range       = [0.8, 1.2],
    fill_mode              = 'nearest'
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

print(f'Train batches : {len(train_set):,}  ({train_set.n:,} images)')
print(f'Val batches   : {len(val_set):,}  ({val_set.n:,} images)')
print(f'Test batches  : {len(test_set):,}  ({test_set.n:,} images)')
print(f'Classes       : {NUM_CLASSES}')
print(f'Class indices : {train_set.class_indices}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLASS WEIGHTS (imbalance handling)
# ═══════════════════════════════════════════════════════════════════════════════
y_train_labels = train_set.classes
unique_classes = np.unique(y_train_labels)
weights = compute_class_weight(
    class_weight='balanced', classes=unique_classes, y=y_train_labels
)
class_weight_dict = dict(zip(unique_classes, weights))
print('Class weights:')
for idx, w in class_weight_dict.items():
    print(f'  [{idx}] {CLASS_LABELS[idx]:<55}  weight={w:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL ARCHITECTURE (VGG16 + Deep Classification Head)
# ═══════════════════════════════════════════════════════════════════════════════
def build_vgg16_model(num_classes, image_size=(224, 224)):
    """
    VGG16 Transfer Learning with deep classification head.

    Architecture:
        VGG16 (ImageNet) ──► GlobalAveragePooling2D
        ──► Dense(512) ──► BN ──► ReLU ──► Dropout(0.5)
        ──► Dense(256) ──► BN ──► ReLU ──► Dropout(0.3)
        ──► Dense(num_classes, softmax)

    Phase 1: VGG16 fully frozen, train head only
    Phase 2: Unfreeze block5, fine-tune with 10x lower LR
    """
    base_model = VGG16(
        input_shape=image_size + (3,),
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False  # frozen in Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    model   = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

model, base_model = build_vgg16_model(NUM_CLASSES, IMAGE_SIZE)

trainable_p     = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_p = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f'Trainable params     : {trainable_p:,}')
print(f'Non-trainable params : {non_trainable_p:,}')
print(f'Total params         : {trainable_p + non_trainable_p:,}')
model.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PHASE 1: TRAIN CLASSIFICATION HEAD (VGG16 FROZEN)
# ═══════════════════════════════════════════════════════════════════════════════
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_P1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH), monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1
    ),
    CSVLogger(str(LOG_PATH), append=False)
]

print('═'*60)
print(' Phase 1: Training classification head (VGG16 frozen)')
print('═'*60)
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
# SECTION 9 — PHASE 2: FINE-TUNE VGG16 BLOCK5
# ═══════════════════════════════════════════════════════════════════════════════
UNFREEZE_FROM = 'block5_conv1'
unfreeze      = False
for layer in base_model.layers:
    if layer.name == UNFREEZE_FROM:
        unfreeze = True
    layer.trainable = unfreeze

trainable_now = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f'Trainable params after unfreezing block5: {trainable_now:,}')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_P2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH), monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-9, verbose=1
    ),
    CSVLogger(str(LOG_PATH), append=True)
]

print('═'*60)
print(' Phase 2: Fine-tuning VGG16 block5')
print('═'*60)
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
# SECTION 10 — TRAINING HISTORY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

full_history = merge_histories(history_p1, history_p2)
n_p1         = len(history_p1.history['loss'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, metric, title in [
    (axes[0], 'loss',     'Loss'),
    (axes[1], 'accuracy', 'Accuracy'),
]:
    ax.plot(full_history[metric],          label=f'Train {title}', color='#e74c3c', lw=2)
    ax.plot(full_history[f'val_{metric}'], label=f'Val {title}',   color='#2ecc71', lw=2)
    ax.axvline(x=n_p1 - 0.5, color='#3498db', linestyle='--', lw=1.5, label='Phase 1 → 2')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('VGG16 Training History — Phase 1 & Phase 2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: training_history.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EVALUATION ON HELD-OUT TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
best_model = load_model(str(BEST_MODEL_PATH))

val_set.reset()
val_loss, val_acc = best_model.evaluate(val_set, verbose=1)
print(f'\nValidation  — Loss: {val_loss:.4f}  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)')

test_set.reset()
test_loss, test_acc = best_model.evaluate(test_set, verbose=1)
print(f'Test (held-out) — Loss: {test_loss:.4f}  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

# Predictions
test_set.reset()
y_pred_probs = best_model.predict(test_set, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_set.classes
y_true_bin   = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

# Metrics
kappa   = cohen_kappa_score(y_true, y_pred)
mcc     = matthews_corrcoef(y_true, y_pred)
f1_mac  = f1_score(y_true, y_pred, average='macro')
f1_wt   = f1_score(y_true, y_pred, average='weighted')
auc_macro = roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')
top3_correct = sum(
    y_true[i] in np.argsort(y_pred_probs[i])[-3:]
    for i in range(len(y_true))
)
top3_acc = top3_correct / len(y_true)

print('\n' + '='*70)
print('CLASSIFICATION REPORT (Test Set)')
print('='*70)
report = classification_report(y_true, y_pred, target_names=SHORT_LABELS, digits=4)
print(report)

print('='*70)
print('SUMMARY METRICS (Test Set)')
print('='*70)
metrics_summary = {
    'Test Accuracy'        : f'{test_acc*100:.2f}%',
    'Top-3 Accuracy'       : f'{top3_acc*100:.2f}%',
    'Macro F1-Score'       : f'{f1_mac:.4f}',
    'Weighted F1-Score'    : f'{f1_wt:.4f}',
    'AUC-ROC (macro OvR)'  : f'{auc_macro:.4f}',
    "Cohen's Kappa"        : f'{kappa:.4f}',
    'MCC'                  : f'{mcc:.4f}',
    'Test Loss'            : f'{test_loss:.4f}',
}
for k, v in metrics_summary.items():
    print(f'  {k:<30} {v}')

with open(MODEL_DIR / 'test_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print('\nSaved: test_metrics.json')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
cm_mat  = confusion_matrix(y_true, y_pred)
cm_norm = cm_mat.astype('float') / cm_mat.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
sns.heatmap(cm_mat, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
            ax=axes[0], linewidths=0.5)
axes[0].set_title('Confusion Matrix (Counts) — Test Set', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalised) — Test Set', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: confusion_matrix.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — PER-CLASS ACCURACY BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
per_class_acc = cm_mat.diagonal() / cm_mat.sum(axis=1)
colors = ['#2ecc71' if a >= 0.90 else '#f39c12' if a >= 0.80 else '#e74c3c'
          for a in per_class_acc]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(SHORT_LABELS, per_class_acc * 100, color=colors,
              edgecolor='gray', linewidth=0.6)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=90, color='#27ae60', linestyle='--', alpha=0.7, label='90% threshold')
ax.axhline(y=80, color='#e67e22', linestyle='--', alpha=0.7, label='80% threshold')
ax.set_ylim(0, 110)
ax.set_title('Per-Class Accuracy — Test Set', fontsize=13, fontweight='bold')
ax.set_xlabel('Disease Class')
ax.set_ylabel('Accuracy (%)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: per_class_accuracy.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ROC CURVES (ONE-VS-REST)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
palette = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

for i, (cls_label, color) in enumerate(zip(SHORT_LABELS, palette)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{cls_label} (AUC={roc_auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — One-vs-Rest (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: roc_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — GRAD-CAM VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    """Compute Grad-CAM heatmap for the top predicted class."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index    = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads   = tape.gradient(class_channel, conv_outputs)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap, int(pred_index), float(tf.reduce_max(predictions).numpy())

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap_resized = np.uint8(255 * heatmap)
    jet        = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    colored    = jet_colors[heatmap_resized]
    colored    = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).resize(
        (original_img.shape[1], original_img.shape[0]), Image.LANCZOS
    )
    colored_arr  = np.array(colored_img)
    superimposed = (colored_arr * alpha + original_img * (1 - alpha)).astype(np.uint8)
    return superimposed

# One Grad-CAM per class
test_files = []
for cls in CLASS_LABELS:
    cls_path = SPLIT_DIR / 'test' / cls
    cls_imgs = sorted(cls_path.glob('*.jpg')) + sorted(cls_path.glob('*.JPG')) + \
               sorted(cls_path.glob('*.png'))
    if cls_imgs:
        test_files.append((cls, cls_imgs[0]))

n_show = min(NUM_CLASSES, 10)
fig, axes = plt.subplots(n_show, 3, figsize=(12, n_show * 3.5))

for row_idx, (cls_name, img_path) in enumerate(test_files[:n_show]):
    ax_orig, ax_hm, ax_cam = axes[row_idx]
    orig    = np.array(Image.open(img_path).convert('RGB').resize(IMAGE_SIZE))
    img_arr = preprocess_input(np.expand_dims(orig.astype('float32'), 0))
    heatmap, pred_idx, conf = make_gradcam_heatmap(img_arr, best_model)
    pred_cls   = CLASS_LABELS[pred_idx].replace('Tomato___', '').replace('_', ' ')
    short      = cls_name.replace('Tomato___', '').replace('_', ' ')
    correct    = CLASS_LABELS[pred_idx] == cls_name
    tick_color = 'green' if correct else 'red'

    ax_orig.imshow(orig)
    ax_orig.set_title(f'True: {short}', fontsize=8, fontweight='bold')
    ax_orig.axis('off')

    ax_hm.imshow(heatmap, cmap='jet')
    ax_hm.set_title('Grad-CAM Heatmap', fontsize=8)
    ax_hm.axis('off')

    superimposed = overlay_gradcam(orig, heatmap)
    ax_cam.imshow(superimposed)
    ax_cam.set_title(f'Pred: {pred_cls}\n({conf*100:.1f}%)',
                     fontsize=8, color=tick_color)
    ax_cam.axis('off')

plt.suptitle('Grad-CAM Visualizations — One Sample per Class',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'gradcam_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: gradcam_visualization.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — INFERENCE FUNCTION (production-ready)
# ═══════════════════════════════════════════════════════════════════════════════
def predict_disease(image_path_or_pil, model, class_indices, image_size=(224, 224), top_k=3):
    """
    Predict tomato leaf disease from an image.

    Parameters
    ----------
    image_path_or_pil : str | Path | PIL.Image
    model             : loaded Keras model
    class_indices     : dict mapping class_name -> index
    image_size        : (H, W), must match training size (224, 224)
    top_k             : number of top predictions to return

    Returns
    -------
    list of (class_name, confidence_pct) sorted by confidence desc
    """
    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(image_path_or_pil).convert('RGB')
    else:
        img = image_path_or_pil.convert('RGB')

    img       = img.resize(image_size)
    img_array = preprocess_input(np.array(img, dtype='float32'))   # CRITICAL: VGG16 norm
    img_array = np.expand_dims(img_array, axis=0)

    probs       = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    idx_to_class = {v: k for k, v in class_indices.items()}
    return [(idx_to_class[i], float(probs[i]) * 100) for i in top_indices]

# Quick test
test_imgs_all = list((SPLIT_DIR / 'test').glob('**/*.jpg'))
if test_imgs_all:
    sample_img = random.choice(test_imgs_all)
    true_label = sample_img.parent.name
    results = predict_disease(sample_img, best_model, train_set.class_indices)
    print(f'\nTrue label : {true_label}')
    print('Top-3 predictions:')
    for rank, (cls, conf) in enumerate(results, 1):
        marker = ' ✓' if cls == true_label else ''
        print(f'  {rank}. {cls:<55} {conf:.2f}%{marker}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — SAVE MODELS & LABEL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
best_model.save(str(FINAL_H5_PATH))
print(f'H5 model saved       : {FINAL_H5_PATH}')

best_model.save(str(MODEL_DIR / 'tomato_disease_vgg16.keras'))
print(f'Keras model saved    : {MODEL_DIR / "tomato_disease_vgg16.keras"}')

labels_payload = {
    'class_indices' : train_set.class_indices,
    'idx_to_class'  : {str(v): k for k, v in train_set.class_indices.items()},
    'num_classes'   : NUM_CLASSES,
    'image_size'    : list(IMAGE_SIZE),
    'preprocessing' : 'vgg16_preprocess_input',
    'split_ratio'   : f'{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}',
    'test_metrics'  : metrics_summary,
}
with open(str(LABELS_PATH), 'w') as f:
    json.dump(labels_payload, f, indent=2)
print(f'Class labels saved   : {LABELS_PATH}')

h5_mb = FINAL_H5_PATH.stat().st_size / (1024**2)
print(f'H5 size              : {h5_mb:.1f} MB')
print('\n✓ All artifacts saved. Ready for deployment.')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — SAMPLE PREDICTION GRID (20 random test images)
# ═══════════════════════════════════════════════════════════════════════════════
n_show = 20
fig, axes = plt.subplots(4, 5, figsize=(18, 15))
axes = axes.flatten()
idx_to_class  = {v: k for k, v in train_set.class_indices.items()}
all_test_files = (sorted((SPLIT_DIR / 'test').glob('**/*.jpg')) +
                  sorted((SPLIT_DIR / 'test').glob('**/*.JPG')) +
                  sorted((SPLIT_DIR / 'test').glob('**/*.png')))
sample_files = random.sample(all_test_files, min(n_show, len(all_test_files)))

for ax, img_path in zip(axes, sample_files):
    true_cls = img_path.parent.name
    orig     = np.array(Image.open(img_path).convert('RGB').resize(IMAGE_SIZE))
    img_arr  = preprocess_input(np.expand_dims(orig.astype('float32'), 0))
    probs    = best_model.predict(img_arr, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_cls = idx_to_class[pred_idx]
    conf     = probs[pred_idx] * 100
    correct  = pred_cls == true_cls
    ax.imshow(orig)
    true_short = true_cls.replace('Tomato___', '')[:18]
    pred_short = pred_cls.replace('Tomato___', '')[:18]
    color = 'green' if correct else 'red'
    ax.set_title(f'True: {true_short}\nPred: {pred_short} ({conf:.0f}%)',
                 fontsize=7, color=color)
    ax.axis('off')

plt.suptitle('Sample Predictions — Green=Correct, Red=Wrong (Test Set)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: sample_predictions.png')
print('\n' + '='*70)
print('ALL DONE — Models, metrics, and visualizations saved to:', MODEL_DIR)
print('='*70)
