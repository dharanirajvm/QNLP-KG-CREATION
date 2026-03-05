import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psutil
import gc
import time
from sklearn.model_selection import train_test_split
from lambeq import IQPAnsatz, PennyLaneModel, AtomicType
from lambeq.backend import Ty


# ======================================================
# RAM MONITOR UTILITIES
# ======================================================

def log_ram(stage=""):
    mem = psutil.virtual_memory()
    print(f"\n[RAM LOG] {stage}")
    print(f"Used      : {mem.used/1e9:.2f} GB")
    print(f"Available : {mem.available/1e9:.2f} GB")
    print(f"Percent   : {mem.percent}%")
    print("-"*50)


def prevent_ram_crash(threshold=0.85):
    """
    Prevent OS-level crash by stopping early.
    """
    mem = psutil.virtual_memory()
    if mem.percent/100 > threshold:
        print("\n⚠ RAM CRITICAL — stopping safely")
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError("RAM safety stop triggered")


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ======================================================
# LOAD DATASET
# ======================================================

print("="*60)
print("[LOG] Loading dataset")
print("="*60)

with open("simplified_bobcat_diagrams2.pkl", "rb") as f:
    data = pickle.load(f)

print("[LOG] Total samples:", len(data))
log_ram("After dataset load")


# ======================================================
# SELECT TOP 5 CLASSES
# ======================================================

print("\n[LOG] Selecting top 5 classes by frequency")

# Count samples per relation
from collections import Counter
relation_counts = Counter([x['relation'] for x in data])

# Get top 5 most frequent relations
top_5_relations = [rel for rel, count in relation_counts.most_common(5)]

print(f"[LOG] Top 5 relations: {top_5_relations}")
for rel in top_5_relations:
    print(f"  - {rel}: {relation_counts[rel]} samples")

# Filter dataset to only include top 5 classes
data_filtered = [x for x in data if x['relation'] in top_5_relations]

print(f"\n[LOG] Original samples: {len(data)}")
print(f"[LOG] Filtered samples (5 classes): {len(data_filtered)}")

# Use filtered data from here on
data = data_filtered

log_ram("After filtering to 5 classes")


# ======================================================
# LABEL ENCODING (5 CLASSES ONLY)
# ======================================================

relations = sorted(top_5_relations)  # Use only the 5 selected classes
rel2idx = {r:i for i,r in enumerate(relations)}
idx2rel = {i:r for r,i in rel2idx.items()}
num_classes = len(relations)  # Should be 5

for d in data:
    d['label'] = rel2idx[d['relation']]

print(f"\n[LOG] Number of classes: {num_classes}")
print("[LOG] Label mapping:")
for rel, idx in rel2idx.items():
    print(f"  {idx}: {rel}")


# ======================================================
# DATA SPLIT
# ======================================================

print("\n[LOG] Splitting data")

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=[d['label'] for d in data],
    random_state=42
)

train_data, dev_data = train_test_split(
    train_data,
    test_size=0.2,
    stratify=[d['label'] for d in train_data],
    random_state=42
)

print(f"[LOG] Train: {len(train_data)} samples")
print(f"[LOG] Dev: {len(dev_data)} samples")
print(f"[LOG] Test: {len(test_data)} samples")

# Print class distribution
print("\n[LOG] Class distribution in train set:")
train_label_counts = Counter([d['label'] for d in train_data])
for label in sorted(train_label_counts.keys()):
    print(f"  Class {label} ({idx2rel[label]}): {train_label_counts[label]} samples")


# ======================================================
# QUANTUM ANSATZ
# ======================================================

print("\n[LOG] Building ansatz")

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = Ty('p')

ansatz = IQPAnsatz(
    ob_map={
        N: 1,
        S: 1,      # keep minimal
        P: 0
    },
    n_layers=1
)


# ======================================================
# CIRCUIT GENERATION
# ======================================================

def to_circuits(dataset, name):



    circuits = []
    labels = []

    for i, x in enumerate(dataset):

        #prevent_ram_crash()

        circuits.append(ansatz(x['diagram']))
        labels.append(x['label'])

        if i % 50 == 0:
            print(f"[LOG] {name}: {i}/{len(dataset)}")
            log_ram(f"{name} circuit build")
            #draw the circuit for the first sample
            if i == 0 or i == 50  :
                try:
                    circuits[0].draw(path=f"{name}_{i}_sample_circuit.png")
                    print(f"[LOG] Sample circuit drawn → {name}_{i}_sample_circuit.png")
                except Exception as e:
                    print("[WARN] Could not draw sample circuit:", e)

    cleanup_memory()

    return circuits, torch.tensor(labels)


print("\n[LOG] Generating circuits for 5 classes only")

train_circuits, train_labels = to_circuits(train_data, "TRAIN")
dev_circuits, dev_labels = to_circuits(dev_data, "DEV")
test_circuits, test_labels = to_circuits(test_data, "TEST")

print("[LOG] Circuit generation complete")
log_ram("After circuit generation")


# ======================================================
# MODEL INITIALIZATION
# ======================================================

print("\n[LOG] Initializing PennyLane model")

all_circuits = train_circuits + dev_circuits + test_circuits

model = PennyLaneModel.from_diagrams(
    all_circuits,
    probabilities=True,
    normalize=True
)

model.initialise_weights()
# model = model.double()

log_ram("After quantum model init")


# ======================================================
# OUTPUT DIMENSION
# ======================================================

with torch.no_grad():
    sample_out = model([train_circuits[0]])

output_dim = sample_out.shape[-1]
print(f"[LOG] Quantum output dim: {output_dim}")
print(f"[LOG] Number of output classes: {num_classes}")


# ======================================================
# CLASSIFIER (5 CLASSES)
# ======================================================

class ClassifierHead(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)  # Output: 5 classes
        )

    def forward(self, x):

        if torch.is_complex(x):
            x = x.real

        while x.dim() > 2:
            x = x.squeeze(1)

        return self.fc(x)


classifier = ClassifierHead(output_dim, num_classes)

print(f"[LOG] Classifier initialized for {num_classes} classes")


# ======================================================
# DEVICE
# ======================================================

device = torch.device("cpu")  # safer for PennyLane
model.to(device)
classifier.to(device)

train_labels = train_labels.long()
dev_labels = dev_labels.long()
test_labels = test_labels.long()


# ======================================================
# OPTIMIZER
# ======================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.01
)


# ======================================================
# TRAINING FUNCTIONS
# ======================================================

def train_epoch(epoch):

    model.train()
    classifier.train()

    total_loss = 0

    log_ram(f"Epoch {epoch} start")

    for i, (circ, lbl) in enumerate(zip(train_circuits, train_labels)):

        # prevent_ram_crash()

        optimizer.zero_grad()

        try:
            quantum_out = model([circ])
        except Exception as e:
            print("Quantum error:", e)
            continue

        logits = classifier(quantum_out)

        target = lbl.view(1)

        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            log_ram(f"Epoch {epoch} step {i}")

        cleanup_memory()

    log_ram(f"Epoch {epoch} end")

    return total_loss / len(train_circuits)


def eval_acc(circuits, labels):

    model.eval()
    classifier.eval()

    correct = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for circ, lbl in zip(circuits, labels):

            quantum_out = model([circ])
            logits = classifier(quantum_out)

            pred = logits.argmax(-1).item()
            predictions.append(pred)
            true_labels.append(lbl.item())

            if pred == lbl.item():
                correct += 1

    return correct / len(circuits), predictions, true_labels


def print_confusion_matrix(predictions, true_labels, dataset_name):
    """Print confusion matrix for 5 classes"""
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"\n[{dataset_name}] Confusion Matrix:")
    print("Pred ->", end=" ")
    for i in range(num_classes):
        print(f"{i:4d}", end=" ")
    print()
    
    for i in range(num_classes):
        print(f"True {i} |", end=" ")
        for j in range(num_classes):
            print(f"{cm[i][j]:4d}", end=" ")
        print()
    
    print(f"\n[{dataset_name}] Classification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=[idx2rel[i] for i in range(num_classes)]))


# ======================================================
# TRAIN LOOP
# ======================================================

print("\n" + "="*60)
print("[TRAINING START - 5 CLASS CLASSIFICATION]")
print("="*60)

best_dev_acc = 0
best_epoch = 0

for epoch in range(1, 11):

    try:
        loss = train_epoch(epoch)
    except RuntimeError as e:
        print("Training stopped safely:", e)
        break

    train_acc, train_preds, train_true = eval_acc(train_circuits, train_labels)
    dev_acc, dev_preds, dev_true = eval_acc(dev_circuits, dev_labels)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_epoch = epoch

    print(
        f"[EPOCH {epoch}] "
        f"Loss={loss:.4f} | "
        f"TrainAcc={train_acc:.4f} | "
        f"DevAcc={dev_acc:.4f}"
    )

    # Print confusion matrix every 5 epochs
    if epoch % 5 == 0:
        print_confusion_matrix(dev_preds, dev_true, "DEV SET")

    # checkpoint save
    if epoch % 2 == 0:
        torch.save({
            'quantum_model_state': model.state_dict(),
            'classifier_state': classifier.state_dict(),
            'epoch': epoch,
            'num_classes': num_classes,
            'rel2idx': rel2idx,
            'idx2rel': idx2rel
        }, "checkpoint_qnlp_5class.pt")

        print("Checkpoint saved")


# ======================================================
# FINAL RESULTS
# ======================================================

print("\n" + "="*60)
print("[FINAL EVALUATION - 5 CLASS CLASSIFICATION]")
print("="*60)

test_acc, test_preds, test_true = eval_acc(test_circuits, test_labels)

print(f"\nBest Dev Accuracy: {best_dev_acc:.4f} (Epoch {best_epoch})")
print(f"Test Accuracy: {test_acc:.4f}")

print_confusion_matrix(test_preds, test_true, "TEST SET")


# ======================================================
# SAVE FINAL MODEL
# ======================================================

torch.save({
    'quantum_model_state': model.state_dict(),
    'classifier_state': classifier.state_dict(),
    'rel2idx': rel2idx,
    'idx2rel': idx2rel,
    'output_dim': output_dim,
    'num_classes': num_classes,
    'test_acc': test_acc,
    'best_dev_acc': best_dev_acc,
    'top_5_relations': top_5_relations
}, "final_qnlp_model_5class.pt")

print("\n[LOG] Model saved successfully")
print(f"[LOG] Classes used: {top_5_relations}")
print(f"[LOG] Final test accuracy: {test_acc:.4f}")