import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lambeq import IQPAnsatz, PennyLaneModel, AtomicType
from lambeq.backend import Ty


print("="*60)
print("[LOG] Loading diagram dataset")
print("="*60)

with open("llm_simplified_bobcat_diagrams.pkl","rb") as f:
    data = pickle.load(f)

print("[LOG] Total samples:", len(data))


# ======================================================
# LABEL ENCODING
# ======================================================
relations = sorted({x['relation'] for x in data})
rel2idx = {r:i for i,r in enumerate(relations)}
idx2rel = {i:r for r,i in rel2idx.items()}

for d in data:
    d['label'] = rel2idx[d['relation']]

num_classes = len(rel2idx)

print("[LOG] Relation map:", rel2idx)


# ======================================================
# TRAIN/DEV/TEST SPLIT
# ======================================================
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

print("[LOG] Train:", len(train_data))
print("[LOG] Dev:", len(dev_data))
print("[LOG] Test:", len(test_data))


# ======================================================
# ANSATZ CONFIGURATION (IMPORTANT)
# ======================================================
print("\n[LOG] Configuring quantum ansatz")

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = Ty('p')

ansatz = IQPAnsatz(
    ob_map={
        N: 1,
        S: 1,  # critical: avoid multi-qubit explosion
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

        circuits.append(ansatz(x['diagram']))
        labels.append(x['label'])

        if i % 50 == 0:
            print(f"[LOG] {name}: {i}/{len(dataset)} circuits built")

    return circuits, torch.tensor(labels)


train_circuits, train_labels = to_circuits(train_data, "TRAIN")
dev_circuits, dev_labels = to_circuits(dev_data, "DEV")
test_circuits, test_labels = to_circuits(test_data, "TEST")

print("[LOG] Circuit generation complete")


# ======================================================
# QUANTUM MODEL INITIALIZATION
# ======================================================
print("\n[LOG] Building PennyLane model")

# CRITICAL: Use ALL circuits for from_diagrams
all_circuits = train_circuits + dev_circuits + test_circuits

model = PennyLaneModel.from_diagrams(
    all_circuits,
    probabilities=False,
    normalize=False
)

print("[LOG] Initializing quantum weights")
model.initialise_weights()

# Ensure double precision
model = model.double()


# ======================================================
# GET OUTPUT DIMENSION
# ======================================================
with torch.no_grad():
    sample_out = model([train_circuits[0]])

output_dim = sample_out.shape[-1]

print("[LOG] Quantum output dimension:", output_dim)


# ======================================================
# CLASSIFIER HEAD
# ======================================================
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # Handle complex numbers
        if torch.is_complex(x):
            x = x.real
        
        # Flatten to ensure we have [batch_size, features]
        # x might be [1, 1, 2] or [1, 2], we want [1, 2]
        while x.dim() > 2:
            x = x.squeeze(1)
        
        # Pass through classifier
        out = self.fc(x)
        
        # Ensure output is [batch_size, num_classes]
        # Should be [1, 9] for single sample
        return out

classifier = ClassifierHead(output_dim, num_classes).double()

print("[LOG] Classifier created")


# ======================================================
# DEVICE SETUP
# ======================================================
# CRITICAL FIX: PennyLane quantum circuits run on CPU by default
# Keep everything on CPU to avoid device mismatch
device = torch.device("cpu")
print("[LOG] Using device:", device)
print("[LOG] Note: Using CPU because PennyLane quantum circuits run on CPU")

model.to(device)
classifier.to(device)

train_labels = train_labels.to(device).long()
dev_labels = dev_labels.to(device).long()
test_labels = test_labels.to(device).long()


# ======================================================
# OPTIMIZATION SETUP
# ======================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.01
)


# ======================================================
# TRAINING FUNCTIONS
# ======================================================
def train_epoch():

    model.train()
    classifier.train()

    total_loss = 0

    for circ, lbl in zip(train_circuits, train_labels):

        optimizer.zero_grad()

        quantum_out = model([circ])
        
        logits = classifier(quantum_out)
        
        # CrossEntropyLoss expects:
        # - input: (N, C) = (batch_size, num_classes)
        # - target: (N,) = (batch_size,) containing class indices
        
        # logits should be [1, 9]
        # lbl is a scalar, so we make it [1]
        
        # Ensure logits is exactly [1, num_classes]
        if logits.shape[0] != 1 or logits.shape[1] != num_classes:
            print(f"WARNING: Unexpected logits shape: {logits.shape}")
            logits = logits.view(1, -1)
        
        # Ensure label is [1]
        target = lbl.view(1)
        
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_circuits)


def eval_acc(circuits, labels):

    model.eval()
    classifier.eval()

    correct = 0

    with torch.no_grad():
        for circ, lbl in zip(circuits, labels):

            quantum_out = model([circ])
            logits = classifier(quantum_out)

            pred = logits.argmax(-1).item()

            if pred == lbl.item():
                correct += 1

    return correct / len(circuits)


# ======================================================
# TRAIN LOOP
# ======================================================
print("\n" + "="*60)
print("[TRAINING START]")
print("="*60)

best_dev_acc = 0
best_epoch = 0

for epoch in range(1, 11):

    loss = train_epoch()

    train_acc = eval_acc(train_circuits, train_labels)
    dev_acc = eval_acc(dev_circuits, dev_labels)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_epoch = epoch

    print(
        f"[EPOCH {epoch:02d}] "
        f"Loss={loss:.4f} | "
        f"TrainAcc={train_acc:.4f} | "
        f"DevAcc={dev_acc:.4f}"
    )


test_acc = eval_acc(test_circuits, test_labels)

print("\n" + "="*60)
print("[FINAL RESULTS]")
print("="*60)

print("Best Dev Accuracy:", best_dev_acc, "Epoch:", best_epoch)
print("Test Accuracy:", test_acc)


# ======================================================
# SAVE MODEL
# ======================================================
print("\n[LOG] Saving trained model")

torch.save({
    'quantum_model_state': model.state_dict(),
    'classifier_state': classifier.state_dict(),
    'rel2idx': rel2idx,
    'idx2rel': idx2rel,
    'output_dim': output_dim,
    'num_classes': num_classes,
    'test_acc': test_acc,
    'best_dev_acc': best_dev_acc
}, "/content/drive/MyDrive/archive/final_qnlp_model.pt")

print("[LOG] Model saved successfully")
print("="*60)