# import pickle, torch, torch.nn as nn, torch.optim as optim
# from sklearn.model_selection import train_test_split
# from lambeq import IQPAnsatz, PennyLaneModel, AtomicType
# from lambeq.backend import Ty

# print("[LOG] Loading dataset...")

# with open("/content/drive/MyDrive/archive/llm_simplified_bobcat_diagrams.pkl","rb") as f:
#     data = pickle.load(f)

# print("[LOG] Total samples loaded:", len(data))

# # ========== LABEL ENCODING ==========
# relations = sorted({x['relation'] for x in data})
# rel2idx = {r:i for i,r in enumerate(relations)}
# idx2rel = {i:r for r,i in rel2idx.items()}
# print("[LOG] Relation map:", rel2idx)

# for d in data:
#     d['label'] = rel2idx[d['relation']]

# # ========== SPLITTING (TRAIN/DEV/TEST) ==========
# train_data, test_data = train_test_split(
#     data, test_size=0.2, stratify=[d['relation'] for d in data], random_state=42
# )

# train_data, dev_data = train_test_split(
#     train_data, test_size=0.2, stratify=[d['relation'] for d in train_data], random_state=42
# )

# print("[LOG] Train size:", len(train_data))
# print("[LOG] Dev size:", len(dev_data))
# print("[LOG] Test size:", len(test_data))

# # ========== CIRCUIT GENERATION ==========
# N = AtomicType.NOUN
# S = AtomicType.SENTENCE
# P = Ty('p')  # ← IMPORTANT
# CONJ  = Ty('conj')
# DET   = Ty('det')
# REL   = Ty('rel')

# s_qubits = int(np.ceil(np.log2(num_classes)))
# print("[INFO] Sentence qubits:", s_qubits)
# ansatz = IQPAnsatz(
#     ob_map={
#         N: 1,
#         S: s_qubits,
#         P: 0,
#         CONJ:  0,
#         DET:   0,
#         REL:   0,
#     },
#     n_layers=1
# )

# def to_circuits(ds, label):
#     circs = []
#     labels = []
#     for i, x in enumerate(ds):
#         circs.append(ansatz(x['diagram']))
#         labels.append(x['label'])
#         if i % 20 == 0:
#             print(f"[LOG] {label} circuit gen {i}/{len(ds)}")
#     return circs, torch.tensor(labels)

# train_circuits, train_labels = to_circuits(train_data, "TRAIN")
# dev_circuits, dev_labels     = to_circuits(dev_data, "DEV")
# test_circuits, test_labels   = to_circuits(test_data, "TEST")

# print("[LOG] Circuit generation complete!")

# # ========== MODEL INITIALIZATION ==========
# num_classes = len(rel2idx)

# # CRITICAL: Use ALL circuits for from_diagrams
# # This ensures the model can execute ANY circuit built from the vocabulary
# all_circuits = train_circuits + dev_circuits + test_circuits

# print(f"[LOG] Building PennyLane model with {len(all_circuits)} circuits...")

# model = PennyLaneModel.from_diagrams(
#     all_circuits,
#     probabilities=True,
#     normalize=True
# )

# print("[LOG] PennyLane model created!")

# # CRITICAL: Initialize weights BEFORE training
# print("[LOG] Initializing quantum weights...")
# model.initialise_weights()
# print("[LOG] Quantum weights initialized!")

# # Get output dimension
# with torch.no_grad():
#     test_output = model([train_circuits[0]])
#     output_dim = test_output.shape[-1]
# print(f"[LOG] Quantum output dimension: {output_dim}")

# # Create classifier
# class ClassifierHead(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, num_classes)
#         )

#     def forward(self, x):
#         if torch.is_complex(x):
#             x = x.real
#         return self.fc(x)

# classifier = ClassifierHead(output_dim, num_classes)
# print(f"[LOG] Classifier: {output_dim} -> 16 -> {num_classes}")

# # ========== DEVICE SETUP ==========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[LOG] Using device: {device}")

# model.to(device)
# classifier.to(device)

# train_labels = train_labels.to(device).long()
# dev_labels = dev_labels.to(device).long()
# test_labels = test_labels.to(device).long()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     list(model.parameters()) + list(classifier.parameters()),
#     lr=0.01
# )

# # ========== TRAINING ==========
# def train_epoch():
#     model.train()
#     classifier.train()
#     total_loss = 0

#     for circ, lbl in zip(train_circuits, train_labels):
#         optimizer.zero_grad()
#         quantum_out = model([circ])
#         logits = classifier(quantum_out)
#         loss = criterion(logits, lbl.unsqueeze(0))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(train_circuits)

# def eval_acc(circuits, labels):
#     model.eval()
#     classifier.eval()
#     correct = 0

#     with torch.no_grad():
#         for circ, lbl in zip(circuits, labels):
#             quantum_out = model([circ])
#             logits = classifier(quantum_out)
#             pred = logits.argmax(1).item()
#             if pred == lbl.item():
#                 correct += 1

#     return correct / len(circuits)

# print("[LOG] Starting training...")
# print("="*60)

# best_dev_acc = 0.0

# for epoch in range(1, 11):
#     loss = train_epoch()
#     train_acc = eval_acc(train_circuits, train_labels)
#     dev_acc   = eval_acc(dev_circuits, dev_labels)

#     if dev_acc > best_dev_acc:
#         best_dev_acc = dev_acc
#         best_epoch = epoch

#     print(f"[EPOCH {epoch:02d}] loss={loss:.4f} | train_acc={train_acc:.4f} | dev_acc={dev_acc:.4f} {'🌟' if dev_acc == best_dev_acc else ''}")

# print("="*60)
# test_acc = eval_acc(test_circuits, test_labels)
# print(f"[LOG] Final Test Accuracy = {test_acc:.4f}")
# print(f"[LOG] Best Dev Accuracy = {best_dev_acc:.4f} (epoch {best_epoch})")

# # ========== SAVE MODEL ==========
# print("[LOG] Saving model...")

# # CRITICAL: Save all the metadata needed for reconstruction
# torch.save({
#     'quantum_model_state': model.state_dict(),
#     'classifier_state': classifier.state_dict(),
#     'rel2idx': rel2idx,
#     'idx2rel': idx2rel,
#     'output_dim': output_dim,
#     'num_classes': num_classes,
#     'test_acc': test_acc,
#     'best_dev_acc': best_dev_acc,
#     # Save the data split info so we can recreate exact same circuits
#     'random_state': 42
# }, "/content/drive/MyDrive/archive/approach2_trained_model.pt")

# print("[LOG] Model saved!")
# print("[LOG] Training complete!")
import numpy as np
import torch
print(torch.__version__)
