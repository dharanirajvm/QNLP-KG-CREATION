import os
import time
import pickle
import pandas as pd

from lambeq import BobcatParser
from lambeq.backend import Diagram
from lambeq.text2diagram.model_based_reader.bobcat_parser import BobcatParseError


# ===============================
# CONFIG
# ===============================
CSV_PATH = r"C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\relation_extraction_discocat_v2.csv"
OUT_PATH = "simplified_bobcat_diagrams_toy_v2.pkl"
PATTERN_DIR = "patterns"

LOG_EVERY = 50

os.makedirs(PATTERN_DIR, exist_ok=True)


# ===============================
# LOAD DATASET FROM CSV
# ===============================
print("[LOAD] Loading simplified dataset...")

df = pd.read_csv(CSV_PATH)

dataset = df.to_dict("records")
total = len(dataset)

print("Total samples:", total)


# ===============================
# INIT PARSER
# ===============================
print("[INIT] Loading Bobcat parser...")
parser = BobcatParser()


# ===============================
# TRACK GRAMMAR PATTERNS
# ===============================
seen_patterns = set()


# ===============================
# PARSE LOOP
# ===============================
parsed = []
failed_parse = 0
failed_compose = 0

start_time = time.time()

print("\n[START] Parsing simplified sentences")
print("="*70)

for i, d in enumerate(dataset):

    sent = d["simplified_sentence"]

    try:
        diag = parser.sentence2diagram(sent)

        if not isinstance(diag, Diagram):
            raise ValueError("Invalid diagram")

        # -------- PATTERN DETECTION --------
        pattern_key = str(diag.cod)  # grammar output type

        if pattern_key not in seen_patterns:
            seen_patterns.add(pattern_key)

            # Save diagram visualization
            try:
                img_path = os.path.join(
                    PATTERN_DIR,
                    f"pattern_{len(seen_patterns)}.png"
                )

                diag.draw(path=img_path)

                print(f"[NEW PATTERN] Saved → {img_path}")
                print("Pattern type:", pattern_key)

            except Exception as e:
                print("[WARN] Could not draw pattern:", e)

        # -------- STORE RESULT --------
        parsed.append({
            "diagram": diag,
            "relation": d["relation"],
            "sentence": sent,
            "head": d["head_entity"],
            "tail": d["tail_entity"]
        })

    except BobcatParseError:
        failed_parse += 1

    except ValueError:
        failed_compose += 1

    # -------- LOGGING --------
    if (i + 1) % LOG_EVERY == 0:

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed else 0

        print(
            f"[PROGRESS] {i+1}/{total} | "
            f"Parsed={len(parsed)} | "
            f"ParseFail={failed_parse} | "
            f"ComposeFail={failed_compose} | "
            f"Patterns={len(seen_patterns)} | "
            f"{rate:.2f} sent/sec"
        )

        # incremental save
        with open(OUT_PATH, "wb") as f:
            pickle.dump(parsed, f)


# ===============================
# FINAL SUMMARY
# ===============================
elapsed = time.time() - start_time

print("\n" + "="*70)
print("[DONE] Parsing finished")
print("="*70)

print("✔ Parsed successfully :", len(parsed))
print("✖ Parse failures      :", failed_parse)
print("✖ Composition failures:", failed_compose)
print("✔ Patterns discovered :", len(seen_patterns))
print("⏱ Total time (sec)    :", round(elapsed, 2))
print("⚡ Avg speed           :", round(total/elapsed, 2))
print("="*70)


# ===============================
# SAVE FINAL DATA
# ===============================
with open(OUT_PATH, "wb") as f:
    pickle.dump(parsed, f)

print(f"[SAVED] Diagrams dataset → {OUT_PATH}")
