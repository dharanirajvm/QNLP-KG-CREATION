import re
import pandas as pd

SEMEVAL_PATH = r"C:\Users\DHARANIRAJ VM\Downloads\archive\TRAIN_FILE.TXT"

BASE_REL = {
    "Cause-Effect":"cause_effect",
    "Component-Whole":"part_of",
    "Content-Container":"in",
    "Entity-Destination":"moves_to",
    "Entity-Origin":"originates_from",
    "Instrument-Agency":"used_by",
    "Member-Collection":"member_of",
    "Message-Topic":"talks_about",
    "Product-Producer":"produced_by"
}


def load_semeval_entities(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0
    while i < len(lines):

        if not lines[i] or not lines[i][0].isdigit():
            i += 1
            continue

        sentence_line = lines[i]
        relation_line = lines[i+1]
        i += 3

        if relation_line == "Other":
            continue

        raw = sentence_line.split("\t",1)[1].strip().strip('"')

        e1 = re.search(r'<e1>(.*?)</e1>', raw)
        e2 = re.search(r'<e2>(.*?)</e2>', raw)

        if not e1 or not e2:
            continue

        clean_sent = re.sub(r'</?e[12]>', '', raw).lower()

        m = re.match(r"(.+)\((e\d),(e\d)\)", relation_line)
        if not m:
            continue

        base = m.group(1)
        if base not in BASE_REL:
            continue

        samples.append({
            "sentence": clean_sent,
            "head": e1.group(1).lower(),
            "tail": e2.group(1).lower(),
            "relation": BASE_REL[base]
        })

    return pd.DataFrame(samples)


df = load_semeval_entities(SEMEVAL_PATH)

# Balanced ~110 per class
balanced_df = (
    df.groupby("relation", group_keys=False)
    .apply(lambda x: x.sample(min(len(x),110), random_state=42))
    .reset_index(drop=True)
)

print("Balanced dataset:", len(balanced_df))
print(balanced_df["relation"].value_counts())

balanced_df.to_csv("balanced_input_for_llm.csv", index=False)
