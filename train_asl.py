import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


# -----------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------
df = pd.read_csv("asl_data.csv", header=None)

# Convert numeric columns (0–62) to floats
for col in range(63):
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert label column (63) to string
df[63] = df[63].astype(str)

# Drop rows where any numeric column failed conversion
df = df.dropna(subset=list(range(63)))

# Extract features and labels
X = df.iloc[:, :-1].values   # 63 numeric values
y = df.iloc[:, -1].values    # label (A, B, C, ...)

# -----------------------------
# 2. NORMALIZATION FUNCTION
# -----------------------------
def normalize_landmarks(sample):
    sample = sample.astype(float)          # ensure numeric
    sample = sample.reshape(21, 3)         # 21 landmarks × 3 coords

    wrist = sample[0]                      # landmark 0 = wrist
    sample = sample - wrist                # translate to origin

    max_dist = np.max(np.linalg.norm(sample, axis=1))
    if max_dist == 0:
        max_dist = 1                       # avoid divide-by-zero

    sample = sample / max_dist             # scale to unit size

    return sample.flatten()

# Apply normalization to all samples
X_norm = np.array([normalize_landmarks(row) for row in X])

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

clf.fit(X_norm, y)

print("Training complete!")
print("Classes learned:", sorted(set(y)))

# -----------------------------
# 4. SAVE MODEL
# -----------------------------
with open("asl_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved as asl_model.pkl")