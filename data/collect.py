import pandas as pd

def load_file(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    return df

def load_combined():
    df1 = load_file("data/train_FD001.txt")
    df2 = load_file("data/train_FD002.txt")

    # Add source label (IMPORTANT)
    df1["source"] = "FD001"
    df2["source"] = "FD002"

    # Combine
    df = pd.concat([df1, df2], ignore_index=True)

    # Rename columns
    cols = ["unit", "cycle"] + [f"sensor_{i}" for i in range(1, df.shape[1]-2)] + ["source"]
    df.columns = cols

    return df


if __name__ == "__main__":
    df = load_combined()
    df.to_csv("data/cmapss_combined.csv", index=False)
    print("Combined dataset ready!")