from datasets import load_dataset

ds = load_dataset("ag_news")

print(ds)
print(ds["train"][0])
print("Number of labels:", len(set(ds["train"]["label"])))