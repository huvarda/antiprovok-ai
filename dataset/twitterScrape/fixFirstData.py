import pandas as pd

df = pd.read_csv("output/outputNewCategories0Fixed.csv")
print("neither or political or provocative? n/p")
for i in range(len(df)):
    if df["label"][i] == "neither":
        newValue = input(str(df["example"][i])+" (n/p/pr)?: ")
        if newValue == "p":
            df.at[i, "label"] = "political"
        if newValue == "pr":
            df.at[i, "label"] = "provocative"
        else:
            df.at[i, "label"] = "neither"
            
df.to_csv("output/outputNewCategoriesFixAmbiguous.csv", columns=["example", "label"], index=None)