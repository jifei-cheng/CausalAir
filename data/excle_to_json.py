import pandas as pd
import json

# Read Excel file
df = pd.read_excel("narratives-pre2008.xlsx")

# Clean text
def clean_text(x):
    if isinstance(x, str):
        return (
            x.replace("_x000d_", "")
             .replace("_x000a_", "")
             .replace("\r", "")
             .replace("\n", " ")
        )
    return x

df = df.applymap(clean_text)

# Convert Timestamp to string (to avoid JSON errors)
df = df.applymap(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, "strftime") else x)

# Convert to dict
data = df.to_dict(orient="records")

# Beautify JSON output
json_str = json.dumps(data, ensure_ascii=False, indent=4)

# Save file
with open("narratives-pre2008.json", "w", encoding="utf-8") as f:
    f.write(json_str)

print("Conversion complete!")
