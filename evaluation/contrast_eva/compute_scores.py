"""
Calculate the Average Scores of Results
"""
import os
import json

def compute_average_scores(folder_path):
    print(f" Analyzing folder: {folder_path}\n" + "="*40)

    for filename in os.listdir(folder_path):
        # Ignore non-json files and _fail.json files
        if filename.endswith(".json") and not filename.endswith("_fail.json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Unable to read file: {filename}")
                    continue

            # Use two dictionaries to track: total scores and valid count
            metric_sums = {}
            metric_counts = {}

            total_items = len(data)

            # Iterate over each data item
            for item in data:
                scores = item.get("scores", {})
                
                # Iterate over each scoring dimension of the data
                for k, v in scores.items():
                    # Key checks:
                    # 1. v cannot be None
                    # 2. v must be a number (int or float), to avoid interference from "error" field strings
                    if v is not None and isinstance(v, (int, float)):
                        metric_sums[k] = metric_sums.get(k, 0) + v
                        metric_counts[k] = metric_counts.get(k, 0) + 1

            # Calculate averages
            avg_scores = {}
            for k, total_score in metric_sums.items():
                count = metric_counts.get(k, 0)
                if count > 0:
                    avg_scores[k] = total_score / count
                else:
                    avg_scores[k] = 0.0

            # --- Output results ---
            print(f"\nFile: {filename}")
            print(f"   (Total data rows: {total_items})")
            
            # For readability, we sort the output by key name
            for k in sorted(avg_scores.keys()):
                # Print the valid sample count for reference
                valid_count = metric_counts.get(k, 0)
                print(f"   - {k:<20}: {avg_scores[k]:.4f} (Sample count: {valid_count})")
            
            print("-" * 40)

# Example usage
if __name__ == "__main__":
    # Ensure the path is correct
    folder = "evaluation/contrast_eva/eva_results" 
    
    if os.path.exists(folder):
        compute_average_scores(folder)
    else:
        print(f"Error: Folder {folder} not found")
