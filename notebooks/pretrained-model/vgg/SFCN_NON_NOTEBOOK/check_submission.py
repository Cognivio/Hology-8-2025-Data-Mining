import pandas as pd

df = pd.read_csv('submission_fixed.csv')
print('Submission Statistics:')
print(f'Total images: {len(df)}')
print(f'Min prediction: {df["predicted_count"].min()}')
print(f'Max prediction: {df["predicted_count"].max()}')
print(f'Mean prediction: {df["predicted_count"].mean():.1f}')
print(f'Zero predictions: {(df["predicted_count"] == 0).sum()}')
print(f'Non-zero predictions: {(df["predicted_count"] > 0).sum()}')
print(f'Percentage non-zero: {(df["predicted_count"] > 0).mean() * 100:.1f}%')