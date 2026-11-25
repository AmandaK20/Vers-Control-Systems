import glob
import os
import pandas as pd


def standardize_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    rename_map = {
        'model_name': 'model',
        'classifier': 'model',
        'regressor': 'model',
        'f1': 'f1_score',
        'auc': 'roc_auc'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df['source_file'] = os.path.basename(source_file)
    return df


def main():
    csv_files = glob.glob('results/*.csv')
    csv_files = [f for f in csv_files if 'final_comparison' not in f]

    if not csv_files:
        raise FileNotFoundError("No result CSVs found in results/.")

    frames = [standardize_columns(pd.read_csv(f), f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    os.makedirs('results', exist_ok=True)
    combined.to_csv('results/final_comparison_summary.csv', index=False)
    print("Combined results saved to results/final_comparison_summary.csv\n")
    print(combined.to_string(index=False))

    if 'task' in combined.columns:
        print("\n--- Best Model Per Task ---")
        for task, group in combined.groupby('task'):
            if 'f1_score' in group.columns and group['f1_score'].notna().any():
                best = group.sort_values('f1_score', ascending=False).iloc[0]
                print(f"Task: {task:18s} | Best: {best['model']} (F1={best['f1_score']})")
            elif 'rmse' in group.columns and group['rmse'].notna().any():
                best = group.sort_values('rmse', ascending=True).iloc[0]
                print(f"Task: {task:18s} | Best: {best['model']} (RMSE={best['rmse']}, MAE={best['mae']})")


if __name__ == '__main__':
    main()