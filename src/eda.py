"""Simple EDA script for `data/startup_funding.csv`.
Generates a short text report and saves a histogram of `Amount in USD` if available.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src/fetch_data.py first to download data.")
    return pd.read_csv(path, encoding='latin1')


def summarize(df):
    lines = []
    lines.append(f"Rows: {len(df)}")
    lines.append("\nColumns and non-null counts:")
    lines.append(str(df.info(buf=None)))
    lines.append("\nSummary statistics:")
    lines.append(str(df.describe(include='all').transpose()))
    return "\n".join(lines)


def main():
    data_path = os.path.join('data', 'startup_funding.csv')
    out_dir = os.path.join('outputs')
    os.makedirs(out_dir, exist_ok=True)

    df = safe_read(data_path)

    # Normalize common amount column names
    amount_cols = [c for c in df.columns if 'amount' in c.lower() or 'raised' in c.lower() or 'amt' in c.lower()]

    report = summarize(df)
    report_path = os.path.join(out_dir, 'eda_report.txt')
    with open(report_path, 'w', encoding='utf8') as f:
        f.write(report)

    print('EDA report written to', report_path)

    # Try to parse a numeric amount column and plot
    for col in amount_cols:
        try:
            series = df[col].astype(str).str.replace('[^0-9.]', '', regex=True)
            series = pd.to_numeric(series, errors='coerce')
            if series.notna().sum() > 0:
                plt.figure(figsize=(6,4))
                series.dropna().hist(bins=50)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('count')
                img_path = os.path.join(out_dir, f'{col}_hist.png')
                plt.savefig(img_path, bbox_inches='tight')
                plt.close()
                print('Saved histogram to', img_path)
                break
        except Exception:
            continue


if __name__ == '__main__':
    main()
