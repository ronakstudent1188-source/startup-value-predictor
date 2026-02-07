"""Generate a realistic synthetic startup dataset for modeling."""
import os
import pandas as pd
import numpy as np


def generate_startup_data(n_rows=500, random_state=42):
    """Generate a realistic synthetic startup dataset."""
    rng = np.random.RandomState(random_state)
    
    data = {
        'startup_name': [f'Startup_{i}' for i in range(n_rows)],
        'founded_year': rng.randint(2010, 2024, size=n_rows),
        'team_size': rng.exponential(scale=15, size=n_rows).astype(int) + 1,
        'funding_rounds': rng.randint(0, 8, size=n_rows),
        'total_funding_usd': rng.exponential(scale=2e6, size=n_rows),
        'industry': rng.choice(['SaaS', 'AI/ML', 'Fintech', 'Healthcare', 'E-Commerce', 'Edtech', 'ClimTech'], size=n_rows),
        'country': rng.choice(['USA', 'India', 'UK', 'Canada', 'Germany', 'Singapore'], size=n_rows),
        'has_vc_backing': rng.choice([0, 1], size=n_rows, p=[0.4, 0.6]),
        'months_since_founding': (2024 - rng.randint(2010, 2024, size=n_rows)) * 12,
    }
    
    df = pd.DataFrame(data)
    
    # Create valuation as synthetic target (correlated with features)
    df['valuation_usd'] = (
        100000 +
        df['total_funding_usd'] * 3.5 +
        df['team_size'] * 50000 +
        df['funding_rounds'] * 500000 +
        df['has_vc_backing'] * 2000000 +
        rng.normal(0, 1e6, size=n_rows)
    )
    
    # Binary success column (funding > median and growth signal)
    df['is_successful'] = ((df['valuation_usd'] > df['valuation_usd'].median()) & 
                            (df['funding_rounds'] >= 2)).astype(int)
    
    return df


def main():
    out_path = os.path.join("data", "startup_funding.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print("Generating synthetic startup dataset...")
    df = generate_startup_data(n_rows=500)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
