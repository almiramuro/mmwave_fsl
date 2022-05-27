import pandas as pd

if __name__ == "__main__":
    with open('all_points.csv', 'r') as points:
        df = pd.read_csv(points)
    
    print(df.describe())
    print()
    print(df.quantile(0.90))
    print(df.quantile(0.10))