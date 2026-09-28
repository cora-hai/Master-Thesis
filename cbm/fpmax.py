import pandas as pd 
from mlxtend.frequent_patterns import fpmax

def run_fpmax(sub_df: pd.DataFrame, min_support) -> list:

    # get dataframe of itemsets and their supports
    result = fpmax(sub_df, min_support = min_support, use_colnames = True)

    if len(result) == 0:

        return []

    else:
        # get itemsets with maximal support
        max_sets = result.loc[result["support"] == result["support"].max()]["itemsets"]

        if len(max_sets) == 1:
            # if there is a unique solution, return that
            return list(max_sets.item())

        else:
            # if multiple itemsets have maximal support, merge them and return merged set
            result = []
            for ms in max_sets:
                result.extend(list(ms))

            return list(set(result))
        


if __name__ == "__main__":
    infile = "dili/bioalerts/train_dili.csv"
    min_support = 0.6

    df = pd.read_csv(infile)
    sub_df = df.iloc[ : , 3: ].astype(bool)
    cover = run_fpmax(sub_df, min_support)
    print(cover)