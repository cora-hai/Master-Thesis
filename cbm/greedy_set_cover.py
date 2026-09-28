import pandas as pd 

def get_cover(sub_df: pd.DataFrame) -> list[str]:

    cover = []

    # check if either every molecule is covered or each substructure has been checked
    while len(sub_df) != 0 and sub_df.sum().max() != 0:

        # get name of substructure with maximal support
        c = sub_df.sum().idxmax()

        # append to cover
        cover.append(c)

        # drop those substructures that have already been covered
        sub_df.drop(labels = sub_df[sub_df[c] == 1].index, inplace = True)
    
    return cover


if __name__ == "__main__":

    infile = "dili/krfp/train_dili.csv"

    df = pd.read_csv(infile)
    # select columns with substructure annotations only
    sub_df = df.iloc[ : , 3: ]
    
    cover = get_cover(sub_df)
    print(cover)
    print(f"# selected substructures = {len(cover)}")
    