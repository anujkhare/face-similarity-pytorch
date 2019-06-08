import pandas as pd
import pathlib
from typing import Tuple


def get_df_from_folder(path: str) -> pd.DataFrame:
    all_data = []
    for ix, p in enumerate(pathlib.Path(path).glob('**/*.jpg')):
        all_data.append({
            'path': p.absolute(),
            'label': p.parent.name,
            'idx': ix,
        })
    df = pd.DataFrame(all_data)
    return df


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=1111)
    n = len(df)
    n_train = int(0.8 * n)
    df_train, df_val = df.iloc[:n_train], df.iloc[n_train:]
    return df_train, df_val
