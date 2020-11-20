import os
from glob import glob
from tqdm import tqdm
import pandas as pd


def main():
    PATH = os.path.join(os.getcwd(),'SCRAPPINGDATA')
    kotas = [subdir for _ , subdir , _ in os.walk(PATH)][0]

    # read df in each kota
    dict_df = {}
    for kota in kotas:
        dict_df[kota] = []
        for file in glob(os.path.join(PATH, kota, "*.csv")):         
            dict_df[kota].append(pd.read_csv(file))

    # merge df in each kota
    dict_df_kota = {}
    for kota in kotas:
        df_kota_tmp = pd.DataFrame()
        for df in dict_df[kota]:
            df_kota_tmp = pd.concat([df_kota_tmp,df.iloc[:,2]])
        dict_df_kota[kota] = df_kota_tmp
        dict_df_kota[kota]['kota'] = kota
        dict_df_kota[kota].rename(columns = {0:"texts"}, inplace = True)

    df_all = pd.DataFrame()

    # concat all df from DF_ALL into df_new vertically
    for _ , v in dict_df_kota.items():
        df_all = pd.concat([df_all,v])

    df_all = df_all.drop_duplicates()

    df_all.to_csv("merged_kota_drop_dupl.csv",index=False)


if __name__ == '__main__':
    main()


