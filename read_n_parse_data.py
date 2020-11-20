import pandas as pd
import re


def get_sisa_kamar(x):
    try:
        gotdata = x.split('\n \n\t')[1].split('\n   \n\t\t\t\t')[0]
    except IndexError:
        gotdata = x
    return gotdata

def get_area(x):
    try:
        gotdata = x.split('\n \n\t')[1]\
        .split('\n   \n\t\t\t\t')[1].split('\n\t\t\t')[1].strip()
    except IndexError:
        gotdata = x
    return gotdata

def get_fasilitas(x):
    try:
        gotdata = x.split('\n \n\t')[1]\
        .split('\n   \n\t\t\t\t')[1].split('\n\t\t\t')[2].replace('\t','').split('·')
    except IndexError:
        gotdata = x
    return gotdata

def get_harga(x):
    try:
        gotdata = x.split('\n \n\t')[1]\
        .split('\n   \n\t\t\t\t')[1].split('\n\t\t\t')[4].replace('\t\t\t\tRp ','')
    except IndexError:
        gotdata = x
    return gotdata

def main():

    df = pd.read_csv('merged_kota_drop_dupl.csv')

    # isinstance(message, bytes)
    df['type_kos'] = df['texts'].apply(lambda x: x.split('\n \n\t')[0] \
                        if isinstance(x, float) is False else "no-type")
        
    df['sisa_kamar'] = df['texts'].apply(lambda x: get_sisa_kamar(x) \
                        if isinstance(x, float) is False else "no-sisa-kamar")

    df['area'] = df['texts'].apply(lambda x: get_area(x) \
                    if isinstance(x, float) is False else "no-area")

    df['fasilitas'] = df['texts'].apply(lambda x: get_fasilitas(x) \
                        if isinstance(x, float) is False else "no-fasilitas")

    df['harga'] = df['texts'].apply(lambda x: get_harga(x) \
                    if isinstance(x, float) is False else "no-harga")

    df['harga_nomina'] = df['harga'].apply(lambda x: int(x.replace('.','')) \
                    if x.replace('.','').isdigit() is True else x)

    df.to_csv("transform_1.csv", index=False)


if __name__ == '__main__':
    main()


