import pandas as pd
import re


def main():
    df = pd.read_csv('merged_data - Sheet7.csv')

    df = df[1:]

    df = df.reset_index(drop=True)


    def get_type_kos(x):
        try:
            gotdata = x.split('\n \n\t')[1].split('\n    \n\t\t\t\t')[0]
        except IndexError:
            gotdata = x
        return gotdata

    def get_area(x):
        try:
            gotdata = x.split('\n \n\t')[1].split('\n    \n\t\t\t\t')[1]\
            .split('\n\t\t\t')[1].strip()
        except IndexError:
            gotdata = x
        return gotdata

    def get_fasilitas(x):
        try:
            gotdata = x.split('\n \n\t')[1].split('\n    \n\t\t\t\t')[1]\
            .split('\n\t\t\t')[2].replace('\t','').split(' • ')
        except IndexError:
            gotdata = x
        return gotdata

    def get_harga(x):
        try:
            gotdata = x.split('\n \n\t')[1].split('\n    \n\t\t\t\t')[1]\
            .split('\n\t\t\t')[4].split(' ')[1].replace('.','')
        except IndexError:
            gotdata = x
        return gotdata


    df['type_kos'] = df['texts'].apply(lambda x: get_type_kos(x) if isinstance(x, float) is False else "no-type")

    df['area'] = df['texts'].apply(lambda x: get_area(x) if isinstance(x, float) is False else "no-area")

    df['fasilitas'] = df['texts'].apply(lambda x: get_fasilitas(x) if isinstance(x, float) is False else "no-fasilitas")

    df['harga'] = df['texts'].apply(lambda x: get_harga(x) if isinstance(x, float) is False else "no-harga")

    df['harga_nomina'] = df['harga'].apply(lambda x: int(x) if x.isdigit() is True else x)

    df['tes_len'] = df['harga_nomina'].apply(lambda x: 'string' if type(x) is str else 'int')

    df_tr1 = df[['texts', 'kota', 'type_kos', 'area', 'fasilitas', 'harga',
           'harga_nomina']][df.tes_len == 'int'].reset_index(drop=True)

    df_1 = df[['texts', 'kota', 'type_kos', 'area', 'fasilitas', 'harga',
           'harga_nomina']][df.tes_len == 'string'].reset_index(drop=True)

    def get_type_kos(x):
        try:
            gotdata = x.split('\n    \n\t\t\t\t')[0]
        except IndexError:
            gotdata = x
        return gotdata

    def get_area(x):
        try:
            gotdata = x.split('\n    \n\t\t\t\t')[1].split('\n\t\t\t  ')[1].split(' \n\t\t\t\t')[0]
        except IndexError:
            gotdata = x
        return gotdata

    def get_fasilitas(x):
        try:
            gotdata = x.split('\n    \n\t\t\t\t')[1].split('\n\t\t\t  ')[1].split(' \n\t\t\t\t')[1].split(' · ')
        except IndexError:
            gotdata = x
        return gotdata

    def get_harga(x):
        try:
            gotdata = x.split('\n\t\t\t  ')[2].split(' ')[4].split()[0].replace('.','')
        except IndexError:
            gotdata = x
        return gotdata

    df_1['type_kos'] = df_1['texts'].apply(lambda x: get_type_kos(x) if isinstance(x, float) is False else "no-type")

    df_1['area'] = df_1['texts'].apply(lambda x: get_area(x) if isinstance(x, float) is False else "no-area")

    df_1['fasilitas'] = df_1['texts'].apply(lambda x: get_fasilitas(x) if isinstance(x, float) is False else "no-fasilitas")

    df_1['harga'] = df_1['texts'].apply(lambda x: get_harga(x) if isinstance(x, float) is False else "no-harga")

    df_1['harga_nomina'] = df_1['harga'].apply(lambda x: int(x) if x.isdigit() is True else x)

    df_1['tes_len'] = df_1['harga_nomina'].apply(lambda x: 'string' if type(x) is str else 'int')

    df_1[df_1.tes_len == 'string']

    df_tr2 = df_1[['texts', 'kota', 'type_kos', 'area', 'fasilitas', 'harga',
           'harga_nomina']][df.tes_len == 'int'].reset_index(drop=True)

    df_all = pd.DataFrame()
    df_all = pd.concat([df_tr1,df_tr2])

    df_all.to_csv('transform_2.csv', index=False)


if __name__ == '__main__':
    main()
