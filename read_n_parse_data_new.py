import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


df = pd.read_excel('MERGED_DATA_BARU.xlsx',sheet_name='Sheet3',engine='openpyxl')

df['kota_cln'] = df['kota'].apply(lambda x: x.split()[0])


def join_string(list_string):
	# Join the string based on '' delimiter
	string = ''.join(list_string)
	return string

def get_max_price(str_txt):
    ls_cek = re.findall(r'[0-9.]+', str_txt)
    return max([int(join_string(s.split('.'))) for s in ls_cek])


df['price_cln_max'] = df['harga'].apply(lambda x: get_max_price(x))

sns.displot(df, x="price_cln_max")

plt.hist(df['price_cln_max'], density=True, bins=30)  # `density=False` would make counts
plt.ylabel('count')
plt.xlabel('price_range');

df['fasilitas_kos_split'] = df['fasilitas_kos'].apply(lambda x: x.split('·') if isinstance(x, float) is False else "no-fasilitas")

# exclude price above 5.000.000
df_obs = df[df.price_cln_max <= 5000000].reset_index(drop=True)

# lowercase and strip
def lowstrip(x):
	ls = []
	for val in x:
		ls.append(val.lower().strip())
	return ls


df_obs['fasilitas_kos_split_low'] = df_obs['fasilitas_kos_split'].apply(lambda x: lowstrip(x))

count_area = df_obs.groupby(['area']).size().reset_index(name='counts')
observed_area = list(count_area[count_area['counts']>=4]['area'].values)

df_observed = df_obs[df_obs['area'].isin(observed_area)].reset_index(drop=True)

df_observed = df_observed[['nama_kos', 'jenis_kos', 'area', 'kota_cln', 'price_cln_max', 'fasilitas_kos_split_low']]

for i in [i for i in df_observed.columns][1:3]:
    df_observed[i] = df_observed[i].apply(lambda x: x.lower())

# replace'|penuh'
df_observed['jenis_kos'] = df_observed['jenis_kos'].apply(lambda x: x.replace('|penuh',''))


# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

df_observed.groupby(['area']).size().reset_index(name='counts')

dict_score = {'k. mandi dalam':2, 'wifi':2, 'akses 24 jam':2}

def check_kmd(x):
    if 'k. mandi dalam' in x:
        return dict_score['k. mandi dalam']
    else:
        return 0

def check_wifi(x):
    if 'wifi' in x:
        return dict_score['wifi']
    else:
        return 0

def check_akdjam(x):
    if 'akses 24 jam' in x:
        return dict_score['akses 24 jam']
    else:
        return 0

df_observed['k_mndi_dlm'] = df_observed['fasilitas_kos_split_low'].apply(lambda x: check_kmd(x))
df_observed['wifi'] = df_observed['fasilitas_kos_split_low'].apply(lambda x: check_wifi(x))
df_observed['aks_dmpat_jam'] = df_observed['fasilitas_kos_split_low'].apply(lambda x: check_akdjam(x))

df_observed['facility_score'] = df_observed['k_mndi_dlm'] + df_observed['wifi'] + df_observed['aks_dmpat_jam']

df_features = df_observed[['nama_kos', 'kota_cln', 'jenis_kos', 'area', 'facility_score', 'price_cln_max']]

df_features = df_features.rename(columns={'nama_kos':'kost_name_rough', 'kota_cln':'kota', 'jenis_kos':'type_kos', 'area':'area', 'facility_score':'facility_score', 'price_cln_max':'harga_nomina'})

df_features.to_csv('features.csv',index=False)











