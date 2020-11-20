import pandas as pd

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# <img src="img_notes/histo_harga_data_choosen.png">

df = pd.read_csv('merged_data - data_choosen.csv')

count_area = df.groupby(['area']).size().reset_index(name='counts')
observed_area = list(count_area[count_area['counts']>=4]['area'].values)

df_observed = df[df['area'].isin(observed_area)].reset_index(drop=True)

df_observed['kost_name_rough'] = df_observed['texts'].apply(lambda x: ' '.join(x.split()))
df_observed['fasilitas_ls'] = df_observed['fasilitas']\
.apply(lambda x: str(x).replace("'","").replace("[","").replace("]","").split(","))

# lowercase and strip
def lowstrip(x):
	ls = []
	for val in x:
		ls.append(val.lower().strip())
	return ls

df_observed['fasilitas_lsclean'] = df_observed['fasilitas_ls'].apply(lambda x: lowstrip(x))
# df_observed['fasilitas_lsclean_str'] = df_observed['fasilitas_lsclean'].apply(lambda x: str(x))

df_observed = df_observed[['kost_name_rough', 'kota', 'type_kos', 'area', 
                           'fasilitas_lsclean', 'harga_nomina']]

# lowercase 'type_kos', 'area'
for i in [i for i in df_observed.columns][2:4]:
    df_observed[i] = df_observed[i].apply(lambda x: x.lower())

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

df_observed['k_mndi_dlm'] = df_observed['fasilitas_lsclean'].apply(lambda x: check_kmd(x))
df_observed['wifi'] = df_observed['fasilitas_lsclean'].apply(lambda x: check_wifi(x))
df_observed['aks_dmpat_jam'] = df_observed['fasilitas_lsclean'].apply(lambda x: check_akdjam(x))

df_observed['facility_score'] = df_observed['k_mndi_dlm'] + df_observed['wifi'] + df_observed['aks_dmpat_jam']

df_features = df_observed[['kost_name_rough','kota','type_kos',
                           'area','facility_score','harga_nomina']]

df_features.to_csv('features.csv',index=False)


