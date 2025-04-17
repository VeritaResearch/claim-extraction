import pandas as pd

def process_AVeriTeC():
    full_path = '../data/AVeriTeC/train.json'
    df = pd.read_json(full_path)
    df = df.filter(items=['claim'])
    df = df.rename(columns={'claim': 'text'})
    df['label'] = 1
    return df

def process_PoliClaim():
    path = '../data/PoliClaim/'
    file_names = ['AL2003_G4_1.xlsx','CT2014_G4_1.xlsx','DE1999_G4_1.xlsx','DE2021_G4_1.xlsx','IN2001_G4_1.xlsx','IN2011_G4_1.xlsx','KY2018_G4_1.xlsx','US2016_G4_1.xlsx']
    dfs = []
    for f in file_names:
        dfs.append(pd.read_excel(path+f))
    df = pd.concat(dfs)
    df = df.filter(items=['SENTENCES','golden'])
    df = df.rename(columns={'SENTENCES': 'text', 'golden': 'label'})
    return df

def process_Claimbuster():
    full_path = '../data/Claimbuster/full.json'
    df = pd.read_json(full_path)
    df = df.filter(items=['text','label'])
    return df