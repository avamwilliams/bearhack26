import pandas as pd
import numpy as np

df = pd.read_csv("wustl-ehms-2020_with_attacks_categories.csv")
df.columns = df.columns.str.strip()
print("Shape:", df.shape)

drop = ['SrcMac', 'DstMac', 'Dir', 'SrcAddr', 'DstAddr', 'Sport', 'Label']
df = df.drop(columns=drop)
bio_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
#TODO: one hot encode this later
label_column = 'Attack Category'
network_column = [col for col in df.columns if col not in bio_columns + [label_column]]

print(f"Network features:{len(network_column)}")
print(f"Bio features:{len(bio_columns)}")
print("Shape:", df.shape)

