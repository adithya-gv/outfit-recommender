import pandas as pd
from fastai.tabular.all import * 
from fastai.learner import save_model, load_model, mk_metric


learn = load_learner("data/training/model.pkl")

df = pd.read_csv("data/articles.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
print(df)

row, cluster, prob = learn.predict(df.iloc[36010])

print("row", row)
print("cluster", cluster)
print("prob", prob)


