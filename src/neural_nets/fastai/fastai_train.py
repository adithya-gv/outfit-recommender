import pandas as pd
from fastai.tabular.all import * 
from fastai.learner import save_model, load_model, mk_metric, Recorder, save
# from fastai.tabular import *
df = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no','cluster']] # you can add more features
print(df)
cat_names = ['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']
procs = [Categorify, FillMissing, Normalize]


dls = TabularDataLoaders.from_df(df, '.', procs=procs,cat_names=cat_names, y_names="cluster",y_block=CategoryBlock(), bs=32)

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(10)
learn.show_results()

learn.export('models/model.pkl')
print(learn.recorder)
learn.recorder.plot_loss()



# row, clas, probs = learn.predict(df.iloc[0])