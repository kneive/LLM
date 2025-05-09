# model predicting income based on socioeconomic factors using tabular dataloader
from fastai.tabular.all import *
from pathlib import Path

model_path = Path(__file__).resolve().parent/'..'/'..'/'models'

path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv',
								  path=path,
								  y_names="salary",
								  cat_names=['workclass',
								  			 'education',
								  			 'marital-status',
								  			 'occupation',
								  			 'relationship',
								  			 'race'],
								  cont_names=['age',
								  			  'fnlwgt',
								  			  'education-num'],
								  procs=[Categorify,
								  		 FillMissing,
								  		 Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.path = model_path
learn.fit_one_cycle(3)
learn.show_results(max_n=6)

if not Path(model_path/'income_prediction.pkl').exists():
    learn.export('income_prediction.pkl')