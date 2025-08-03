# model predicting movie ratings based on the MovieLens dataset using a collab dataloader 

from fastai.collab import *
from pathlib import Path

model_path = Path(__file__).resolve().parent/'..'/'..'/'models'

path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')

learn = collab_learner(dls, y_range=(0.5, 5.5))
learn.path = model_path
learn.fine_tune(10)
learn.show_results(max_n=6)

if not Path(model_path/'movieLens.pkl').exists():
    learn.export('movieLens.pkl')