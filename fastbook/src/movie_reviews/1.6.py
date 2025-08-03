# classifier for movie reviews using an IMDB dataset 

from fastai.text.data import *
from fastai.text.learner import *
from fastai.text.models import *
from pathlib import Path

model_path = Path(__file__).resolve().parent/'..'/'..'/'models'

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB),
                                  valid='test')

learn=text_classifier_learner(dls,
                              AWD_LSTM,
                              drop_mult=0.5,
                              metrics=accuracy)


learn.path = model_path
learn.fine_tune(4, 1e-2)
learn.predict("I really liked that movie")

if not Path(model_path/'movie_reviews.pkl').exists():
    learn.export('movie_reviews.pkl')