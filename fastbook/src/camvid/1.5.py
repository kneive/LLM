from fastai.vision.all import *
from pathlib import Path

def _image(o):
	return path/'labels'/f'{o.stem}_P{o.suffix}'

path = untar_data(URLs.CAMVID_TINY)

dls = SegmentationDataLoaders.from_label_func(path,
                                              bs=8,
											  fnames=get_image_files(path/"images"),
											  label_func=_image,
											  codes=np.loadtxt( path/'codes.txt',
                              									dtype=str))

learn = unet_learner(dls, resnet34)
learn.path=Path(__file__).resolve().parent/'..'/'..'/'models'
learn.fine_tune(8)
learn.show_results(max_n=6, figsize=(7,8))

if not Path(learn.path/'vision.pkl').exists():
    learn.export('vision.pkl')
