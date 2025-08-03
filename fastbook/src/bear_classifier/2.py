##################
## Prepare data ##
##################

from utils import *
from fastai.vision.utils import download_images, verify_images, get_image_files
from pathlib import Path
# from azure.identity import DefaultAzureCredential

data_path = Path(Path(__file__).resolve().parent/'..'/'..'/'training_sets')
model_path = Path(__file__).resolve().parent/'..'/'..'/'models'

key = os.environ['AZURE_SEARCH_KEY']
bear_types= 'grizzly', 'black', 'teddy'

# download images from a bing search if there is no training set 
# (requires an azure search api key)
i = 0
if not data_path.exists():
	data_path.mkdir()
	for o in bear_types:
		i = i+1
		print(f"Downloading images for {o} ({i}/{len(bear_types)})")
		dest = (data_path/o)
		dest.mkdir(exist_ok=True)
		results = search_images_bing(key, term=f"{o} bear", total_count=400)
		download_images(dest, urls=results, n_workers=0)

# collect all images in a single list
fns = get_image_files(data_path)

# verify whether any images are corrupted
failed = L()
for o in fns:
	if verify_images(o) == False:
		failed.append(o)

# remove failed images by unlinking them from the Path object
failed.map(Path.unlink)

###########
## Model ##
###########

from fastai.data.block import *
from fastai.data.transforms import *
from fastai.vision.data import *
from fastai.vision.augment import *
from fastai.vision.learner import *
from torchvision.models import *
from fastai.metrics import *
from fastai.callback.schedule import *

bears = DataBlock(blocks=(ImageBlock, CategoryBlock),
				  get_items=get_image_files,
				  splitter=RandomSplitter(valid_pct=0.2, seed=42),
				  get_y=parent_label,
				  item_tfms=Resize(128))

bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5),
				  batch_tfms=aug_transforms())

dls = bears.dataloaders(data_path)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.path = model_path
learn.fine_tune(4)

if not Path(model_path/'bears.pkl').exists():
    learn.export('bears.pkl')