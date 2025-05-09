# classifier for cats and dogs

from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

path = untar_data(URLs.PETS)/'images'

dls = ImageDataLoaders.from_name_func(path,
                                      get_image_files(path), 
                                      valid_pct=0.2,
                                      seed=42,
                                      label_func=is_cat,
                                      item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune

categories = ('Dog', 'Cat')

# test image

im = PILImage.create('/test_images/gadse1.jpg')
im.thumbnail((224, 224))
classify_image(im)
is_cat,_,probs = learn.predict(im)

print(f"is this a cat? {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
print(f"Probability it's a dog: {probs[0].item():.6f}")