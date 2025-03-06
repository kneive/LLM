"""History

1943 Warren McCulloch (neurophysiologist) and Walter Pitts (logician) developed 
    first model of an artificial neuron
    paper: "A logical calculus of the Ideas Immanen in nerous Activity" [1]

    insight:    nervous activity has a binary nature, therefore it can be treated 
                by propositional logic

Frank Rosenblatt improves the model giving it the ability learn: 

    Mark I Perceptron can recognize simple shapes

1969 Minsky and Papert "Perceptrons" [2]: a single layer of perceptrons fail at 
    simple mathematical functions

    solution: using multiple layers

1986 Rumelhart, McClelland, PDP Research Group: "Parallel Distributed Processing" [3]
    requirements for parallel distributed processing:
        1) set of processing unites
        2) state of activation
        3) output function for each unit
        4) pattern of connectivity among units
        5) propagation rule for propagating patterns of activities (through network)
        6) activation rule for combining inputs impinging on a unit with the 
            current state of that unit to produce an output for the unit

        7) learning rule (modifiying patterns of activity by experience)
        8) environment the unit interacts with

1980 models are build with 2 layers of neurons (response to minsky and papert)
    models are used in 80s and 90s for practical applications

problem: 2 layer neural networks often too big and slow

problem: good practical results need even more layers



[4] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
"""
################################################################################
""" first model

best choice GPU server: https://www.kaggle.com/

dataset Oxford IIIT Pet Dataset (7,349 images of cats and dogs, 37 breeds)
"""

#fine tuning the model (classifier)

from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): 
    return x[0].usupper()

dls = ImageDataLoaders.from_name_func(path, get_image_files(path), 
                                      valid_pcrt=-0.2, seed=42, 
                                      label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

# testing the model with a new picture
img = PILImage.create("filename")
is_cat,_,probs = learn.predict(img)
print("Is this a cat?: {is_cat}.")
print("Probability it's a cat: {probs[1].item():.6f}")


"""Steps for recognizing objects in  images:
1949 Arthur Samuel: machine learning (as different way for computers to complete tasks)
1962 Samuel: "Artificial Intelligence: A Frontiwer of Automation" [5]

    basic idea: 1) show computer program examples of problems to solve and
                2) let the program figure out how to solve the problems
                    a)  provide automatic means to test the effectiveness of current
                        weight assignment (in terms of actual performance)
                    b)  provide mechanism to change weight assignment

    1961 Samuels checkers program beat Conneticut state champion

concepts of machine Learning implied by Samuel:

    1)  idea of a weight assignment
    2)  every weight assignment has an actual performance
    3)  requirement for automatic means to test the performance of a 
        weight assignment
    4)  mechanism for improving performance by changing weight assignments

weights:            variables
                    also input (to the model)
                    
weight assignments: particular choice of values for each variable (weight)    

note:   Samuel's weights are called model parameters today.
        today: weights are a particular type of model parameter

note:   Samuels idea of training machine learning model: figure-1 
        Samuels model distinguished between Results and Performance

note:   the weights are only part of the model AFTER IT HAS BEEN TRAINED, 
        i.e. the final weight assignment is chosen

a trained model can be treated like a regular program        
"""

################################################################################
""" neural network

Problem 1:    flexible function that can be used to solve any problem by simply 
            varying its weights

solution:   neural network (is such a function)

mathematical proof: universal approximation theorem shows this function can 
                    solve any problem to any level of accuracy

Problem 2:  general way to update the weights of a neural network

solution:   stochastik gradient descent (SGD)

possible performance measure: accuracy of predicting correct answers 
"""

################################################################################

""" Jargon

architecture:   functional form of the model
                note: sometimes model and architecture are used synonymously
parameters:     weights

independent variable:   data without labels
dependent variable:     data with labels (also called targets)

predictions:    results of the model
                note: calculated from the  independent variable

loss:           measure of performance 
                note: loss depends on predictons and correct labels

updated training loop: Figure [2]
"""

################################################################################

"""Limitations of machine learning

1) models need data
2) models learn can only learn to operate on patterns in the training data
3) model learning can only create predictions not recommended actions
4) training data needs labels

note:   not enough data often means not enough labeled data (i.e. organizations 
        interested in implementing a model presumably have inputs to run the 
        model against, i.e. data)

note:   models only make predictions, i.e. replicate labels

problem:    there can be a gap between capabilities of a model and 
            organizational goals
            example: a recommendation system might predict what a user might 
            purchase based on their buying history leading to recommendations 
            the user probably already knows about or has (not new products)

feedback loop: predictive policing example (similar for video recommendation )
            1) predictive policing based on past arrests is not predicting crime 
                but arrest which might have been biased
            2) usage of the model might lead to increased policing and more 
                arrests in those areas
            3) new arrests are used as data to further train the model
"""

################################################################################

"""Explanation of the Image recognizer code:"""

from fastai.vision.all import *

"""
import * is bad practice (acceptable for interactive use)

fastai library is designed to support interactive use, i.e. import only 
    necessary functions
"""

path = untar_data(URLs.PETs)/'images'

"""
downloads the STANDARD DATASET from the fast.ai datasets collection

untar_data EXTRACTS the dataset and RETURNS a PATH OBJECT (of extracted location)
"""

def is_cat(x): return x[0].isupper()

"""
is_cat(x) labels cats based on a FILENAME RULE, i.e. filnames of cats start 
    with an uppercase letter (note: Image data is usually structured in a way  
    that the label is part of the filename / path)
"""

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

"""
ImageDataLoaders.from_name_func() creates an ImageDataLoader object (for Images)

label_func = is_cat specifies the FUNCTION TO USE FOR LABELING (note: fastai 
    has a number of standardized labeling methods)

item_tfms = Resize(224) specifies the needed ITEM TRANSFORMATIONS, i.e. code 
    applied automatically during training
       item_tfms (applied to each item) and 
       batch_tfms (applied to batches of items, using GPU)

Resize(224) standard size for historical reasons

    note:   ANY SIZE can be used and BIGGER SIZE of translate into BETTER models, 
            at the COST of SPEED and MEMORY usage

classification mode:    attempts to predict a class / category from a number of 
                        discrete possibilities
regression model:       attempts to predict one or more numeric quantities 
                        (temperature, location,...)

from_name_func() specifies HOW to get FILENAMES (here by applying a function, 
    i.e. is_cat which returns true if x[0].isupper())

valid_pct=0.2 specifies the SIZE of the VALIDATION SET (i.e. 20% of the data)
    fastai defaults valid_pct=0.2 if not specified

seed=42 specifies the RANDOM SEED to select the VALIDATION SET 
    choosing a fixed number ensures REPROUCIBILITY (same validation set for 
    every run)

    REPRODUCIBILITY is important to ensure differences in performance result #
        from changes in the model, not the data

        
overfitting:    if a model is trained for too long (too many epochs) it will
                memorize the training data and not find generalizable patterns
                i.e. the model will perform well on the training data and worse 
                on new data see figure [3]
"""

learn = cnn_learner(dls, resnet34, metrics=error_rate)

"""
cnn_learner() creates a convolutional neural network (CNN)
    note:   CNNs are the state-of-the-art approach for COMPUTER VISION MODELS
            CNNS are based on human vision

resnet34 specifies the ARCHITECTURE to use (ResNet) and 34 specifies the 
    NUMBER OF LAYERS (other options: 18, 50, 101, 152)

note:   picking an architecture is often not very important (different for 
        academics)

note:   some architectures like ResNet work most of the time 

note:   models using architectures with more layers   
            -take longer to train
            -more prone to overfitting

            +can be more accurate (with more data)

metrics=error_rate specifies the METRIC to use for EVALUATION
        
metric: measures the quality of the model's predictions USING the VALIDATION SET                                         

error_rate give a percentage of images in the VALIDATION SET classified 
    incorrectly (another metric: accuracy i.e. 1.0 - error_rate)

note:   loss    is a measure of performance the TRAINING SYSTEM can use to update 
                weights (might be a useful metric, needn't be)
        metric  is a measure of performance for HUMANS to evaluate the model

pretrained (parameter for cnn_lerner, default: True) SETS THE WEIGHTS of the 
    model to values that have already been trained on another dataset

note:   if possible use a pretrained model

parts of pretrained models handle for example   edge
                                                gradient
                                                color detection
    
cnn_learner will REMOVE THE LAST LAYER (head )when using pretrained models (the 
    last layer is usually task specific) and REPLACE IT with 1 or more layers 
    with RANDOM WEIGHTS (of appropriate size for the dataset)

head of a model:    last layer(s) of a model

advantage of using pretrained models:
    1) more accurate models
    2) quicker training
    3) less data needed)
    4) consequently: less money and train needed

transfer learning:  using models for different tasks (than their original purpose)
                    note: understudied
"""

learn.fine_tune(1)

"""
fine_tune(1) fine_tunes the model for a number of epochs (here: 1) in 2 steps:

    1) ONE EPOCH to fit the parts necessary to the new random head to work
    2) NUMBER OF EPOCHS requested (here: 1) to fit the entire model 
       note: later layers are UPDATED FASTER than earlier layers

note:   if the number of epochs is too low the model can be retrained later

note:   there is a difference between the function fine_tune() and fit()

        fit()   fits a model, i.e. exposes the model to images in the training 
                set and updates the parameters to improve predictions towards 
                the target labels each time
                note: fit() would refit a pretrained model (losing its capabilities)

        fine_tune() adapts a pretrained model to a new dataset
                    updates parameters by training for additional epochs on a 
                    different task than the pretrained task

Epoch:  complete passthrough the dataset
"""

################################################################################

""" deep inspection of deep learning models:

vast body of research on how to deeply inspect deep learning models

2013 Zeiler, Fergus "Visualizing and Understanding Convolutional Networks" [6]
    (used by the model that won 2012 ImageNet competition)
"""

################################################################################

"""Image recognizers for non-image tasks:

many things can be represented as images (e.g. sounds, text, time series, malware)

"""

################################################################################

"""other applications of image classification:


autonomous driving: important to localize objects in images

segmentation:   recognize the content of every individual pixel in an image

"""

#segmentation model with the CamVid dataset
#
#see Brostow et al.: "Semantic Object Classes in Video: A High-Definition Grount Truth Database" [7]
#
#note:  original code: label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}'
#       can't be pickled b/c of lambda
#
#note:  performance might be tested by letting the model color-cod each pixel of 
#       an image


from fastai.vision.all import *
from pathlib import Path

def _image(o):
    return path/'labels'/f'{o.stem}_P{o.suffix}'

path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = _image,
    codes = np.loadtxt(path/'codes.txt', dtype=str)
    )
learn = unet_learner(dls, resnet34)
learn.path = Path('D:/playground/Python/LLM')   #changing the model path
learn.fine_tune(8)

learn.show_results(max_n=6, figsize=(7,8))      #shows results of the model

learn.export('vision.pkl')


"""
note:   pickle doesn't pickle function objects. It expects to find the function 
        object by importing its module and looking up its name. lambdas are 
        anonymous functions (no name) so that doesn't work.

        solution: turn lambda into a method
"""

#sentiment classifier for movie reviews

from fastai.text.data import *
from fastai.text.learner import *
from fastai.text.models import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

learn.predict("I really liked that movie!")

"""
not:    if the ownload of a model is interrupted delete the folder and run the 
        cell again
        Alternatively use: untar_data(URLs.IMDB, force_download=True)
"""

#tabular dataloader for predicting income based on socioeconomic brackground 
#   with Adult dataset
#
#see Kohavi: "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid" [8]


from fastai.tabular.all import *
from pathlib import Path

path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(  path/'adult.csv', path=path, y_names="salary", 
                                    cat_names=['workclass', 'education', 
                                             'marital-status', 'occupation', 
                                             'relationship', 'race'], 
                                    cont_names=['age', 'fnlwgt', 'education-num'], 
                                    procs=[Categorify, FillMissing, Normalize])
learn = tabular_learner(dls, metrics=accuracy)
learn.path = Path('D:/playground/Python/LLM')   #changing the model path
learn.fit_one_cycle(3)
learn.show_results(max_n=6)
learn.export('tabular.pkl')

"""
cat_names=[]    specifies categorical characteristics with discret choices
cont_names=[]   specifies continuous characteristics that represent a quantity

note:   there is no pretrained model available for this task, therefore 
        fine_tune() is not available, instead 

        fit_one_cycle() is used (most commonly used for training fastai models 
            from scratch)
        
        in general: there aren't many widely available pretrained models for 
                    tabular modeling tasks
"""

#Collab dataloader with MovieLens dataset for recommendation system

from fastai.collab import *

path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5, 5.5))
learn.path = Path('D:/playground/Python/LLM')   #changing the model path
learn.fine_tune(10)
learn.show_results(max_n=6)
learn.export('collab.pkl')

"""
the model predicts the RATINGS of movies by users on a scale of 0.5-5.0 with 0.6 
    averrage error (note: the model predicts no category but a continuous value, 
    hence a range must be supplied, i.e. y_range)

y_range parameter specifies the RANGE OF THE TARGET VARIABLE (here: ratings)    
"""

"""datasets:

MOST USEFUL datasets are those that become academic baselines, i.e. datasets 
    that are widely studied by researchers and used to compare algorithmic changes
    e.g. MNIST, CIFAR-10, ImageNet

note:   when building models in practice EXPERIMENT and PROTOTYPE on SUBSETS OF 
        THE DATA. ONLY use the whole dataset when you have a good understanding
        of what the model has to do (good validation set!)
"""

################################################################################

""" Validation set and test sets

training process itself is fundamentally dumb

1) Step:    separate the data into  1) training set
                                    2) validation set (also development set)

note:   the aim is a model that makes good predictions based on LEARNED 
        CHARACTERISTICS of the data and NOT b/c of "MEMORIZATION" (overfitting)

IN PRACTICE models are rarely build by TRAINING the parameters ONCE, but 
            DIFFERENT VERSIONS of a model will be explored by varying choices 
            regarding:  network architecture, 
                        learning rate, 
                        data augmentation strategies etc.

hyperparameters:    parameters about parameters (i.e. the choices made about the 
                    parameters of the models, i.e. the aforementioned choices)

note:   the TRAINING PROCESS is looking at predictions based on the TRAINING DATA

        BUT the MODELER is evaluating the model by looking at predictions based 
        on the VALIDATION SET to decide whether hyperparameters should be changed

        problem:    subsequent models are shaped by the modeler having seen the 
                    validation set, i.e. they run the RISK of

                    OVERFITTING the validation data by human trial and error and 
                    exploration

        solution:   another layer of reserved data: TEST SET

                    the test set is only used once the modeling is complete

note:   test set and validation set should have enough data to get good 
        estimates of the models accuracy

note:   it is necessary to have a validation and a test set b/c models tend to 
        gravitate towards the simplest way to make good predictions: 
        "Memorization"

        (if only very little data is available it might suffice to have merely 
        a validation set)

note:   subcontracting modeling tasks always keep a test set the subcontractor 
        never gets to see and run the model against it with a metric of your 
        choosing based on what matters to you in practice and you decide on an 
        adequate level of performance
"""

################################################################################

"""defining test sets:

sometimes it might not suffice to RANDOMNLY SELECT a fraction of the data as 
    validation / test set

key property:   validation and test sets must be REPRESENTATIVE OF FUTURE DATA

for example cases see kaggle competitions

    1)  time series data:

        choosing randomn subsets will not be representative of most business 
        usecases, problem:  too easy to fill random gaps in continuous data
                            not indicative of what is actually required of the 
                            model 

        choose a continuous section with the latest days as validation set 
        (last 2 weeks / last month)

    2)  data where it is easy to anticipate ways the production data may be 
        qualitatively different from the training data
        
        e.g. distracted driver competition (or the fisheries competition)

        independent variable:   pictures of drivers behinde the wheel
        dependent variable:     categories such as  texting, 
                                                    eating, 
                                                    safely looking ahead
        
        e.g. an insurance company might be interested in predictions on drivers 
             the model has never seen

        problem: if the validation set contains only pictures of drivers already 
                 in the training set it will increase the models performance 
                 (making predictions easier)
                 alternatively: the model might be overfitting on characteristics 
                                of the people in the photo not learning the states

note:   it might not always be clear how the validation data will differ 
"""


################################################################################
                                # chapter 2 #
################################################################################

"""practice of deep learning

problems:   underestimating the constraints and overestimating the capabilities
            overestimating the constraints and underestimating the capabilities
            
            underestimating capabilities: not try beneficial things

            underestimaning constraints: might fail to consider and react to important issues
"""

"""starting:

most important: data availability

note:   the goal is not to get the perfect dataset / model, but to get started
        
        idea: iterate from end to end, to see the trickiest bits
        (i.e. complete every step as well as you can in a reasonable amount of time)
"""

################################################################################

""" state of deep learning: (2020)

computer vision:    object recognition  (recognizing items in images)
                    object detection    (recognizing where objects are in images, 
                                        highlight locations, name them)

                    not good at recognizing images with significantly different 
                    structure or style than the training data

note:   there is no general way to check which types of images are missing in the 
        training data (those images are called out-of-domain data)

                    labeling can be slow (problem for object detection systems)

note:   possible solution is synthetically generate vartiations of input images 
        (rotation, brightness, contrast), i.e. data augmentation

                    problems don't look like a vision problem, but can be 
                    transformed into one

Text:   classification documents based on categories (e.g. spam, sentiment, 
        author, SRC etc)

        generating context appropriate text (e.g. replies in social media, 
        imitating an author's style)        

        not good at generating correct responses

note:   text generating models will always be slightly ahead of models recognizing 
        automatically generated text

        natural language processing (NLP): translation, summarizing, finding 
                                           all mentions of a specific concept etc
        note: translations and summaries can contain completelty incorrect information

combining text & images:

        training:       combining images with output captions in English to 
        application:    automatically appropriate captions to new images

        note: application of these captions might be incorrect 

takeaway:   deep learning should not be used as an entirely automated system, it 
            is a tool for increasing the productivity of humans

tabular data:   analyzing time series data (used as part of an ensemble of 
                multiple types of models)

                note:   if already using random forests or 
                                         gradient boosting machines 
                        
                        deep learning may not result in dramatic improvements

                deep learning increases the variety of columns that can be used,
                e.g. columns containing natural language text (book titles, reviews), 
                high cardinality categorical colums (containing large number of 
                discrete choices),

                deep learning models take longer to train than random forests or 
                gradient boosting machines 

recommendation systems: just a special type of tabular data, 
                i.e. high cardinality variable representing users and 
                     another high cardinality variable representing products
                
                e.g. amazon represents purchases by customers as giant sparse 
                     matrix (rows: customers, columns: products)
                     
                     the recommendation systems applies some collaborative 
                     filtering to fill the matrix to recommend products

other data types:   domain-specific data types fit well into existing categories, 
                    e.g. protein chains are simmilar natural languages
                         sounds can be transformed into spectograms (images)
"""

################################################################################

"""drivetrain approach:

2012 Zwemer, Loukides "Desgning great data products" [9]

basic idea: 1)  consider the objective
            2)  what actions can you take to meet that objective
                what data do you have/ can you acquire that can help
            3)  build a model to determine the best actions to take for the 
                best result (regarding the objective), i.e. predictive modeling

note:   datas is used to produce actionable outcomes (not just more data)

drivetrain approach:    1)  defining a clear objective
                            (e.g. google: 
                            what's the main objective in typing a search query
                            -> show the most relevant search result)

                        2)  consider what leves you can pull tp better achieve 
                            the goal
                            (e.g. google: ranking search results)

                        3)  consider what new data is needed to produce the ranking

                    note: predictive modelling begins only AFTER the steps 1-3

                    predictive models: input    the levers and 
                                                any uncontrollable variable

                                        combine outputs to predict the final state

example:    recommendation system
            1) objective:   drive additional sales by surprising customers with 
                            recommendations of items they would not have 
                            purchased without the recommendations
            2) Lever:       ranking recommendations
            3) new data:    conduct randomized exzperiments to collect data 
                            about a wide range of recommendations (often missed)        
            
            predictive modelling: build 2 models
                            
                            1) for purchase probabilities without recommendation
                            2) for purchase probabilities with recommendation   
                            
                            difference between the the probabilities of 1) and 2)
                            is utility function for for a given recommendation  
                            to a customer

                            (low if a book is recommended a customer has already rejected or
                            a book is recommended a customer would have bought anyway)
"""

################################################################################

"""gathering data: example - bear detector

categories Ö grizzly, black and teddy bear

task:   create an image recognition model

collecting data:    bing Image search

"""
from utils import *
from fastai.vision.utils import download_images, verify_image, get_image_files

from pathlib import Path


from azure.identity import DefaultAzureCredential

key = os.environ['AZURE_SEARCH_KEY']
#results = search_images_bing(key, 'grizzly bear', 500)
#ims = results.attrgot('content_url')
#path = Path('D:/playground/python/LLM/media/bears/grizzly')
#len(results)

bear_types = 'grizzly', 'black', 'teddy'
path = Path('bears')

"""
download_url(results[0], path)
im = Image.open(path+'897b2e5d-6d4c-40fa-bbe8-6829455747e2.jpg')
im.to_thumb(128,128)
im.show()
download_images(results, path)
"""

i =0
if not path.exists():
    path.mkdir()
    for o in bear_types:
        i= i+1
        print(f"Downloading images for {o} ({i}/{len(bear_types)})")
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, term=f"{o} bear", total_count=400)
        print(len(results))
        download_images(dest, urls=results, n_workers=0)


#collect all images in a single list
fns = get_image_files(path)

#verify if any images are corrupted, verify_images() return a L object
failed = L()
for o in fns:
    if verify_image(o) == False:
        failed.append(o)
        #os.unlink(o)

failed

#to remove failed images use unlink
failed.map(Path.unlink)


"""
note:   models can reflect only the data used to train them, if the data is 
        biased so is the model, see Deb Raji 
        "Actionable Auditing: Investigating the impact of publicly naming biased 
        performance results of commercial ai products" [10]

        be careful to think about the data you expect to see in practice and 
        check to ensure all these types are reflected in the source (training, 
        validation, test) data

"""

################################################################################

""" From data to dataloaders:

DataLoaders class stores DataLoader objects for training, 
                                                validation and 
                                                test
                        
class DataLoaders(GetAttr):
    def __init__(self, *.loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train, valid = add_propps(lambda i, self: self[i])

requirements for turning data into a Dataloader object:

    1)  what kind of data we are working with
    2)  how to get the list of items
    3)  how to label these items
    4)  how to create a validation set

note:   there are factory methods for particular combination of these 
        requirements (e.g. ImageDataLoaders, TextDataLoaders, TabularDataLoaders)

if no factory method fits the combination of application and data structure use 
fastai's DATA BLOCK API, e.g. 
"""

from fastai.data.block import *         #DataBlock
from fastai.data.transforms import *    #RandomSplitter
from fastai.vision.data import *        #ImageBlock
from fastai.vision.augment import *     #Resize

bears = DataBlock(blocks = (ImageBlock, CategoryBlock),
                  get_items=get_image_files,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  get_y=parent_label,
                  item_tfms=Resize(128))

#Explanation

blocks = (ImageBlock, CategoryBlock)

"""
typle specifying the types for the  independent (ImageBlock) and 
                                    dependent (CategoryBlock) variables

note:   independent variable is used to make predictions from
        dependent variable is the target of that prediction

        e.g. predictions are made from the Images about the type of bear
"""

get_items=get_image_files

"""
note:   the underlying items for this DataLoaders are Filepaths (of the images)

the get_image_files function takes a path and returns a LIST of all images on 
this path (recursively)
"""

splitter = RandomSplitter(valid_pct=0.2, seed=42)

"""
the example uses a random splitter with a fixed seed, i.e. 42 to split the data 
into a training (80% of the data) and a validation (20%) set

note:   online available datasets often already have a predefined validation set
        by  1) putting training and validation data into DIFFERENT FOLDERS
            2) using a csv file listing every filename together with its subset
            ...
"""

get_y = parent_label

""" 
specifying function to call to create the label (set the dependent variable)

note:   independent variable often referred to as x
        dependent variable often referred to as y

parent_label returns the name of the parent folder a file is in
"""

item_tfms=Resize(128)

"""
since images are not fed one at a time but as mini batches within an array 
    (usually referred to as tensor) the images have to have the same size, hence 
    they are resized

note:   by default Resize() crops images to fit a square of the requested size
        (here: 128x128)
        problem:    might loose important details
        alternatively:  pad images with 0s
                        squish / strech
"""
#Squish
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

#probnlem:   shapes tend to become unrealistic

#padding
bears = bears.net(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

#problem:   tends to remove features that might be important

"""
in practice:    randomly select a part of the image and crop it in each epoch 
                anew so that the model can learn to focus on and recognize 
                different parts of the image (see different pictures of the same 
                thing irl)
                helps the network to understand the basic concept of what an 
                object is and how it can be represented in an image
"""

#RandimResizedCrop (data augmentation)

bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloader(path)
dls.valid.show_batch(max_n=4, nrows=1, unique=True)

"""
DataBlock objects are like TEMPLATES for creating DataLoaders, but the 
DataLoader still needs the path to the data, e.g.
"""

dls = bears.dataloader(path)

"""
returns a DATALOADERS OBJECT for the DATABLOCK bears specifying the PATH TO THE 
    DATA of the DataBlock

the number of items a dataloader returns in 1 cycle by default 64 (batch size)
"""

dls.valid.show_batch(max_n=4, nrows=1)

"""
show_batch() shows items from a batch max_n speccifies the number of items and 
    nrows specifies the number of rows to show 
"""

################################################################################

"""Data augmentation:

data augmentation:  creating random variations of the unput data

augmentation techniques:    rotation, flippping, perspective warping, 
                            brightness, contrast


note:   if all images are same size, augmentation can be applied to entire 
        batches of images using the GPU with batch_fms

example: 
"""

bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)


"""
mult=2 refers to double the amount of augmentations to be applied compared 
    to the default
"""

################################################################################

"""Training the bear classifier

crops images tpo 224x224 using RandomResizedCrop and applzing aug_transforms
"""

from fastai.vision.learner import *         #cnn_learner
from torchvision.models import *            #resnet18
from fastai.metrics import *                #error_rate
from fastai.callback.schedule import *      #fine_tune

bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), 
                  batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
learn.export('bears.pkl')

from fastai.vision.all import *
import os

path = untar_data(URLs.MNIST_SAMPLE, data="D:/playground/Python/LLM/media")
path.ls()
(path/'train').ls()

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
sevens

im3_path = threes[1]
im3 = Image.open(im3_path)
im3

array(im3)[4:10, 4:10]

im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15, 4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors), len(seven_tensors)

show_image(three_tensors[1])

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape

len(stacked_threes.shape)

mean3 = stacked_threes.mean(0)
show_image(mean3);

mean7 = stacked_sevens.mean(0)
show_image(mean7);

a_3 = stacked_threes[1]

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 -mean3)**2).mean().sqrt()
dist_3_abs, dist_3_sqr

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs, dist_7_sqr

import torch.nn.functional as F

F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt()

data = [[1,2,3], [4,5,6]]
arr = array(data)
tns = tensor(data)

arr, tns

tns[1]
tns[:,1]
tns[1,1:3]
tns+1
tns.type()
tns*1.5

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255

valid_3_tens.shape, valid_7_tens.shape

def mnist_distance(a,b):
    return (a-b).abs().mean((-1,-2))

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape

(valid_3_tens - mean3).shape

def is_3(x):
    return mnist_distance(x, mean3) < mnist_distance(x, mean7)

is_3(a_3), is_3(a_3).float()

is_3(valid_3_tens)

accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1-is_3(valid_7_tens).float()).mean()

accuracy_3s, accuracy_7s, (accuracy_3s+accuracy_7s)/2

from fastbook import *
from fastai.vision.all import *
from torch.nn import *

def f(x): return x**2

plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');

xt = tensor(3.).requires_grad_()

yt = f(xt)
yt

yt.backward()

xt.grad

xt = tensor([3.,4.,10.]).requires_grad_()
xt

def f(x): return (x**2).sum()
yt = f(xt)
yt

yt.backward()
xt.grad

w -= w.grad * lr



from fastbook import *
from fastai.vision.all import *

time = torch.arange(0, 20).float()
#time
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 +1
#plt.scatter(time, speed);

def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c

params = torch.randn(3).requires_grad_()

preds = f(time, params)

#show_preds(preds)

def mse(preds, targets):
    return ((preds-targets)**2).mean()

#loss = mse(preds, speed)
#loss

#loss.backward()
#params.grad

#params.grad * 1e-5

#params

lr = 1e-5
#params.data -= lr * params.grad.data
#params.grad = None


#preds = f(time, params)
#mse(preds, speed)
#show_preds(preds)

def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
#    if prn:
#        print(loss.item())
    return preds

def show_preds(preds, ax=None):
    if ax is None:
        ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300, 100)


for i in range(10):
    apply_step(params)

    _,axs = plt.subplots(1,4, figsize=(12,3))
    for ax in axs:
        show_preds(apply_step(params, False), ax)
    plt.tight_layout()



_,axs = plt.subplots(1,4, figsize=(12,3))
for ax in axs:
        show_preds(apply_step(params, False), ax)
plt.tight_layout()

from fastbook import *
from fastai.vision.all import *

path = Path('media/mnist_sample')
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
three_tensors = [tensor(Image.open(o)) for o in threes]
seven_tensors = [tensor(Image.open(o)) for o in sevens]
stacked_threes = torch.stack(three_tensors).float()/255
stacked_sevens = torch.stack(seven_tensors).float()/255

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) +[0]*len(sevens)).unsqueeze(1)
train_x.shape, train_y.shape

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28,1))
bias = init_params(1)
(train_x[0]*weights.T).sum() + bias

def linear1(xb):
    return xb@weights + bias

preds = linear1(train_x)
preds

corrects = (preds>0.0).float() == train_y
corrects

corrects.float().mean().item()

with torch.no_grad():
    weights[0] *= 1.0001
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

help(torch.where)

trgts = tensor([1,0,1])
prds = tensor([0.9, 0.4, 0.2])

torch.where(trgts==1, 1-prds, prds)

plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
