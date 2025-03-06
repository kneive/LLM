"""default model"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

"""workflow: 

1) preprocessing:   here applying a tokenizer
2) Application:     applying the model
3) postprocessing:  presenting the result 
                    [{'label': 'POSITIVE', 'score': 0.9598049521446228}]
"""

"""specific model"""

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,                     #returns 2 sequences
)

print(res)

"""Output:

[{'generated_text': 'In this course, we will teach you how to create, share and 
edit applications in more powerful way. Your course is built on the principle of 
\u200fself-improvement\u200f and it allows you to take advantage of the new 
technology in terms of advanced software.
\n
\n
\n
\nThis post may contain affiliate links.\n
Follow Our Content Management blog for more information and other affiliate links.\n
If you would like to learn more about this course or other resources, click here.\n
Learn more about the Advanced Advanced Software in the Advanced Software 
section in this section.'}, {'generated_text': "In this course, we will teach 
you how to build a scalable, secure website.
\n
\n
\n
\n
If you read this course you will see this article in one of our YouTube video 
tutorials:\n
For full access to the course check out our blog.\n
How to use Chrome\n
It will take a few more clicks to build a truly secure website. However, you 
will need some knowledge that's important to you. And you're the first to use 
Chrome in your new browser. How do you build one of JavaScript's most popular 
JavaScript features?\n
If you have already set up a website you can use for your website on your own 
and have more access to the website in your browser. The easiest way to get 
access to it is the following:\n
Let's take a look at some of the components used in a simple browser like 
browser. We can take a look at some of our components from our tutorial:\n
The Browser component\n
Each component contains a JavaScript-related components.\n
The browser component itself is composed of all our JavaScript-related 
components (not just JS that uses our jQuery library, it's also a library you 
can add to your website):\n
Each component contains a JavaScript-related component that provides JavaScript-
related function() {\n
};\n
Here we have a simple JavaScript- related component that provides JavaScript-
related function() {\n
We can use jQuery, but we have to do a different thing:\n
As mentioned before, in the first step we will use jQuery to"}]
"""


"""zero-shot classification"""

from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(res)

"""
here a text is supplied without knowing the corresponding label and supply 
candidate labels

Output: (formatted)

{'sequence': 'This is a course about Python list comprehension', 
'labels': ['education',         'business',             'politics'], 
'scores': [0.9622026085853577,  0.026841387152671814,   0.01095600426197052]}

"""

"""available tasks for pipeline:


    "audio-classification": will return a AudioClassificationPipeline.
    "automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
    "depth-estimation": will return a DepthEstimationPipeline.
    "document-question-answering": will return a DocumentQuestionAnsweringPipeline.
    "feature-extraction": will return a FeatureExtractionPipeline.
    "fill-mask": will return a FillMaskPipeline:.
    "image-classification": will return a ImageClassificationPipeline.
    "image-feature-extraction": will return an ImageFeatureExtractionPipeline.
    "image-segmentation": will return a ImageSegmentationPipeline.
    "image-to-image": will return a ImageToImagePipeline.
    "image-to-text": will return a ImageToTextPipeline.
    "mask-generation": will return a MaskGenerationPipeline.
    "object-detection": will return a ObjectDetectionPipeline.
    "question-answering": will return a QuestionAnsweringPipeline.
    "summarization": will return a SummarizationPipeline.
    "table-question-answering": will return a TableQuestionAnsweringPipeline.
    "text2text-generation": will return a Text2TextGenerationPipeline.
    "text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
    "text-generation": will return a TextGenerationPipeline:.
    "text-to-audio" (alias "text-to-speech" available): will return a TextToAudioPipeline:.
    "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.
    "translation": will return a TranslationPipeline.
    "translation_xx_to_yy": will return a TranslationPipeline.
    "video-classification": will return a VideoClassificationPipeline.
    "visual-question-answering": will return a VisualQuestionAnsweringPipeline.
    "zero-shot-classification": will return a ZeroShotClassificationPipeline.
    "zero-shot-image-classification": will return a ZeroShotImageClassificationPipeline.
    "zero-shot-audio-classification": will return a ZeroShotAudioClassificationPipeline.
    "zero-shot-object-detection": will return a ZeroShotObjectDetectionPipeline.

"""
################################################################################

"""Tokenizer

'puts text into mathematical representation the model undersrtands'


"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

#default model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print(res)                                          #prints rresult

tokens = tokenizer.tokenize(sequence)
print(tokens)                                       
ids = tokenizer.convert_tokens_to_ids(tokens)       #turns tokens into ids
print(ids)                                          
decoded_string = tokenizer.decode(ids)              #returns tokens from ids
print(decoded_string)

"""Output: (formatted)

>>> res = tokenizer(sequence)

[{'label': 'POSITIVE', 'score': 0.9598049521446228}]

>>> sequence = "Using a Transformer network is simple"

{   'input_ids':        [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102], 
    'attention_mask':   [1, 1, 1, 1, 1, 1, 1, 1, 1]
}

>>> tokens = tokenizer.tokenize(sequence)

['using', 'a', 'transform', '##er', 'network', 'is', 'simple']

>>> ids = tokenizer.convert_tokens_to_ids(tokens)

[2478, 1037, 10938, 2121, 2897, 2003, 3722]

>>> decoded_string = tokenizer.decode(ids)

using a transformer network is simple
"""

"""
attention mask: List of 0s and 1s, 0 means the attention layer should ignore the token 
Ids:    decoded(ids) loose capitalization
        input ids:          [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102]
        ids from tokens:    [2478, 1037, 10938, 2121, 2897, 2003, 3722]

        note: 101 and 102 are beginning and end of sequence tokens
"""

################################################################################

"""Pytorch"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#apply the pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#data: applies 2 sentences as list
X_train = ["I've been waiting for a HuggingFace course my whole life.",
           "Python is great!"]

res = classifier(X_train)
print(res)

#apply the pipeline sepperately
#1) calls tokenizer with X_train data and sets  padding, 
#                                               truncation, 
#                                               max_length 
#                                               return format

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

#2)inference in pytorch

with torch.no_grad():                               
    outputs = model(**batch)                        #calls the model and unpacks the batch (dictionary)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)  #returns the predictions
    print(predictions)
    labels = torch.argmax(predictions, dim=1)       #returns the labels
    print(labels)


"""Output: (formatted)

>>> res = classifier(X_train)

[   {'label': 'POSITIVE', 'score': 0.9598049521446228}, 
    {'label': 'POSITIVE', 'score': 0.9998615980148315}]

>>> batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

{   'input_ids':    tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101, 18750,  2003,  2307,   999,   102,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]]), 

    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
        [-4.2745,  4.6111]]), hidden_states=None, attentions=None)


Predictions:        tensor([[4.0195e-02, 9.5980e-01],
                            [1.3835e-04, 9.9986e-01]])

Labels:             tensor([1, 1])
"""

################################################################################

"""save tokenizer and model"""

#save

save_directory = "saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

#load

tok = AutoTokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassification.from_pretrained(save_directory)

################################################################################

"""fine-tuning"""

#steps
#1. prepare datsset
#2. load pretrained Tokenizer, call with dataset -> encoding
#3. build PyTorch Dataset with encodings
#4. Load pretrained Model
#5. a) Load Trainer and train int
#   b) native PyTorch training loop

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
                    model,
                    training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )

trainer.train()

################################################################################
###########################to be continued######################################
################################################################################

"""fastai deep learning"""

from fastai.vision.all import *
import gradio as gr

#set PosixPath to Windows Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
#later set: pathlib.PosixPath=temp

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')       #problem with Posix Path on Windows (fix above)

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

im = PILImage.create('gadse3.jpg')
im.thumbnail((192, 192))

classify_image(im)

is_cat,_,probs = learn.predict(im)
print(f"is this a cat? {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
print(f"Probability it's a dog: {probs[0].item():.6f}")

#to be investigated
"""gradio for web interface

image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['dog.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False, share=True)

pathlib.PosixPath=temp
"""

################################################################################