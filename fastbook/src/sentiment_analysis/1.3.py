# sentiment analysis with pytorch

# using pipeline

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = ["I've been waiting for a HuggingFace course my whole life.",
           "Python is great!"]

res = classifier(X_train)
print(res)

""" Output
[{'label': 'POSITIVE', 'score': 0.9598050713539124}, 
 {'label': 'POSITIVE', 'score': 0.9998615980148315}]
"""

# applying the pipeline separately

batch = tokenizer(X_train, 
                  padding=True, 
                  truncation=True, 
                  max_length=512, 
                  return_tensors="pt")

print(batch)

""" Output
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
                        2607,  2026,  2878,  2166,  1012,   102],
                      [  101, 18750,  2003,  2307,   999,   102,     0,     0,     0,     0,
                           0,     0,     0,     0,     0,     0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
"""

with torch.no_grad():                               
    outputs = model(**batch)    #calls the model and unpacks the batch (dictionary)
    print(outputs)

    """ Output
    SequenceClassifierOutput(loss=None, 
                         logits=tensor([[-1.5607,  1.6123],
                                        [-4.2745,  4.6111]]), 
                         hidden_states=None, 
                         attentions=None)
    """

    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    
    """ Output
    tensor([[4.0195e-02, 9.5981e-01],
            [1.3835e-04, 9.9986e-01]]) 
    """
    
    labels = torch.argmax(predictions, dim=1)
    print(labels)

    """ Output
    tensor([1, 1])
    """