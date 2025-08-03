#sentiment analysis specifying a model and using a tokenizer

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

""" Output
[{'label': 'POSITIVE', 'score': 0.9598050713539124}]
"""

sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print(res)

""" Output
{'input_ids': [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

tokens = tokenizer.tokenize(sequence)
print(tokens)

""" Output
['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
"""

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids) 

""" Output
[2478, 1037, 10938, 2121, 2897, 2003, 3722]
"""

decoded_string = tokenizer.decode(ids)
print(decoded_string)

""" Output
using a transformer network is simple
"""