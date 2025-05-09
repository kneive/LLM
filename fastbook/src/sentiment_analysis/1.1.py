# sentiment analysis without specifying models

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

""" Output
Device set to use cpu
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'In this course, we will teach you how to find all the answers in one convenient and easy manner. In the beginning, the following series will teach'}, {'generated_text': 'In this course, we will teach you how to use your skills to make sense of the situation and understand how to avoid them and how to stop them'}]
"""


from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,                     #returns 2 sequences
)

print(res)

""" Output
Device set to use cpu
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'In this course, we will teach you how to use the techniques of writing, by using the concepts of a computer program and a book. We have'}, {'generated_text': 'In this course, we will teach you how to write a new feature and to work with it in the future. From the point of view of the'}]
"""


from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(res)

""" Output
No model was supplied, defaulted to facebook/bart-large-mnli and revision d7645e1 (https://huggingface.co/facebook/bart-large-mnli).
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use cpu
{'sequence': 'This is a course about Python list comprehension', 'labels': ['education', 'business', 'politics'], 'scores': [0.9622024893760681, 0.02684144675731659, 0.010956036858260632]}
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

""" Output
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use cpu
[{'label': 'POSITIVE', 'score': 0.9598050713539124}]
"""
