# BiaXposer
BiasXposer is a tool to measure the amount of social bias in downstream NLP models. BiaXposer is built on the HuggingFace library :hugs:, so you can use BiaXposer on your regular HuggingFace models :smiley:

## Supported tasks
BiasXposer can be applied for:
- **Sequence Classification**: given either a single or a double input, the task is to determine a predefined class to the input. Example of sequence classification tasks include (but are not limited to): Sentiment Analysis, Textual Inference, Hate Speech Detection, Paraphrase Detection...
- **Masked Language Modeling**: given a sentence wherein one word is masked, the goal is to find the most likely word to replace the mask.
- **Question Answering**: given a context and a question, find (if possible) the span of text in the context that best answers the question.

## Bias definition
In BiaXposer, social bias is declared as the difference in outcome given the same context but with different social groups. For example, there is no reason a sentiment classification model would provide different sentiments for *He loved the new movies* and *She loved the new movie*. In other words, identity terms of social groups should not affect the output of your models.

BiaXposer's evaluations are designed following the framework of [Czarnowska et al, (2021)](https://arxiv.org/pdf/2106.14574.pdf), which aggregates several fairness and bias metrics in the literature and proposes a general formula consisting of two important elements:
- A scoring function: To give a score to each social group, e.g. F1 score, accuracy, predictions...
- A distance function: To compare between scores of different groups, e.g. absolute distance, Wasserstein distance...

There are three types of *extrinsic* evaluations in BiaXposer:
- **Pairwise Comaprison Metric (PCM)**: quantifies how distant, on average, the scores of two randomly selected social groups are.
- **Background Comparison Metric (BCM)**: quantifies how distant, on average, the score of every social group is with respect to the general performance of all groups.
- **Mult-group Comparison Metric (MCM)**: quantifies how distant the scores of all groups are from each other when taken as a whole.

Each evaluation type can be used in two different modes:
- Group fairness.
- Counterfactual fairness.


# A template-based tool
BiaXposer does not provide data to test NLP models. It's up to you to define these in the form of templates. Worry not, BiaXposer makes this easy for you.

## Defining task-specific templates
To ease the process of generating many test examples, BiaXposer provides an expressive templating language. Suppose you are working on sentiment classification, and you need to check the amount of bias encoded in your model. You may give the following template:

```json
{
    "text": "<group> <eating_verb> a <food>",
    "label": 1
}
```

Here, *<group>* represents a social group and will replaced by words describing social groups to form a new sentence each time. *<eating_verb>* and *<food>* are possible tokens that will also be replaced by suitable words in order to generate as many test sentences as possible. Of course, you can create your own tokens.

The label defines the gold class. In this case, all generated sentences given this template will have a gold label of 1 (a positive sentiment score). If you are interested in a finer-grained analysis, e.g. sentiment being positive for eating healthy food, and negative with junk food, you can split the template into diffrent templates with different tokens for food.

```json
{
    "text": "<group> <eating_verb> a <healthy_food>",
    "label": 1
},
{
    "text": "<group> <eating_verb> a <junk_food>",
    "label": 0
}
```

## Defining filling words
Of course, BiaXposer needs to know about which words to fill which token. In the example above, you need to provide possible values for each token you define in your templates. An example filling words for *<healthy_food>* is the following:
- lettuce
- fish
- broccoli
- beet
- mango
- quinoa bowl
- blueberry and banana milkshake

## Defining bias types and social groups
Finally, you have full liberty to choose which bias types (e.g. gender, race, religion...) you are interested in. You need to define a bias type by a list of its constituent social groups, and each group by a list of some definition words.

```json
{
    "gender":{
        "male": ["man", "boy", "father"],
        "female": ["woman", "girl", "mother"]
    },

    "race": {
        "white": ["white", "caucasian"],
        "black": ["black", "dark-skinned"],
        "asian": ["asian"]
    },

    "religion":{
        "muslim": ["muslim"],
        "christian": ["christian"],
        "jew": ["jew"]
    }
}
```




# Usage
BiaXposer is extensible and easy to use. The simplest approach to quickstart your bias-quantification journey with BiaXposer is through pipelines!

The following is a step-by-step walkthrough how to use BiaXposer to measure the amount of bias in a textual inference model from Huggingface hub.

## 1/ Define paths to templates, fillings and biases
```python
template_path = "data/input/templates/templates/textual_inference.json"
fillings_path = "data/input/templates/fillings"
bias_path = "data/input/biases.json"
```

## 2/ Load your models from Huggingface
```python
model_name = "yoshitomo-matsubara/bert-base-uncased-mnli"
AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

## 3/ Instantiate the task-specific pipeline
```python
from biaxposer.pipelines import TextualInferencePipeline

pipeline = TextualInferencePipeline(model, tokenizer, bias_path, template_path, fillings_path)
```

## Choose the scoring and distance functions
```python
from biaxposer.metrics.distances import AbsoluteDistance
from biaxposer.metrics.scoring import F1_Score

scoring_function = F1_Score("macro")
distance = AbsoluteDistance()
```

## Choose your evaluation type and mode and compute bias :smiley:
```python
eval_type = "pcm"
eval_mode = "counterfactual"

bias_score = pipeline.compute_bias(eval_type, eval_mode, scoring_function, distance)
```

That's it. You are ready to expose hidden biases in your NLP models!