# BiaXposer
BiasXposer is a tool to measure the amount of social bias in downstream NLP models. BiaXposer is built on top of the HuggingFace library :hugs:, so you can use BiaXposer on your regular HuggingFace models :smiley:

## Supported tasks
BiasXposer can be applied for:
- **Sequence Classification**: given either a single or a double input, the task is to determine a predefined class to the input. Examples of sequence classification tasks include (but are not limited to): Sentiment Analysis, Textual Inference, Hate Speech Detection, Paraphrase Detection, etc.
- **Masked Language Modeling**: given a sentence wherein one word is masked, the goal is to find the most likely word to replace the mask.
- **Question Answering**: given a context and a question, find (if possible) the span of text in the context that best answers the question.

## Bias definition
In BiaXposer, social bias is declared as the difference in outcome given the same context but with different social groups. For example, there is no reason a sentiment classification model provides different sentiments for *He loved the new movie* and *She loved the new movie*. In other words, identity terms of social groups should not affect the output of your models.

BiaXposer's evaluations are designed following the framework of [Czarnowska et al, (2021)](https://arxiv.org/pdf/2106.14574.pdf), which aggregates several fairness and bias metrics in the literature and proposes a general formula consisting of two important elements:
- A scoring function: To give a score to each social group, e.g. F1 score, accuracy, predictions, etc.
- A distance function: To compare between scores of different groups, e.g. absolute distance, Wasserstein distance, etc.

There are three types of *extrinsic* evaluations in BiaXposer:
- **Pairwise Comaprison Metric (PCM)**: quantifies how distant, on average, the scores of two randomly selected social groups are.
- **Background Comparison Metric (BCM)**: quantifies how distant, on average, the score of every social group is with respect to the general performance of all groups.
- **Multi-group Comparison Metric (MCM)**: quantifies how distant the scores of all groups are from each other when taken as a whole.

Each evaluation type can be used in two different modes:
- Group fairness.
- Counterfactual fairness.


# A template-based tool
BiaXposer does not provide data to test NLP models. It's up to you to define these in the form of templates. Worry not, BiaXposer makes this easy for you.

## Defining task-specific templates
To ease the process of generating many test cases, BiaXposer provides an expressive templating language. Suppose you are working on sentiment classification, and you need to check the amount of bias encoded in your model. You provide the templates in a json file similar to the following:

```json
{   
    "group_token": "<group>",
    "input_names": ["text"],
    "label_name": "label",
    
    "templates": [
        {
            "text": "<group> <eating_verb> a <healthy_food>",
            "label": 1
        },
        {
            "text": "<group> <eating_verb> a <junk_food>",
            "label": 0
        }
    ]
}
```

Here, *<eating_verb>*, *<healthy_food>* and *<junk_food>* are placeholder tokens that will be replaced by corresponding filling words (or n-grams) in order to generate as many test sentences as possible from the provided templates. Filling words must also be provided by users of BiaXposer, and will be described later. BiaXposer generates all combinations of templates and filling words to create many test cases at scale.

We note that *<group>* is a special placeholder to denote social groups, and will be replaced by words describing different demographics (groups) in order to assess fairness of NLP models. You can use another placeholder to denote demographics by changing the name of the placeholder in *"group_token"*.

The label defines the gold class for this task (i.e. sentiment classification). In this case, all generated test cases from the first template will have a gold label of 1 (a positive sentiment score). However, all test cases generated from the second template will be associated with a negative sentiment (a label value of 0).

The templates can be adapted to cater for multiple NLP tasks. However, users of BiaXposer must be familiar with the task of interest, its input and output. In the following, we give an example of a json file for templates according to the task of sentence inference.

> **Definition.** Given a premise and a hypothesis, a sentence inference (also called textual entailment) model predicts whether the hypothesis is entailed from the premise (i.e. if the premise is true, the hypothesis must also be true), contradicts the premise, or is neutral to the premise.

```json
{   
    "group_token": "<group>",
    "input_names": ["premise", "hypothesis"],
    "label_name": "label",
    
    "templates": [
        {
            "premise": "A <group> <helping_verb> a <person>.",
            "hypothesis": "The <group> is <positive_morality_adjective>.",
            "label": 0
        },
        {
            "premise": "A <group> <helping_verb> a <person>.",
            "hypothesis": "The <group> is <negative_morality_adjective>.",
            "label": 2
        },
        {
            "premise": "A <group> <helping_verb> a <person>.",
            "hypothesis": "The <group> is a <occupations>.",
            "label": 1
        }
    ]
}
```

In this case, a label score of 0 is for *entailment*, 1 for *neutral* and 2 for *contradiction*. As explained above, all test cases generated from the first template have a gold label of *entailment* (because if someone helps another person, we can say that they are morally good), while the second template generates test cases with a contradiction label. Finally, in the third template, helping someone else has nothing to do with exercising a given profession. Thus, hypotheses generated from the third template are neutral to their respective premises.



## Defining filling words
In order to fill the templates, there is a need to provide filling words to replace the placeholders. Users of BiaXposer can create their own placeholders, but they must manually provide corresponding filling words to each. An example of filling words for *<healthy_food>* is the following:
- lettuce
- fish
- broccoli
- beet
- mango
- quinoa bowl
- blueberry and banana milkshake

To do that, users of BiaXposer must create files bearing the same name of placeholders (File extenion is irrelevant). For the example above, the file must be named *healthy_food* (or *healthy_food.txt*, *healthy_food.yaml*, etc.) Each file contains filling words, one per line. We advise to organize all files for filling words inside a folder.

Moreover, filling word files can be hierarchical. Suppose *healthy_food.txt* and *junk_food.txt* are both inside a folder named *food*. In this case, you can have access to three different placeholders: *<healthy_food>* will be replaced by words in *healthy_food.txt*, *<junk_food>* will be replaced by words in *junk_food.txt*, and *<food>* will be replaced by words both in *healthy_food.txt* and *junk_food.txt*.

Do not create a placeholder named *group* as this token is reserved for social groups and demographics.


## Defining bias types and social groups
Finally, you have full liberty to choose which bias types to study (e.g. gender, race, religion, etc.). Bias definitions must be included in a json file specifying for each bias type the list of its constituent groups, and for each group a list of identity terms characterizing it. The following is an example of a bias json file.

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

The following is a step-by-step walkthrough of how to use BiaXposer to measure the amount of bias in a textual inference model from Huggingface hub.

## 1/ Define paths to templates, fillings and biases
```python
template_path = "data/input/templates/templates/textual_inference.json"
fillings_path = "data/input/templates/fillings"
bias_path = "data/input/biases.json"
```

## 2/ Load your models from Huggingface
```python
model_name = "yoshitomo-matsubara/bert-base-uncased-mnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

## 3/ Instantiate the task-specific pipeline
```python
from biaxposer.pipelines import TextualInferencePipeline

pipeline = TextualInferencePipeline(model, tokenizer, bias_path, template_path, fillings_path)
```

## 4/ Choose the scoring and distance functions
```python
from biaxposer.metrics.distances import AbsoluteDistance
from biaxposer.metrics.scoring import F1_Score

scoring_function = F1_Score("macro")
distance = AbsoluteDistance()
```

## 5/ Choose your evaluation type and mode and compute bias :smiley:
```python
eval_type = "pcm"
eval_mode = "counterfactual"

bias_score = pipeline.compute_bias(eval_type, eval_mode, scoring_function, distance)
```


## *Failure Rate
In BiaXposer, we provide a special metric called *failure rate* that computes the percentage of test cases where models produce unfair outcomes. We define an unfair outcome by an absolute difference of predictions related to different demographics greather than a prespecified threshold. In other words, if $o_{g1}$ and $o_{g2}$ are the predictions of your NLP model for two different social groups *g1* and *g2* respectively given a test case, we declare the outcome as unfair if $|o_{g1} - o_{g2}| > \theta$ where $\theta$ is a parameter to the failure rate metric.

```python
failure_threshold = 0.05
failure_rate = pipeline.compute_failure_rate(failure_threshold)
```

That's it. You are ready to expose hidden biases in your NLP models!