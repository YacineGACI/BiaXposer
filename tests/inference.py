from transformers import AutoTokenizer, AutoModelForSequenceClassification
from biaxposer.metrics.distances import AbsoluteDistance, WassersteinDistance, AbsoluteDivergenceFromExpectedOutcome, AbsoluteDivergenceFromExpectedOutcome_PerGroup
from biaxposer.metrics.scoring import F1_Score, Accuracy_Score, Precision_Score, Recall_Score, ClassPrediction_Score, AverageClassPrediction_Score, ScoringFunction
from biaxposer.pipelines import SentimentClassificationPipeline, TextualInferencePipeline





# template_path = "data/input/templates/templates/sentiment_classification.json"
template_path = "data/input/templates/templates/textual_inference.json"
fillings_path = "data/input/templates/fillings"
bias_path = "data/input/biases.json"
# model_name = "siebert/sentiment-roberta-large-english"
model_name = "yoshitomo-matsubara/bert-base-uncased-mnli"
# model_name = "ajrae/bert-base-uncased-finetuned-mrpc"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


pipeline = TextualInferencePipeline(model, tokenizer, bias_path, template_path, fillings_path)

scoring_function = F1_Score("macro")
# scoring_function = Recall_Score("macro")
# scoring_function = ClassPrediction_Score(1)
scoring_function = AverageClassPrediction_Score(1)

distance = AbsoluteDistance()
# distance = WassersteinDistance()
distance = AbsoluteDivergenceFromExpectedOutcome()
distance = AbsoluteDivergenceFromExpectedOutcome_PerGroup()




# print()
# print("pcm group", pipeline.compute_bias("pcm", "group", scoring_function, distance))
# print("pcm counterfactual", pipeline.compute_bias("pcm", "counterfactual", scoring_function, distance))
# print()
# print("bcm group", pipeline.compute_bias("bcm", "group", scoring_function, distance))
# print("bcm counterfactual", pipeline.compute_bias("bcm", "counterfactual", scoring_function, distance))
# print()
# print("vbcm group", pipeline.compute_bias("vbcm", "group", scoring_function, distance))
# print("vbcm counterfactual", pipeline.compute_bias("vbcm", "counterfactual", scoring_function, distance))


print("vmcm group", pipeline.compute_bias("vmcm", "group", scoring_function, distance))
print("vmcm counterfactual", pipeline.compute_bias("vmcm", "counterfactual", scoring_function, distance))

