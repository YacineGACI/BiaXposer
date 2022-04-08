from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.pipelines import LanguageModelingPipeline
from src.metrics.scoring import AverageLikelihood_Score, AverageLikelihood_Score
from src.metrics.distances import AbsoluteDistance, AbsoluteDivergenceFromExpectedOutcome


# Define paths for templates
template_path = "data/input/templates/templates/language_modeling.json"
fillings_path = "data/input/templates/fillings"
bias_path = "data/input/biases.json"

# Name of the model
model_name = "bert-base-uncased"

# Instantiate the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


# Instantiate the pipeline
pipeline = LanguageModelingPipeline(model, tokenizer, bias_path, template_path, fillings_path)


# Define the scoring and distance functions
scoring_function = AverageLikelihood_Score()
distance = AbsoluteDistance()


# Print different bias scores
print()
print("pcm group", pipeline.compute_bias("pcm", "group", scoring_function, distance))
print("pcm counterfactual", pipeline.compute_bias("pcm", "counterfactual", scoring_function, distance))
print()
print("bcm group", pipeline.compute_bias("bcm", "group", scoring_function, distance))
print("bcm counterfactual", pipeline.compute_bias("bcm", "counterfactual", scoring_function, distance))
print()
print("vbcm group", pipeline.compute_bias("vbcm", "group", scoring_function, distance))
print("vbcm counterfactual", pipeline.compute_bias("vbcm", "counterfactual", scoring_function, distance))



# Print multigroup bias scores
scoring_function = AverageLikelihood_Score()
distance = AbsoluteDivergenceFromExpectedOutcome()

print("mcm group", pipeline.compute_bias("mcm", "group", scoring_function, distance))
print("mcm counterfactual", pipeline.compute_bias("mcm", "counterfactual", scoring_function, distance))
print()
print("vmcm group", pipeline.compute_bias("vmcm", "group", scoring_function, distance))
print("vmcm counterfactual", pipeline.compute_bias("vmcm", "counterfactual", scoring_function, distance))
