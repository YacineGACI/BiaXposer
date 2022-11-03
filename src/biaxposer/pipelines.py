from biaxposer.templates.template_processor import TemplateProcessor
from biaxposer.biases import read_biases
from biaxposer.metrics.metrics import PairwiseComparisonMetric, BackgroundComparisonMetric, MultigroupComparisonMetric, VectorBackgroundComparisonMetric, VectorMultigroupComparisonMetric
from biaxposer.inputs import SingleInputProcessor, DoubleInputProcessor
from biaxposer.outputs import PredictionOutputProcessor, MaskedLanguageModelingOutputProcessor, QuestionAsnweringOutputProcessor
from biaxposer.tasks import SequenceClassificationTask, LanguageModelingTask, QuestionAnsweringTask


class TaskSpecificPipeline:
    def __init__(self, model, tokenizer, bias_path, template_path, fillings_path, task, input_processor, output_processor):
        self.generic_init(model, tokenizer, bias_path, template_path, fillings_path)
        self.task = task
        self.input_processor = input_processor
        self.output_processor = output_processor
        

    
    def generic_init(self, model, tokenizer, bias_path, template_path, fillings_path):
        self.model = model
        self.tokenizer = tokenizer
        self.bias_path = bias_path
        self.template_path = template_path
        self.fillings_path = fillings_path
        self.template_processor = None
        self.templates = None
        self.bias_types = None
        self.task_scores = None
        self.metric = None
        self.all_eval_types = ["pcm", "bcm", "mcm", "vbcm", "vmcm"]
        self.all_eval_modes = ["group", "counterfactual"]

        self.read_bias_types()
        self.generate_templates()
        print("{} Test cases".format(len(self.templates)))

    

    
    def generate_templates(self):
        self.template_processor = TemplateProcessor(self.template_path, self.fillings_path)
        self.templates = self.template_processor.generations

    
    def read_bias_types(self):
        self.bias_types = read_biases(self.bias_path)

    
    def run_task(self):
        self.task_scores = self.task.run()


    
    def compute_bias(self, type, mode, scoring_fct, distance_fct):
        if mode not in self.all_eval_modes:
            raise ValueError

        if self.task_scores is None:
            self.run_task()

        if type == "pcm":
            self.metric = PairwiseComparisonMetric()
        elif type == "bcm":
            self.metric = BackgroundComparisonMetric()
        elif type == "mcm":
            self.metric = MultigroupComparisonMetric()
        elif type == "vbcm":
            self.metric = VectorBackgroundComparisonMetric()
        elif type == "vmcm":
            self.metric = VectorMultigroupComparisonMetric()
        else:
            raise ValueError

        if mode == "group":
            return self.metric.bias_group(self.task_scores, scoring_fct, distance_fct)
        elif mode == "counterfactual":
            return self.metric.bias_counterfactual(self.task_scores, scoring_fct, distance_fct)
        else:
            raise ValueError

    
    def compute_failure_rate(self, threshold):
        if self.task_scores is None:
            self.run_task()
        metric = PairwiseComparisonMetric()
        return metric.compute_failure_rate(self.task_scores, threshold)
        


    



class SentimentClassificationPipeline(TaskSpecificPipeline):
    def __init__(self, model, tokenizer, bias_path, template_path, fillings_path):
        self.generic_init(model, tokenizer, bias_path, template_path, fillings_path)
        self.input_processor = SingleInputProcessor(self.tokenizer, self.template_processor.input_names)
        self.output_processor = PredictionOutputProcessor()
        self.task = SequenceClassificationTask(self.model, self.bias_types, self.templates, self.template_processor.group_token, self.template_processor.label_name, self.input_processor, self.output_processor)







class TextualInferencePipeline(TaskSpecificPipeline):
    def __init__(self, model, tokenizer, bias_path, template_path, fillings_path):
        self.generic_init(model, tokenizer, bias_path, template_path, fillings_path)
        self.input_processor = DoubleInputProcessor(self.tokenizer, self.template_processor.input_names)
        self.output_processor = PredictionOutputProcessor()
        self.task = SequenceClassificationTask(self.model, self.bias_types, self.templates, self.template_processor.group_token, self.template_processor.label_name, self.input_processor, self.output_processor)




class LanguageModelingPipeline(TaskSpecificPipeline):
    def __init__(self, model, tokenizer, bias_path, template_path, fillings_path, topk=5):
        self.generic_init(model, tokenizer, bias_path, template_path, fillings_path)
        self.input_processor = SingleInputProcessor(self.tokenizer, self.template_processor.input_names)
        self.output_processor = MaskedLanguageModelingOutputProcessor()
        self.task = LanguageModelingTask(self.model, self.bias_types, self.templates, self.template_processor.group_token, self.template_processor.label_name, self.input_processor, self.output_processor, topk=topk)
        


class QuestionAnsweringPipeline(TaskSpecificPipeline):
    def __init__(self, model, tokenizer, bias_path, template_path, fillings_path):
        self.generic_init(model, tokenizer, bias_path, template_path, fillings_path)
        self.input_processor = DoubleInputProcessor(tokenizer, self.template_processor.input_names)
        self.output_processor = QuestionAsnweringOutputProcessor(tokenizer)
        self.task = QuestionAnsweringTask(model, self.bias_types, self.templates, self.template_processor.group_token, self.template_processor.label_name, self.input_processor, self.output_processor)
        