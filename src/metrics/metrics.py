from audioop import bias
import itertools

class BiasMetric:
    def __init__(self):
        pass


    def bias(self, task_output, scoring_fct, distance_fct, mode="group"):
        if mode == "group":
            return self.bias_group(task_output, scoring_fct, distance_fct)
        elif mode == "counterfactual":
            return self.bias_counterfactual(task_output, scoring_fct, distance_fct)
        else:
            raise ValueError


    def bias_group(self, task_output, scoring_fct, distance_fct):
        raise NotImplementedError
    

    def bias_counterfactual(self, task_output, scoring_fct, distance_fct):
        raise NotImplementedError


    def process_task_output_to_group(self, task_output):
        """
        Transforms @task_output into a dict
        bias_type -> group -> scores 
        """
        result = {}
        for o in task_output:
            if o.bias_type not in result.keys():
                result[o.bias_type] = {}
            
            if o.group not in result[o.bias_type].keys():
                result[o.bias_type][o.group] = {
                    "pred": [],
                    "label": []
                }

            result[o.bias_type][o.group]["pred"].append(o.output)
            result[o.bias_type][o.group]["label"].append(o.gold_label)
        
        return result

            


    def process_task_output_to_counterfactual(self, task_output):
        """
        Transforms @task_output into a dict
        bias_type -> sentence_id -> group -> scores 
        """
        result = {}
        for o in task_output:
            if o.bias_type not in result.keys():
                result[o.bias_type] = {}

            if o.sentence_id not in result[o.bias_type].keys():
                result[o.bias_type][o.sentence_id] = {}
            
            if o.group not in result[o.bias_type][o.sentence_id].keys():
                result[o.bias_type][o.sentence_id][o.group] = {
                    "pred": [],
                    "label": []
                }

            result[o.bias_type][o.sentence_id][o.group]["pred"].append(o.output)
            result[o.bias_type][o.sentence_id][o.group]["label"].append(o.gold_label)
        
        return result





















class PairwiseComparisonMetric(BiasMetric):
    
    def bias_group(self, task_output, scoring_fct, distance_fct):

        processed_task_output = self.process_task_output_to_group(task_output)

        bias_scores = {
            k: 0 for k in processed_task_output.keys()
        }

        for bias_type in processed_task_output.keys():

            current_bias_score = 0

            groups = processed_task_output[bias_type].keys()

            group_combinations = itertools.combinations(groups, 2)
            for g1, g2 in group_combinations:
                current_bias_score += distance_fct(
                    scoring_fct(processed_task_output[bias_type][g1]),
                    scoring_fct(processed_task_output[bias_type][g2])
                )

            current_bias_score /= len(group_combinations)
            bias_scores[bias_type] = current_bias_score
        
        return bias_scores



    

    def bias_counterfactual(self, task_output, scoring_fct, distance_fct):
        processed_task_output = self.process_task_output_to_counterfactual(task_output)
        bias_scores = {
            k: 0 for k in processed_task_output.keys()
        }

        for bias_type in processed_task_output.keys():
            current_bias_score = 0
            num_sentences = len(processed_task_output[bias_type].keys())

            for s_id in processed_task_output[bias_type].keys():
                groups = processed_task_output[bias_type][s_id].keys()
                group_combinations = itertools.combinations(groups, 2)

                for g1, g2 in group_combinations:
                    current_bias_score += distance_fct(
                        scoring_fct(processed_task_output[bias_type][s_id][g1]),
                        scoring_fct(processed_task_output[bias_type][s_id][g2])
                )
            
            current_bias_score /= (len(group_combinations) * num_sentences)
            bias_scores[bias_type] = current_bias_score

        return bias_scores






























class BackgroundComparisonMetric(BiasMetric):
    def bias_group(self, task_output, scoring_fct, distance_fct):
        pass
    

    def bias_counterfactual(self, task_output, scoring_fct, distance_fct):
        pass




























class MultigroupComparisonMetric(BiasMetric):
    pass