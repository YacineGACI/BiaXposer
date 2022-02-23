




class BiasQuantifier:
    def __init__(self, probe, metric):
        self.probe = probe
        self.metric = metric


    def compute_bias(self):
        # Compute task-specific outputs
        self.probe_output = self.probe.run()

        # Initialize the dict of the final results
        bias_scores = {
            "overall": {},
            "per_category": {},
            "per_template": {}
        }

        # To keep the number of templates by category
        num_templates_per_category = {}

        for t in self.probe_output["per_template"]:

            # Init the result dict for each template. Will be populated by bias types and their scores
            bias_scores["per_template"][t["template"]] = {}

            for bias_type, groups in t["scores"].items():
                # Compute the bias score given group-wise outputs
                current_score = self.metric(groups.values())

                # Add it to the template
                bias_scores["per_template"][t["template"]][bias_type] = current_score

                # If the category does not exist in the keys of the result dict, add it
                if t["category"] not in bias_scores["per_category"].keys():
                    bias_scores["per_category"][t["category"]] = {}
                    num_templates_per_category[t["category"]] = 0

                # Add the category
                if bias_type in bias_scores["per_category"][t["category"]].keys():
                    bias_scores["per_category"][t["category"]][bias_type] += current_score
                else:
                    bias_scores["per_category"][t["category"]][bias_type] = current_score

                # Add it to the overall bias score
                if bias_type in bias_scores["overall"].keys():
                    bias_scores["overall"][bias_type] += current_score
                else:
                    bias_scores["overall"][bias_type] = current_score


            num_templates_per_category[t["category"]] += 1


        # Normalize the scores
        # For the overall
        for b in bias_scores["overall"].keys():
            bias_scores["overall"][b] /= len(self.probe_output["per_template"])

        # For the per category
        for c in bias_scores["per_category"].keys():
            for b in bias_scores["per_category"][c].keys():
                bias_scores["per_category"][c][b] /= num_templates_per_category[c]


        return bias_scores





