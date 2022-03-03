import os, json, re

class TemplateProcessor:
    def __init__(self, templates_path, fillings_path, supported_file_type="json", group_token="<group>"):
        """
        @templates_path: filepath or directory to structurl templates
        @fillings_path: filepath or directory to tokens that fill templates
        """
        self.templates_path = templates_path
        self.fillings_path = fillings_path
        self.supported_file_type = supported_file_type
        self.group_token = group_token

        self.template_files = None
        self.fillings_files = None
        self.all_templates = None
        self.all_fillings = None
        self.token_to_filling_indices = None
        self.token_hierarchies = None
        self.generations = None
        
        # Read file(s) in @templates_path
        self.template_files = self.get_all_files(templates_path, check_extension=True)
        self.read_templates()

        # Read file(s) in @fillings_path
        self.fillings_files = self.get_all_files(fillings_path, check_extension=False)
        self.read_fillings()

        # Generate all templates
        self.process_templates()
        


        



    def get_all_files(self, path, check_extension=False):
        """
        Returns @path is @path is a file, or all files recursively inside @path if @path is a directory
        """
        all_files = []
        if os.path.isfile(path):
            if not check_extension or path.endswith("." + self.supported_file_type):
                all_files = [path]
            else:
                raise ValueError
        elif os.path.isdir(path):
            for p, currentDirectory, files in os.walk(path):
                for file in files:
                    if not check_extension or file.endswith("." + self.supported_file_type):
                        all_files.append(os.path.join(p, file))
        else:
            raise FileNotFoundError
        
        return all_files




    def read_fillings_file(self, filepath):
        with open(filepath, 'r') as f:
            data = f.readlines()
        return data


    def remove_extension(self, filename):
        return filename.rsplit(".", 1)[0]
    

    def read_fillings(self):
        """
        Read all fillings files 
        """
        self.all_fillings = []
        self.token_to_filling_indices = {}
        self.token_hierarchies = {}

        for f in self.fillings_files:
            file_tree = f.split(self.fillings_path)[1].strip("/")
            hierarchy_parts = self.remove_extension(file_tree).split("/")
            for i in range(len(hierarchy_parts)):
                if hierarchy_parts[i] not in self.token_hierarchies.keys():
                    self.token_hierarchies[hierarchy_parts[i]] = None if i == 0 else hierarchy_parts[i - 1]

            new_fillings = self.read_fillings_file(f)
            for new_f in new_fillings:
                if new_f in self.all_fillings:
                    raise ValueError
                else:
                    # Add the new word to self.all_fillings
                    self.all_fillings.append(new_f.strip("\n "))
                    current_index = len(self.all_fillings) - 1

                    # Associate all the token hierarchy with the new word
                    for h in hierarchy_parts:
                        if h in self.token_to_filling_indices.keys():
                            self.token_to_filling_indices[h].append(current_index)
                        else:
                            self.token_to_filling_indices[h] = [current_index]



    def read_templates_file(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data["templates"]



    def read_templates(self):
        self.all_templates = []
        for f in self.template_files:
            self.all_templates += self.read_templates_file(f)
        


        

        
    def process_templates(self):
        # Don't forget to also code the part about the same token such as <verb:1> <verb:2>
        self.generations = []
        for t in self.all_templates:
            self.generations += [{
                "text": g,
                "class": t["class"]
            } for g in self.process_template(t["text"]) ]
            







    def process_template(self, template):
        # Make implicit references explicit
        template = re.sub(r"(<)(?!group)(\w+)(>)", r"\1\2:1\3", template)

        # Get all matched tokens
        matched_tokens = list(set(re.findall(r"<(?!group)\w+:[0-9]+>", template)))

        # Replace all tokens
        return self.generate(template, matched_tokens)





    def generate(self, template, remaining_tokens):
        if len(remaining_tokens) == 0:
            return [template]
        else:
            current_token = remaining_tokens[0]
            current_token_name = current_token.strip("<>").split(":")[0] # Remove <> and :
            if current_token_name not in self.token_to_filling_indices.keys():
                raise ValueError
            else:
                generations = []
                for i in self.token_to_filling_indices[current_token_name]:
                    generations += self.generate(template.replace(current_token, self.all_fillings[i]), remaining_tokens[1:])
            
                return generations


