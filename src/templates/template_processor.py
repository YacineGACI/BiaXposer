import os, json

class TemplateProcessor:
    def __init__(self, templates_path, fillings_path, supported_file_type="json"):
        """
        @templates_path: filepath or directory to structurl templates
        @fillings_path: filepath or directory to tokens that fill templates
        """
        self.templates_path = templates_path
        self.fillings_path = fillings_path
        self.supported_file_type = supported_file_type

        self.template_files = None
        self.fillings_files = None
        
        # Read file(s) in @templates_path
        self.template_files = self.get_all_files(templates_path, check_extension=True)

        # Read file(s) in @fillings_path
        self.fillings_files = self.get_all_files(fillings_path, check_extension=False)
        self.read_fillings()
        


        



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



        

        



