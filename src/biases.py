import logging


class Group:
    def __init__(self, group_name, definition_words=None):
        self.group_name = group_name
        self.definition_words = [] if definition_words is None else definition_words
    
    def set_name(self, new_group_name):
        self.group_name = new_group_name
    
    def add_word(self, word):
        self.definition_words.append(word)

    def remove_word(self, word):
        try:
            self.definition_words.remove(word)
        except:
            logging.warning("{} does not exist in the definition words of the group {}".format(word, self.group_name))

    def __str__(self):
        return "{}: [{}]".format(self.group_name.capitalize(), ", ".join(self.definition_words))





class BiasType:
    def __init__(self, bias_type_name, groups=None):
        self.bias_type_name = bias_type_name
        self.groups = {}
        if groups is not None:
            if not isinstance(groups, dict):
                raise ValueError
            elif all(isinstance(v, Group) for k, v in groups.items()):
                self.groups = groups
            else:
                raise ValueError


    def set_name(self, new_name):
        self.bias_type_name = new_name


    def add_group(self, group, definition_words=None):
        if isinstance(group, Group):
            self.groups[group.group_name] = group
        elif isinstance(group, str):
            self.groups[group] = Group(group, definition_words=definition_words)
        else:
            raise ValueError

    
    def remove_group(self, group):
        if group in self.groups.keys():
            del self.groups[group]


    def __str__(self):
        return_string = self.bias_type_name.upper() + "\n"
        for k, v in self.groups.items():
            return_string += "\t" + v.__str__() + "\n"
        return return_string