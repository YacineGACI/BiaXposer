import json

def zip_longest_with_cycling(*args):
    '''
        zips the set of lists in the input, and iterates over the longest list passed with cycling for the shorter lists 
    '''
    res = []
    max_length = max([len(a) for a in args])
    for i in range(max_length):
        res.append(tuple(a[i % len(a)] for a in args))
    return res





def find_span_edges(context, answer, sep_token_id):
    start_loop = context.index(sep_token_id)
    j = 1
    new_span = True
    
    for i in range(start_loop, len(context) - 1):
        if context[i] == answer[j]:
            if new_span:
                start_pos = i
                new_span = False
            j += 1
        else:
            if j == len(answer) - 1:
                break
            else:
                j = 1
                new_span = True
    
    if j == len(answer) - 1: # If we found the span in the context
        end_pos = start_pos + len(answer) - 2
    else:
        start_pos = 0
        end_pos = 1

    return start_pos, end_pos




def read_templates(filepath):
    with open(filepath, 'r') as f:
        templates = json.load(f)
    return templates




class EvalParametrizationError(Exception):
    pass