import re
import os
from collections import defaultdict
import json
import csv
import numpy as np

# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379

def clear_answer_numina(data):
    '''
    Used in NUMINA dataset
    '''
    cleanup_data = re.sub(r'[\u4e00-\u9fff]+', '', data).replace('\n', '').replace('\r', '').strip().rstrip('.') if data.strip().endswith('.') else re.sub(r'[\u4e00-\u9fff]+', '', data).replace('\n', '').replace('\r', '').strip()


    return cleanup_data

def extract_float_from_string(text):
    # Regular expressions matching numbers (integers or floating point numbers)
    match = re.search(r"-?\d+(\.\d+)?", text)
    
    if match:
        # Extract matching strings and convert to float
        return float(match.group(0))
    else:
        # If no numbers are matched, return None or appropriate error handling
        return None
    
def extract_last_float(text: str) -> float:
    """
    Extracts the last occurrence of a decimal in the text.
    :param text: the input string
    :return: the last decimal number (floating point), if none is found then return None
    """
    matches = re.findall(r"-?\d+\.\d+", text)
    return float(matches[-1]) if matches else None

def extract_keyword(text: str) -> str:
    """
    Extract the last specified word that meets the criteria.
    
    Rules:
    1. looks up the word in the list in the text and returns the last occurrence of the word.
    2. If there is no match, return None.
    
    :param text: input string
    :return: processed string or None
    """
    keywords = ['yes', 'Yes', 'YES', 'Y', 'True', 'true', 'TRUE', 'correct', 'Correct', 'CORRECT',
                'no', 'No', 'NO', 'N', 'False', 'false', 'FALSE', 'incorrect', 'Incorrect', 'INCORRECT']
    
    # Find all matching keywords
    matches = [word for word in text.split() if word in keywords]
    
    return matches[-1] if matches else None

def numina_pm_answer_match(pred, gts):
    '''
    Check if the predicted answer matches any of the ground truth answers in the NUMINA dataset.

    Parameters:
    pred (str): The predicted answer.
    gts (list of str): A list of ground truth answers.

    Returns:
    int: 1 if there is a match, 0 if there is no match, or 0 if pred is an empty string.
    '''
    return 1 if pred and pred in gts else 0

def numina_fv_answer_match(pred, gts):
    return 1 if pred and pred in gts else 0


def numina_ni_answer_match(pred, gt, thresholds=[0.05, 0.10, 0.20]):
    
    if pred == None:
        return [0] * len(thresholds)
    
    gt += 1e-6 if gt == 0 else gt
    
    return [
        1 if abs((pred - gt) / gt) < threshold else 0
        for threshold in thresholds
    ]
        
    

def clean_answer(data):
    '''
    Designed to clean and normalize text data, specifically answers. It performs a series of text substitutions and transformations to ensure the text is in a consistent and clean format.
    '''

    # Convert to Lowercase
    data = data.lower()  

    # Remove leading and trailing whitespaces
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)

    # Replace Multiple Spaces with a Single Space
    data = re.sub(' {2,}', ' ', data)

    # Fix Spacing After Periods
    data = re.sub('\.[ ]{2,}', '. ', data)

    # Remove Unwanted Characters
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)

    # Character Substitutions
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)

    # Correct Common Misspellings
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def answer_match(pred, gts):
    # return EM and refined EM
    if len(pred) == 0:
        return 0, 0
    if pred in gts:
        return 1, 1
    
    # ''.join(pred.split()): Removes all whitespace characters from pred
    # '.join(gt.split()): Removes all whitespace characters from gt
    # The provided code snippet checks if the predicted text (pred) is a substring of any ground truth text (gt) or vice versa, after removing all whitespace characters. If it is, the function returns 0 and 1, indicating that the prediction is not an exact match but is a partial match.
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


# refer to ScanRefer
# https://github.com/daveredrum/ScanRefer/blob/9d7483053e8d29acfd4db4eb1bc28f1564f5dddb/lib/dataset.py#L243-L314
def scanrefer_get_unique_multiple_lookup():
    unique_multiple_lookup_file = 'annotations/scanrefer_unique_multiple_lookup.json'
    if not os.path.exists(unique_multiple_lookup_file):
        type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
        scannet_labels = type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        label_classes_set = set(scannet_labels)
        raw2label = {}
        with open('annotations/scannet/scannetv2-labels.combined.tsv', 'r') as f:
            csvreader = csv.reader(f, delimiter='\t')
            csvreader.__next__()
            for line in csvreader:
                raw_name = line[1]
                nyu40_name = line[7]
                if nyu40_name not in label_classes_set:
                    raw2label[raw_name] = scannet2label['others']
                else:
                    raw2label[raw_name] = scannet2label[nyu40_name]
        all_sem_labels = defaultdict(list)
        cache = defaultdict(dict)
        scanrefer_data = json.load(open('annotations/scanrefer/ScanRefer_filtered.json'))
        for data in scanrefer_data:
            scene_id = data['scene_id']
            object_id = data['object_id']
            object_name = ' '.join(data['object_name'].split('_'))
            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(raw2label[object_name])
                except:
                    all_sem_labels[scene_id].append(17)
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}
        unique_multiple_lookup = defaultdict(dict)
        for data in scanrefer_data:
            scene_id = data['scene_id']
            object_id = data['object_id']
            object_name = ' '.join(data['object_name'].split('_'))
            try:
                sem_label = raw2label[object_name]
            except:
                sem_label = 17
            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
            unique_multiple_lookup[scene_id][object_id] = unique_multiple
        with open(unique_multiple_lookup_file, 'w') as f:
            json.dump(unique_multiple_lookup, f, indent=4)
    else:
        unique_multiple_lookup = json.load(open(unique_multiple_lookup_file))
    return unique_multiple_lookup