import tensorflow as tf
import numpy as np
import os, csv, timeit
from pprint import pprint
import pandas as pd
from os.path import join
try:
    import cv2
except ImportError:
    pass

from collections import defaultdict
import unicodedata
import random
import tokenization
import json


CONFIG_FILE = "variables.json"

def load_config():
    res = {}
    with open(CONFIG_FILE) as config_file:
        data = json.load(config_file)    
    for k, v in data.items():
        res[k] = os.environ.get(k, v)
        
    return res

args = load_config()

# Variables
c_threshold = 0.5
dict_path = args['dict_path']
classes = ['DontCare', 'vendor_name', 'vendor_VAT', 'vendor_address', 'vendor_IBAN', 'client_name', 'client_VAT', 'client_address', 'date', 'invoice_ID', 'prod_ref', 'prod_concept', 'prod_lot', 'prod_qty', 'prod_uprice', 'prod_total', 'base_price', 'VAT', 'total']
num_classes = len(classes)
update_dict = False
special_dict = {'*', '='}
text_case = False
tokenize = True
encoding_factor = 1
dict_vocab = args['dict_vocab']
if tokenize:
    tokenizer = tokenization.FullTokenizer(dict_vocab, do_lower_case=not text_case)
segment_grid = False
cols_segment = 72
rows_target = 64
cols_target = 64
rows_ulimit = 80  # handle OOM, must be multiple of self.encoding_factor
cols_ulimit = 80  # handle OOM, must be multiple of self.encoding_factor
load_dictionary = True
da_extra_rows = 0
da_extra_cols = 0
augment_strategy = 1
num_classes = 19
pm_strategy = 1


def load_dict(dict_path,load_dictionary):
    if load_dictionary:
        dictionary = np.load(dict_path + '_dictionary.npy', allow_pickle=True).item()
        word_to_index = np.load(dict_path + '_word_to_index.npy', allow_pickle=True).item()
        index_to_word = np.load(dict_path + '_index_to_word.npy', allow_pickle=True).item()
    
    return dictionary, word_to_index, index_to_word

dictionary, word_to_index, index_to_word = load_dict(dict_path,load_dictionary)

def load_data(data, update_dict=False):
    """
    label_dressed in format:
    {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
    load doc words with location and class returned in format:
    [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
    """
    doc_dressed = []

    file_id = data['global_attributes']['file_id']
    
    data = _collect_data(file_id, data['text_boxes'], update_dict)
    for i in data:
        doc_dressed.append(i)

    return doc_dressed


def _collect_data(file_name, content, update_dict):
    """
    dress and preserve only interested data.
    """
    content_dressed = []
    left, top, right, bottom, buffer = 9999, 9999, 0, 0, 2
    for line in content:
        bbox = line['bbox']  # handle data corrupt
        if len(bbox) == 0:
            continue
        if line['text'] in special_dict:  # ignore potential overlap causing characters
            continue

        x_left, y_top, x_right, y_bottom = _dress_bbox(bbox)
        # TBD: the real image size is better for calculating the relative x/y/w/h
        if x_left < left: left = x_left - buffer
        if y_top < top: top = y_top - buffer
        if x_right > right: right = x_right + buffer
        if y_bottom > bottom: bottom = y_bottom + buffer

        word_id = line['id']
        #print(line['text'])
        dressed_texts = _dress_text(line['text'], update_dict)
        #print(dressed_texts)
        num_block = len(dressed_texts)
        for i, dressed_text in enumerate(dressed_texts):  # handling tokenized text, separate bbox
            new_left = int(x_left + (x_right - x_left) / num_block * (i))
            new_right = int(x_left + (x_right - x_left) / num_block * (i + 1))
            content_dressed.append([file_name, dressed_text, word_id, [new_left, y_top, new_right, y_bottom]])

    # initial calculation of maximum number of words in rows/cols in terms of image size
    num_words_row = [0 for _ in range(bottom)]  # number of words in each row
    num_words_col = [0 for _ in range(right)]  # number of words in each column
    for line in content_dressed:
        _, _, _, [x_left, y_top, x_right, y_bottom] = line
        for y in range(y_top, y_bottom):
            num_words_row[y] += 1
        for x in range(x_left, x_right):
            num_words_col[x] += 1
    max_row_words = _fit_shape(max(num_words_row))
    max_col_words = 0  # self._fit_shape(max(num_words_col))

    # further expansion of maximum number of words in rows/cols in terms of grid shape
    max_rows = max(encoding_factor, max_row_words)
    max_cols = max(encoding_factor, max_col_words)
    DONE = False
    while not DONE:
        DONE = True
        grid_table = np.zeros([max_rows, max_cols], dtype=np.int32)
        for line in content_dressed:
            _, _, _, [x_left, y_top, x_right, y_bottom] = line
            row = int(max_rows * (y_top - top + (y_bottom - y_top) / 2) / (bottom - top))
            col = int(max_cols * (x_left - left + (x_right - x_left) / 2) / (right - left))
            # row = int(max_rows * (y_top + (y_bottom-y_top)/2) / (bottom))
            # col = int(max_cols * (x_left + (x_right-x_left)/2) / (right))
            # row = int(max_rows * (y_top-top) / (bottom-top))
            # col = int(max_cols * (x_left-left) / (right-left))
            # row = int(max_rows * (y_top) / (bottom))
            # col = int(max_cols * (x_left) / (right))
            # row = int(max_rows * (y_top + (y_bottom-y_top)/2) / bottom)
            # col = int(max_cols * (x_left + (x_right-x_left)/2) / right)

            while col < max_cols and grid_table[row, col] != 0:  # shift to find slot to drop the current item
                col += 1
            if col == max_cols:  # shift to find slot to drop the current item
                col -= 1
                ptr = 0
                while ptr < max_cols and grid_table[row, ptr] != 0:
                    ptr += 1
                if ptr == max_cols:  # overlap cannot be solved in current row, then expand the grid
                    max_cols = _expand_shape(max_cols)
                    DONE = False
                    break

                grid_table[row, ptr:-1] = grid_table[row, ptr + 1:]

            if DONE:
                if row > max_rows or col > max_cols:
                    print('wrong')
                grid_table[row, col] = 1

    max_rows = _fit_shape(max_rows)
    max_cols = _fit_shape(max_cols)

    #print('{} collected in shape: {},{}'.format(file_name, max_rows, max_cols))

    # segment grid into two parts if number of cols is larger than self.cols_target
    data = []
    if segment_grid and max_cols > cols_segment:
        content_dressed_left = []
        content_dressed_right = []
        cnt = defaultdict(int)  # counter for number of words in a specific row
        cnt_l, cnt_r = defaultdict(int), defaultdict(int)  # update max_cols if larger than self.cols_segment
        left_boundary = max_cols - cols_segment
        right_boundary = cols_segment
        for i, line in enumerate(content_dressed):
            file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line

            row = int(max_rows * (y_top + (y_bottom - y_top) / 2) / bottom)
            cnt[row] += 1
            if cnt[row] <= left_boundary:
                cnt_l[row] += 1
                content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                             [left, top, right, bottom], max_rows, cols_segment])
            elif left_boundary < cnt[row] <= right_boundary:
                cnt_l[row] += 1
                cnt_r[row] += 1
                content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                             [left, top, right, bottom], max_rows, cols_segment])
                content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                              [left, top, right, bottom], max_rows,
                                              max(max(cnt_r.values()), cols_segment)])
            else:
                cnt_r[row] += 1
                content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                              [left, top, right, bottom], max_rows,
                                              max(max(cnt_r.values()), cols_segment)])
        # print(sorted(cnt.items(), key=lambda x:x[1], reverse=True))
        # print(sorted(cnt_l.items(), key=lambda x:x[1], reverse=True))
        # print(sorted(cnt_r.items(), key=lambda x:x[1], reverse=True))
        if max(cnt_l.values()) < 2 * cols_segment:
            data.append(content_dressed_left)
        if max(cnt_r.values()) < 2 * cols_segment:  # avoid OOM, which tends to happen in the right side
            data.append(content_dressed_right)
    else:
        for i, line in enumerate(content_dressed):  # append height/width/numofwords to the list
            file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
            content_dressed[i] = [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                  [left, top, right, bottom], max_rows, max_cols]
        data.append(content_dressed)

    return data

def _dress_bbox(bbox):
    positions = np.array(bbox).reshape([-1]).astype(np.float) #JB: needed to add this to assure that all elements are numerical
    x_left = max(0, min(positions[0::2]))
    x_right = max(positions[0::2])
    y_top = max(0, min(positions[1::2]))
    y_bottom = max(positions[1::2])
    w = x_right - x_left
    h = y_bottom - y_top

    return int(x_left), int(y_top), int(x_right), int(y_bottom)

def _dress_text(text, update_dict):
    """
    three cases covered:
    alphabetic string, numeric string, special character
    """
    string = text if text_case else text.lower()
    for i, c in enumerate(string):
        if is_number(c):
            string = string[:i] + '0' + string[i + 1:]

    strings = [string]
    if tokenize:
        strings = tokenizer.tokenize(strings[0])
        #print(string, '-->', strings)

    for idx, string in enumerate(strings):
        if string.isalpha():
            if string in special_dict:
                string = special_dict[string]
            # TBD: convert a word to its most similar word in a known vocabulary
        elif is_number(string):
            pass
        elif len(string) == 1:  # special character
            pass
        else:
            # TBD: seperate string as parts for alpha and number combinated strings
            # string = re.findall('[a-z]+', string)
            pass

        if string not in dictionary.keys():
            if update_dict:
                dictionary[string] = 0
            else:
                print('unknown text: ' + string)
                string = '[UNK]'  # TBD: take special care to unmet words\
        dictionary[string] += 1

        strings[idx] = string

    return strings

def _fit_shape(shape): # modify shape size to fit the encoding factor
    while shape % encoding_factor:
        shape += 1
    return shape

def _expand_shape(shape): # expand shape size with step 2
    return _fit_shape(shape+1)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def prepare_data(docs):

    ## fixed validation shape leads to better result (to be verified)
    real_rows, real_cols, _, _ = _cal_rows_cols(docs, extra_augmentation=False)
    rows = max(rows_target, real_rows)
    cols = max(rows_target, real_cols)

    grid_table, bboxes, bbox_mapids, file_names, updated_cols, ps_indices_x, ps_indices_y, grid_word_ids = \
        _positional_mapping(docs, rows, cols)
    if updated_cols > cols:
        print('Validation grid EXPAND size: ({},{}) from ({},{})' \
              .format(rows, updated_cols, rows, cols))
        grid_table, bboxes, bbox_mapids, file_names, _, ps_indices_x, ps_indices_y, grid_word_ids = \
            _positional_mapping(docs, rows, updated_cols,
                                     update_col=False)

    batch = {'grid_table': np.array(grid_table),
             'data_image': '', 'ps_1d_indices': '',
             # @images and @ps_1d_indices are only used for CUTIEv2
             'bboxes': bboxes, 'bbox_mapids': bbox_mapids,
             'file_name': file_names, 'shape': [rows, cols],
             'grid_word_ids': grid_word_ids}
    return batch

def _cal_rows_cols(docs, extra_augmentation=False, dropout=False):
    max_row = encoding_factor
    max_col = encoding_factor
    for doc in docs:
        for line in doc:
            _, _, _, _, _, max_row_words, max_col_words = line
            if max_row_words > max_row:
                max_row = max_row_words
            if max_col_words > max_col:
                max_col = max_col_words

    pre_rows = _fit_shape(max_row)  # (max_row//self.encoding_factor+1) * self.encoding_factor
    pre_cols = _fit_shape(max_col)  # (max_col//self.encoding_factor+1) * self.encoding_factor

    rows, cols = 0, 0
    if extra_augmentation:
        pad_row = int(random.gauss(0, da_extra_rows * encoding_factor))  # abs(random.gauss(0, u))
        pad_col = int(random.gauss(0, da_extra_cols * encoding_factor))  # random.randint(0, u)

        if augment_strategy == 1:  # strategy 1: augment data by increasing grid shape sizes
            pad_row = abs(pad_row)
            pad_col = abs(pad_col)
            rows = _fit_shape(max_row + pad_row)  # apply upper boundary to avoid OOM
            cols = _fit_shape(max_col + pad_col)  # apply upper boundary to avoid OOM
        elif augment_strategy == 2 or augment_strategy == 3:  # strategy 2: augment by increasing or decreasing the target gird shape size
            rows = _fit_shape(max(rows_target + pad_row, max_row))  # protect grid shape
            cols = _fit_shape(max(cols_target + pad_col, max_col))  # protect grid shape
        else:
            raise Exception('unknown augment strategy')
        rows = min(rows, rows_ulimit)  # apply upper boundary to avoid OOM
        cols = min(cols, cols_ulimit)  # apply upper boundary to avoid OOM
    else:
        rows = pre_rows
        cols = pre_cols
    return rows, cols, pre_rows, pre_cols

def _positional_mapping(docs, rows, cols):
    """
    docs in format:
    [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
    return grid_tables, gird_labels, dict bboxes {file_name:[]}, file_names
    """
    grid_tables = []
    grid_words_ids = []
    #gird_labels = []
    ps_indices_x = []  # positional sampling indices
    ps_indices_y = []  # positional sampling indices
    bboxes = {}
    #label_mapids = []
    bbox_mapids = []  # [{}, ] bbox identifier, each id with one or multiple bbox/bboxes
    file_names = []
    for doc in docs:
        items = []
        cols_e = 2 * cols  # use @cols_e larger than required @cols as buffer
        grid_table = np.zeros([rows, cols_e], dtype=np.int32)
        grid_word_ids = np.zeros([rows, cols_e], dtype=np.int32)
        #grid_label = np.zeros([rows, cols_e], dtype=np.int8)
        ps_x = np.zeros([rows, cols_e], dtype=np.int32)
        ps_y = np.zeros([rows, cols_e], dtype=np.int32)
        bbox = [[] for c in range(cols_e) for r in range(rows)]
        bbox_id, bbox_mapid = 0, {}  # one word in one or many positions in a bbox is mapped in bbox_mapid
        label_mapid = [[] for _ in range(num_classes)]  # each class is connected to several bboxes (words)
        drawing_board = np.zeros([rows, cols_e], dtype=str)
        for item in doc:
            file_name = item[0]
            text = item[1]
            word_id = item[2]
            x_left, y_top, x_right, y_bottom = item[3][:]
            left, top, right, bottom = item[4][:]
            dict_id = word_to_index[text]
 
            bbox_id += 1

            box_y = y_top + (y_bottom - y_top) / 2
            box_x = x_left  # h_l is used for image feature map positional sampling
            v_c = (y_top - top + (y_bottom - y_top) / 2) / (bottom - top)
            h_c = (x_left - left + (x_right - x_left) / 2) / (right - left)  # h_c is used for sorting items
            row = int(rows * v_c)
            col = int(cols * h_c)


            #items.append([row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, class_id, entity_id, bbox_id,
            #              [x_left, y_top, x_right - x_left, y_bottom - y_top]])
            items.append([row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, bbox_id,
                          [x_left, y_top, x_right - x_left, y_bottom - y_top], word_id])

        #items.sort(key=lambda x: (x[0], x[3], x[5]))  # sort according to row > h_c > bbox_id
        items.sort(key=lambda x: (x[0], x[3], x[6]))  #TODO sort according to row > h_c > bbox_id
        for item in items:
            row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, bbox_id, box, word_id = item
            #entity_class_id = entity_id * num_classes + class_id

            while col < cols and grid_table[row, col] != 0:
                col += 1

                # self.pm_strategy 0: skip if overlap
            # self.pm_strategy 1: shift to find slot if overlap
            # self.pm_strategy 2: expand grid table if overlap
            if pm_strategy == 0:
                if col == cols:
                    print('overlap in {} row {} r{}c{}!'.
                          format(file_name, row, rows, cols))
                    # print(grid_table[row,:])
                    # print('overlap in {} <{}> row {} r{}c{}!'.
                    #      format(file_name, self.index_to_word[dict_id], row, rows, cols))
                else:
                    grid_table[row, col] = dict_id
                    #grid_label[row, col] = entity_class_id
                    bbox_mapid[row * cols + col] = bbox_id
                    bbox[row * cols + col] = box
            elif pm_strategy == 1 or pm_strategy == 2:
                ptr = 0
                if col == cols:  # shift to find slot to drop the current item
                    col -= 1
                    while ptr < cols and grid_table[row, ptr] != 0:
                        ptr += 1
                    if ptr == cols:
                        grid_table[row, :-1] = grid_table[row, 1:]
                    else:
                        grid_table[row, ptr:-1] = grid_table[row, ptr + 1:]

                if pm_strategy == 2:
                    while col < cols_e and grid_table[row, col] != 0:
                        col += 1
                    if col > cols:  # update maximum cols in current grid
                        print(grid_table[row, :col])
                        print('overlap in {} <{}> row {} r{}c{}!'.
                              format(file_name, index_to_word[dict_id], row, rows, cols))
                        cols = col
                    if col == cols_e:
                        print('overlap!')

                grid_table[row, col] = dict_id
                grid_word_ids[row, col] = word_id
                ps_x[row, col] = box_x
                ps_y[row, col] = box_y
                #bbox_mapid[row * cols + col] = bbox_id
                bbox[row * cols + col] = box

        cols = _fit_shape(cols)
        grid_table = grid_table[..., :cols]
        grid_word_ids = grid_word_ids[..., :cols]
        ps_x = np.array(ps_x[..., :cols])
        ps_y = np.array(ps_y[..., :cols])

        grid_tables.append(np.expand_dims(grid_table, -1))
        grid_words_ids.append(np.expand_dims(grid_word_ids, -1))
        #gird_labels.append(grid_label)
        ps_indices_x.append(ps_x)
        ps_indices_y.append(ps_y)
        bboxes[file_name] = bbox
        bbox_mapids.append(bbox_mapid)
        file_names.append(file_name)

    return grid_tables, bboxes, bbox_mapids, file_names, cols, ps_indices_x, ps_indices_y, grid_words_ids

def update_word_to_index(word_to_index, dictionary, index_to_word):
    if load_dictionary:
        max_index = len(word_to_index.keys())
        for word in dictionary:
            if word not in word_to_index:
                max_index += 1
                word_to_index[word] = max_index
                index_to_word[max_index] = word            
    else:   
        word_to_index = dict(list(zip(dictionary.keys(), list(range(num_words))))) 
        index_to_word = dict(list(zip(list(range(num_words)), dictionary.keys())))


def make_results(grid_table, model_output_val, bbox_mapids, grid_word_ids, index_to_word):
    #num_tp = 0
    #num_fn = 0
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_correct_soft = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    #index_to_word = np.load(dict_path + '_index_to_word.npy', allow_pickle=True).item()

    preds = []
    total_words = 0
    preds_words = 0

    df = pd.DataFrame()

    words = []
    words_ids = []
    classes_df = []
    predictions = []

    dict_result = {}
    for cls in classes:
        dict_result[cls] = []

    for b in range(grid_table.shape[0]):
        
        data_input_flat = grid_table[b,:,:,0].reshape([-1])
        print(grid_table.shape)
        print(data_input_flat.shape)

        word_ids_flat = grid_word_ids[b,:,:,0].reshape([-1])
        #labels = gt_classes[b,:,:].reshape([-1])
        logits = model_output_val[b,:,:,:].reshape([-1, num_classes])
        #label_mapid = label_mapids[b]
        bbox_mapid = bbox_mapids[b]
        rows, cols = grid_table.shape[1:3]
        bbox_id = np.array([row*cols+col for row in range(rows) for col in range(cols)])

        # ignore inputs that are not word
        indexes = np.where(data_input_flat != 0)[0]
        data_selected = data_input_flat[indexes]
        word_ids_selected = word_ids_flat[indexes]

        # print(word_ids_selected, "todos: ", word_ids_flat)

        #labels_selected = labels[indexes]
        logits_array_selected = logits[indexes]
        bbox_id_selected = bbox_id[indexes]
        print("len indexes: ",len(indexes))
        total_words = len(indexes)
        # calculate accuracy
        for c in range(1, num_classes):
            #print("clase: ", c, classes[c])
            #labels_indexes = np.where(labels_selected == c)[0]
            logits_indexes = np.where(logits_array_selected[:,c] > c_threshold)[0]
            #print(logits_indexes)
            
            logits_flat = logits_array_selected[logits_indexes,c]
              
            #print(logits_flat)
            # if logits_flat != []:
            #     print("no esta vacio")
            for l in range(len(logits_flat)):
                    preds.append(logits_flat[l])
        
            logits_id_word = word_ids_selected[logits_indexes]
           
            #labels_words = list(index_to_word[i] for i in data_selected[labels_indexes])
            logits_words = list(index_to_word[i] for i in data_selected[logits_indexes])
            
            #label_bbox_ids = label_mapid[c] # GT bbox_ids related to the type of class
            logit_bbox_ids = [bbox_mapid[bbox] for bbox in bbox_id_selected[logits_indexes] if bbox in bbox_mapid]            
            
            # print(logits_words, logit_bbox_ids, logits_id_word)
            
            classes_df.append(classes[c])
            words.append(logits_words)
            words_ids.append(np.unique(logits_id_word))
            predictions.append(logits_flat)
                 
            if b==0:
                res = ' '.join(index_to_word[i] for i in data_selected[logits_indexes])
                res += '"'

                for i in data_selected[logits_indexes]:
                    val = index_to_word[i]
                    dict_result[classes[c]].append(val)

    df['class'] = classes_df
    df['id_word'] = words_ids
    df['words'] = words
    df['predictions'] = predictions
    print(df)
    preds_words = len(preds)
    return dict_result, preds, preds_words, total_words, df


def make_json_results(df_pred_ids, data_json):
    val_class = df_pred_ids['class'].values
    val_ids = df_pred_ids['id_word']
    val_preds = df_pred_ids['predictions']
    #print(data_json)

    found_ids = []
    for i, val in enumerate(val_class):
        # print(val, i)
        text = []
        for x in range(0, len(data_json['text_boxes'])):
            for num in val_ids[i]:
                if data_json['text_boxes'][x]['id'] == num:
                    text.append(data_json['text_boxes'][x]['text'])

        # print(text, len(text))
        # print(val_ids[i], len(val_ids[i]))
        data_json['fields'][i+1].update({'value_text': list(text), "value_id": list(val_ids[i])}) 
        #print(data_json['fields'][i+1])

        for v in val_ids[i]:
            found_ids.append(v)

    differ = np.setdiff1d(data_json['fields'][0]['value_id'], found_ids)
    
    text2 = []
    for i in range(0, len(data_json['text_boxes'])):
        for d in differ:
            if data_json['text_boxes'][i]['id'] == d:
                text2.append(data_json['text_boxes'][i]['text'])
    
    data_json['fields'][0].update({'value_text': text2, "value_id": list(differ)}) 

    # add preds
    data_json_convert = json.dumps(data_json, default=int)
    data_json = json.loads(data_json_convert)

    for x in range(0, len(data_json['text_boxes'])):
        for i in range(0,len(val_ids)):
            for j, id in enumerate(val_ids[i]):
                if data_json['text_boxes'][x]['id'] == id:
                    # print(id, data_json['text_boxes'][x]['id'])
                    #print(type(val_preds[i][j]))
                    data_json['text_boxes'][x]['pred'] = float(val_preds[i][j])

    return data_json


def cal_save_results(data_loader, grid_table, gt_classes, model_output_val, label_mapids, bbox_mapids, file_names, save_prefix):
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_correct_soft = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(grid_table.shape[0]):
        filename = file_names[0]
        
        data_input_flat = grid_table[b,:,:,0].reshape([-1])
        labels = gt_classes[b,:,:].reshape([-1])
        logits = model_output_val[b,:,:,:].reshape([-1, data_loader.num_classes])
        label_mapid = label_mapids[b]
        bbox_mapid = bbox_mapids[b]
        rows, cols = grid_table.shape[1:3]
        bbox_id = np.array([row*cols+col for row in range(rows) for col in range(cols)])
        
        # ignore inputs that are not word
        indexes = np.where(data_input_flat != 0)[0]
        data_selected = data_input_flat[indexes]
        labels_selected = labels[indexes]
        logits_array_selected = logits[indexes]
        bbox_id_selected = bbox_id[indexes]
        
        # calculate accuracy
        for c in range(1, data_loader.num_classes):
            labels_indexes = np.where(labels_selected == c)[0]
            logits_indexes = np.where(logits_array_selected[:,c] > c_threshold)[0]
            
            labels_words = list(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
            logits_words = list(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
            
            label_bbox_ids = label_mapid[c] # GT bbox_ids related to the type of class
            logit_bbox_ids = [bbox_mapid[bbox] for bbox in bbox_id_selected[logits_indexes] if bbox in bbox_mapid]            
            
            #if np.array_equal(labels_indexes, logits_indexes):
            if set(label_bbox_ids) == set(logit_bbox_ids): # decide as correct when all ids match
                num_correct_strict += 1  
                num_correct_soft += 1
            elif set(label_bbox_ids).issubset(set(logit_bbox_ids)): # correct when gt is subset of gt
                num_correct_soft += 1
            try: # calculate prevalence with decimal precision
                num_correct += np.shape(np.intersect1d(labels_indexes, logits_indexes))[0] / np.shape(labels_indexes)[0]
            except ZeroDivisionError:
                if np.shape(labels_indexes)[0] == 0:
                    num_correct += 1
                else:
                    num_correct += 0        
            
            # show results without the <DontCare> class      
            
            # ground truth label
            gt = str(' '.join(data_loader.index_to_word[i] for i in data_selected[labels_indexes]))
            predict = str(' '.join(data_loader.index_to_word[i] for i in data_selected[logits_indexes]))
            
        
            # write results to csv
            fieldnames = ['TaskID', 'GT', 'Predicted']
            
            csv_filename = 'data/results/' + save_prefix + '_' + data_loader.classes[c] + '.csv'            
            writer = csv.DictWriter(open(csv_filename, 'a'), fieldnames=fieldnames) 
            row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
            writer.writerow(row)
            
            csv_diff_filename = 'data/results/' + save_prefix + '_Diff_' + data_loader.classes[c] + '.csv'
            if gt != predict:
                writer = csv.DictWriter(open(csv_diff_filename, 'a'), fieldnames=fieldnames) 
                row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
                writer.writerow(row)
            
            if b == 0:
                res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
                res += gt + '" | "' + predict + '"'
                # wrong inferences results
                if not np.array_equal(labels_indexes, logits_indexes): 
                    res += '\n \t FALSES =>>'
                    logits_flat = logits_array_selected[:,c]
                    fault_logits_indexes = np.setdiff1d(logits_indexes, labels_indexes)
                    for i in range(len(data_selected)):
                        if i not in fault_logits_indexes: # only show fault_logits_indexes
                            continue
                        w = data_loader.index_to_word[data_selected[i]]
                        l = data_loader.classes[labels_selected[i]]
                        res += ' "%s"/%s, '%(w, l)
                        #res += ' "%s"/%.2f%s, '%(w, logits_flat[i], l)
                        
            #print(res)
    prevalence = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    accuracy_soft = num_correct_soft / num_all
    return prevalence, accuracy_strict, accuracy_soft, res.encode("utf-8")


def vis_bbox(file_prefix, grid_table, model_output_val, file_name, bboxes, shape):
    data_input_flat = grid_table.reshape([-1])
    labels = classes
    rows = 1
    cols = 1
    logits = model_output_val.reshape([-1, num_classes])
    bboxes = bboxes.reshape([-1])
    
    max_len = 768*2 # upper boundary of image display size 
    img = cv2.imread(join(file_prefix, file_name))
    print(join(file_prefix, file_name))
    print("readed img")
    if img is not None:    
        shape = list(img.shape)
        
        bbox_pad = 1
        #gt_color = [[255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237],
        #            [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128], [255, 105, 180]]

        inf_color = [[255, 222, 173], [0, 255, 255], [50,219,50], [72, 61, 139],
                     [154, 205, 50], [50,50,219], [64,76,64], [255, 0, 0], [255, 20, 147],
                     [255, 222, 173], [0, 255, 255], [50, 219, 50], [72, 61, 139],
                     [154, 205, 50], [50,50,219], [64,76,64], [255, 0, 0], [255, 20, 147], [255, 222, 173]]

        gt_color = [[255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237],
                        [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128],
                        [255, 105, 180], [152, 245, 255], [119,204,119],
                        [100, 149, 237], [192, 255, 62], [119, 119, 204], [114, 124, 114],
                        [240, 128, 128], [255, 105, 180],[152, 245, 255], [119,204,119]]
        
        font_size = 0.5
        font = cv2.FONT_HERSHEY_COMPLEX
        ft_color = [50, 50, 250]
        
        factor = max_len / max(shape)
        shape[0], shape[1] = [int(s*factor) for s in shape[:2]]
        
        img = cv2.resize(img, (shape[1], shape[0]))        
        overlay_box = np.zeros(shape, dtype=img.dtype)
        overlay_line = np.zeros(shape, dtype=img.dtype)
        added = []
        for i in range(len(data_input_flat)):
            if len(bboxes[i]) > 0:
                x,y,w,h = [int(p*factor) for p in bboxes[i]]
            else:
                row = i // rows
                col = i % cols
                x = shape[1] // cols * col
                y = shape[0] // rows * row
                w = shape[1] // cols * 2
                h = shape[0] // cols * 2
                
            #if data_input_flat[i] and labels[i]:
            #    gt_id = labels[i]
            #    cv2.rectangle(overlay_box, (x,y), (x+w,y+h), gt_color[gt_id], -1)
                    
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:                
                    cv2.rectangle(overlay_line, (x+bbox_pad,y+bbox_pad), \
                                  (x+bbox_pad+w,y+bbox_pad+h), inf_color[inf_id], max_len//768*2)
                    text = labels[inf_id]
                    if inf_id not in added:
                        added.append(inf_id)
                        cv2.putText(img, text, (x, y), font, font_size, inf_color[inf_id])#ft_color)
                
            #text = labels[gt_id] + '|' + labels[inf_id]
            #cv2.putText(img, text, (x,y), font, font_size, ft_color)
        
        # legends
        w = shape[1] // cols * 4
        h = shape[0] // cols * 2
        for i in range(1, len(classes)):
           row = i * 3
           col = 0
           x = shape[1] // cols * col
           y = shape[0] // rows * row
           cv2.rectangle(img, (x,y), (x+w,y+h), gt_color[i], -1)
           cv2.putText(img, classes[i], (x+w,y+h), font, 0.8, ft_color)
            
           row = i * 3 + 1
           col = 0
           x = shape[1] // cols * col
           y = shape[0] // rows * row
           cv2.rectangle(img, (x+bbox_pad,y+bbox_pad), \
                         (x+bbox_pad+w,y+bbox_pad+h), inf_color[i], max_len//384)
        
        alpha = 0.4
        cv2.addWeighted(overlay_box, alpha, img, 1-alpha, 0, img)
        cv2.addWeighted(overlay_line, 1-alpha, img, 1, 0, img)
        cv2.imwrite('results/' + file_name[:-4]+'.png', img)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        

def cal_accuracy_table(data_loader, grid_table, gt_classes, model_output_val, label_mapids, bbox_mapids):
    #num_tp = 0
    #num_fn = 0
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_correct_soft = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(grid_table.shape[0]):
        data_input_flat = grid_table[b,:,:,0]
        rows, cols = grid_table.shape[1:3]
        labels = gt_classes[b,:,:]
        logits = model_output_val[b,:,:,:].reshape([rows, cols, data_loader.num_classes])
        label_mapid = label_mapids[b]
        bbox_mapid = bbox_mapids[b]
        bbox_id = np.array([row*cols+col for row in range(rows) for col in range(cols)])
        
        # calculate accuracy
        #test_classes = [1,2,3,4,5]
        #for c in test_classes:
        for c in range(1, data_loader.num_classes):
            label_rows, label_cols = np.where(labels == c)
            logit_rows, logit_cols = np.where(logits[:,:,c] > c_threshold)
            if min(label_rows) == min(logit_rows) and max(label_cols) == max(logit_cols):
                num_correct_strict += 1
                num_correct_soft += 1
                num_correct += 1
            if min(label_rows) > min(logit_rows) and max(label_cols) < max(logit_cols):
                num_correct_soft += 1
                num_correct += 1
            
    prevalence = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    accuracy_soft = num_correct_soft / num_all
    return prevalence, accuracy_strict, accuracy_soft, res.encode("utf-8")


def vis_table(data_loader, file_prefix, grid_table, gt_classes, model_output_val, file_name, bboxes, shape):
    data_input_flat = grid_table.reshape([-1])
    labels = gt_classes.reshape([-1])
    logits = model_output_val.reshape([-1, data_loader.num_classes])
    bboxes = bboxes.reshape([-1])
    
    max_len = 768*2 # upper boundary of image display size 
    img = cv2.imread(join(file_prefix, file_name))
    if img is not None:    
        shape = list(img.shape)
        
        bbox_pad = 1
        gt_color = [[255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237], 
                    [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128], [255, 105, 180]]
        inf_color = [[255, 222, 173], [0, 255, 255], [50,219,50], [72, 61, 139], 
                     [154, 205, 50], [50,50,219], [64,76,64], [255, 0, 0], [255, 20, 147]]
        
        font_size = 0.5
        font = cv2.FONT_HERSHEY_COMPLEX
        ft_color = [50, 50, 250]
        
        factor = max_len / max(shape)
        shape[0], shape[1] = [int(s*factor) for s in shape[:2]]
        
        img = cv2.resize(img, (shape[1], shape[0]))        
        overlay_box = np.zeros(shape, dtype=img.dtype)
        overlay_line = np.zeros(shape, dtype=img.dtype)
        gt_x, gt_y, gt_r, gt_b = 99999, 99999, 0, 0
        inf_x, inf_y, inf_r, inf_b = 99999, 99999, 0, 0
        for i in range(len(data_input_flat)):
            if len(bboxes[i]) > 0:
                x,y,w,h = [int(p*factor) for p in bboxes[i]]
            else:
                row = i // data_loader.rows
                col = i % data_loader.cols
                x = shape[1] // data_loader.cols * col
                y = shape[0] // data_loader.rows * row
                w = shape[1] // data_loader.cols * 2
                h = shape[0] // data_loader.cols * 2
            
            if data_input_flat[i] and labels[i]:
                gt_id = labels[i]                
                cv2.rectangle(overlay_box, (x,y), (x+w,y+h), gt_color[gt_id], -1)                
                gt_x = min(x, gt_x)
                gt_y = min(y, gt_y)
                gt_r = max(x+w, gt_r)
                gt_b = max(y+h, gt_b)
                    
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:                
                    cv2.rectangle(overlay_line, (x+bbox_pad,y+bbox_pad), \
                                  (x+bbox_pad+w,y+bbox_pad+h), inf_color[inf_id], max_len//768*2)                
                    inf_x = min(x, inf_x)
                    inf_y = min(y, inf_y)
                    inf_r = max(x+w, inf_r)
                    inf_b = max(y+h, inf_b)
                
            #text = data_loader.classes[gt_id] + '|' + data_loader.classes[inf_id]
            #cv2.putText(img, text, (x,y), font, font_size, ft_color)  
        
        cv2.rectangle(overlay_box, (gt_x,gt_y), (gt_r,gt_b), [180,180,215], -1)
        cv2.rectangle(overlay_line, (inf_x+bbox_pad,inf_y+bbox_pad), (inf_r+bbox_pad,inf_b+bbox_pad), [0,115,255], max_len//768*2)
        
        # legends
        w = shape[1] // data_loader.cols * 4
        h = shape[0] // data_loader.cols * 2
        for i in range(1, len(data_loader.classes)):
            row = i * 3
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            cv2.rectangle(img, (x,y), (x+w,y+h), gt_color[i], -1)
            cv2.putText(img, data_loader.classes[i], (x+w,y+h), font, 0.8, ft_color)  
            
            row = i * 3 + 1
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            cv2.rectangle(img, (x+bbox_pad,y+bbox_pad), \
                          (x+bbox_pad+w,y+bbox_pad+h), inf_color[i], max_len//384)        
        
        alpha = 0.4
        cv2.addWeighted(overlay_box, alpha, img, 1-alpha, 0, img)
        cv2.addWeighted(overlay_line, 1-alpha, img, 1, 0, img)
        cv2.imwrite('results/' + file_name[:-4]+'.png', img)        
        cv2.imshow("test", img)
        cv2.waitKey(0)


def save_json(path_json,data):
    with open(path_json, 'w') as outfile:
        json.dump(data, outfile)
    return

def load_json(name_json):
    with open(name_json) as f: 
        data_json = json.load(f) 
    return data_json
