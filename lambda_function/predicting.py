import tensorflow as tf
import numpy as np
import argparse
import timeit
from pprint import pprint

import pandas as pd


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model_cutie_aspp import CUTIERes as CUTIEv1
from collections import defaultdict
import utils

classes = ['DontCare', 'vendor_name', 'vendor_VAT', 'vendor_address', 'vendor_IBAN', 'client_name', 'client_VAT', 'client_address', 'date', 'invoice_ID', 'prod_ref', 'prod_concept', 'prod_lot', 'prod_qty', 'prod_uprice', 'prod_total', 'base_price', 'VAT', 'total']
num_classes = len(classes)
load_dictionary = True
update_dict = False


def predict_invoice(ckpt_path, data_json, args):
    
    dictionary, word_to_index, index_to_word = utils.load_dict(args['dict_path'],load_dictionary)
    num_words = 20000
    parser = argparse.ArgumentParser(description='CUTIE parameters')
    parser.add_argument('--embedding_size', type=int, default=128)
    params = parser.parse_args()

    network = CUTIEv1(num_words, num_classes, params)
    model_output = network.get_output('softmax')

    # load model
    tf.reset_default_graph()
    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            #ckpt_path = os.path.join(e_ckpt_path, save_prefix, ckpt_file)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt_path))
            ckpt_saver.restore(sess, ckpt_path)
            print('{} restored'.format(ckpt_path))
        except:
            raise Exception('Check your pretrained {:s}'.format(ckpt_path))

        docs = utils.load_data(data_json, update_dict=update_dict)

        if update_dict:
                num_words = len(dictionary)              
                utils.update_word_to_index(word_to_index, dictionary, index_to_word)
                print("updated word to index")
       
                # save dictionary/word_to_index/index_to_word as file
                # np.save(dict_path + '_dictionary.npy', dictionary)
                # np.save(dict_path + '_word_to_index.npy', word_to_index)
                # np.save(dict_path + '_index_to_word.npy', index_to_word)

                # sorted(self.dictionary.items(), key=lambda x:x[1], reverse=True)  
                # print("Diccionario guardado")
        
        data = utils.prepare_data(docs)

        feed_dict = {
                network.data_grid: data['grid_table'],
            }

        fetches = [model_output]

        print(data['file_name'][0])
        #print(data['grid_table'].shape, data['data_image'].shape, data['ps_1d_indices'].shape)

        timer_start = timeit.default_timer()
        [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
        timer_stop = timeit.default_timer()

        _, _, _, _, df_pred_ids = utils.make_results(np.array(data['grid_table']), model_output_val, data['bbox_mapids'], np.array(data['grid_word_ids']), index_to_word)

        data_labeled = utils.make_json_results(df_pred_ids, data_json)

    return data_labeled


def predict_invoice_v2(MODEL_GRAPH_DEF_PATH, data_json, args):
    
    tf.reset_default_graph()
    
    dictionary, word_to_index, index_to_word = utils.load_dict(args['dict_path'],load_dictionary)
    num_words = 20000
    parser = argparse.ArgumentParser(description='CUTIE parameters')
    parser.add_argument('--embedding_size', type=int, default=128)
    params = parser.parse_args()

    network = CUTIEv1(num_words, num_classes, params)
    model_output = network.get_output('softmax')

    # load model
    #tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer()) 
        model = tf.saved_model.loader.load(export_dir=MODEL_GRAPH_DEF_PATH, sess=sess, tags=[tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING]) #Note the SERVINGS tag is put as default.
 
        docs = utils.load_data(data_json, update_dict=update_dict)

        if update_dict:
                num_words = len(dictionary)
                utils.update_word_to_index(word_to_index, dictionary, index_to_word)
                print("updated word to index")

                # save dictionary/word_to_index/index_to_word as file
                # np.save(dict_path + '_dictionary.npy', dictionary)
                # np.save(dict_path + '_word_to_index.npy', word_to_index)
                # np.save(dict_path + '_index_to_word.npy', index_to_word)

                # sorted(self.dictionary.items(), key=lambda x:x[1], reverse=True)  
                # print("Diccionario guardado")
        
        data = utils.prepare_data(docs)

        feed_dict = {
                network.data_grid: data['grid_table'],
            }

        fetches = [model_output]

        print(data['file_name'][0])
        #print(data['grid_table'].shape, data['data_image'].shape, data['ps_1d_indices'].shape)

        timer_start = timeit.default_timer()
        [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
        timer_stop = timeit.default_timer()

        _, _, _, _, df_pred_ids = utils.make_results(np.array(data['grid_table']), model_output_val, data['bbox_mapids'], np.array(data['grid_word_ids']), index_to_word)

        data_labeled = utils.make_json_results(df_pred_ids, data_json)

    sess.close()
    return data_labeled

