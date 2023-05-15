import json

# pred_json = "predecido1.json"
# conf_json = "model_conf.json"
classes = ['unknown', 'vendor_name', 'vendor_VAT', 'vendor_address', 'vendor_IBAN', 'client_name', 'client_VAT', 'client_address', 'date', 'invoice_ID', 'prod_ref', 'prod_concept', 'prod_lot', 'prod_qty', 'prod_uprice', 'prod_total', 'base_price', 'VAT', 'total']

def load_json(data):
    with open(data) as f: 
        data_json = json.load(f) 
    return data_json

def save_json(path_json,data):
    with open(path_json, 'w') as outfile:
        json.dump(data, outfile)
    return

def get_clases(data_preds):
    clases = []
    for item in data_preds['fields']:
        # if item['field_name'] == 'vendor_name':
        #     vendor = item
        clases.append(item['field_name'])
    return clases

def get_class_value(data_preds, schema):
    for item in data_preds['fields']:
        if item['field_name'] == schema:
            # print(schema)
            # print(item)
            value = item['value_text']
    return value

def get_ids_preds(data_preds, schema):
    
    for item in data_preds['fields']:
        if item['field_name'] == schema:
            ids = item['value_id']

    preds = []
    bboxs = []
    for id in ids:
        for tbox in data_preds['text_boxes']:
            if tbox['id'] == id:
                preds.append(tbox['pred'])
                bboxs.append(tbox['bbox'])
    

    return preds, bboxs

    
def process_children(item, data_preds):
    schema = item['schema_id']
    #print(schema)

    if schema == "tax_details" or schema == "line_items":
        pass
        #return item
    else:
        #print(item['value'])
        #if item['value'] == "":
        if 'value' in item and item['value'] == "": #si sigue teniendo hijos
            if schema == "document_id":
                item['value'] = data_preds['global_attributes']['file_id']
            if schema in classes:
                value = get_class_value(data_preds, schema)
                item['value'] = value
                preds, bboxs = get_ids_preds(data_preds, schema)
                item['rir_confidence'] = preds
                item['bboxes'] = bboxs
                item['type'] = "string"

    for i, child in enumerate(item.get('children', [])):
        item["children"][i] = process_children(child, data_preds)

    return item

def build_results(pred_json, textractResponse, args):

    data_preds = pred_json #load_json(pred_json)
    print(type(data_preds))
    data_conf = load_json(args['model_json_config'])

    for i, item in enumerate(data_conf["content"]):
        # El padre siempre va a tener content y el schema debe ser procesado
        for j, child in enumerate(item.get('children', [])):
            #print(child)
            data_conf["content"][i]["children"][j] = process_children(child, data_preds)

    data_conf["TextractResults"] = textractResponse
    #print(data_conf)
    #save_json("prueba_json2.json",data_conf)
    return json.dumps(data_conf)

