import json
import os
from botocore.retries import bucket
import os
import piWorkflow
import boto3
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO, StringIO
import base64
from urllib.request import urlopen
from werkzeug.utils import secure_filename
import predicting
import build_results
import requests
from pdf2jpg import pdf2jpg

CONFIG_FILE = "variables.json"

def load_config():
    res = {}
    with open(CONFIG_FILE) as config_file:
        data = json.load(config_file)    
    for k, v in data.items():
        res[k] = os.environ.get(k, v)   
    # data['bucket'] = os.environ['bucket']
    # data['snsTopicArn'] = os.environ['snsTopicArn']
    # data['sqsQueueUrl'] = os.environ['sqsQueueUrl']
    # data['roleArn'] = os.environ['roleArn']
    # data['region'] = os.environ['region']

    return res

args = load_config()
print("ARGS: ", args)

MODEL_FILENAME = args["MODEL_FILENAME"]
MODEL_VAR1 = args["MODEL_VAR1"]
MODEL_VAR2 = args["MODEL_VAR2"]
MODEL_GRAPH_DEF_PATH = args["MODEL_GRAPH_DEF_PATH"]

UPLOAD_FOLDER = '/tmp/' 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
PDF_EXTENSION = {'pdf'}
IND_TO_CLASS = {0: 'unknown', 1: 'vendor_name', 2: 'vendor_VAT', 3: 'vendor_address', 4: 'vendor_IBAN', 5: 'client_name', 6: 'client_VAT', 7: 'client_address', 8: 'date', 9: 'invoice_ID', 10: 'prod_ref', 11: 'prod_concept', 12: 'prod_lot', 13: 'prod_qty', 14: 'prod_uprice', 15: 'prod_total', 16: 'base_price', 17: 'VAT', 18: 'total'}

bucket = args["bucket"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_pdf_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in PDF_EXTENSION

def upload_file(file_name, bucket, key, args):
    """
    Function to upload a file to an S3 bucket
    """
    s3_client = boto3.client('s3', region_name=args['region'])
    response = s3_client.upload_file(file_name, bucket, key)

    return response

def save_json(path_json,data):
    with open(path_json, 'w') as outfile:
        json.dump(data, outfile)
    return

def load_json(name_json):
    with open(name_json) as f: 
        data_json = json.load(f) 
    return data_json

def get_invoice_transcript(img_path, args):
    filename = img_path.split('/')[-1]
    if filename and allowed_file(filename):
        bbox_path_json = UPLOAD_FOLDER +'/'+filename.split('.')[0]+'.json'
        #bbox_path_json = filename.split('.')[0]+'.json'
        data, jsonParsed = piWorkflow.process_invoice_local(img_path, filename, IND_TO_CLASS, args)
        save_json(bbox_path_json, data) 
        #data_json = json.dumps(data)
    
        return load_json(bbox_path_json), jsonParsed
    return "Bad image path"

def get_predictions(img_path, args):
    print("Imagen recogida")
    data_json, jsonParsed = get_invoice_transcript(img_path, args)
    
    print("Transcripción realizada, prediciendo...")
    preds = predicting.predict_invoice_v2(MODEL_GRAPH_DEF_PATH, data_json, args)
    preds_serialized = json.dumps(preds)
    return preds_serialized, jsonParsed


def tag_invoice(event, context):
    
    message = json.loads(event['Records'][0]['Sns']['Message'])

    url = message['url'] 
    filename = url.split("/")[-1]
    print(url, filename, "print 1")
    if allowed_file(filename):
        im = Image.open(urlopen(url))
    else:
        if allowed_pdf_file(filename):
            with requests.get(url, allow_redirects =True) as r: #get the url
                print(url)
                open(UPLOAD_FOLDER+filename, 'wb').write(r.content)

            result = pdf2jpg.convert_pdf2jpg(UPLOAD_FOLDER+filename, UPLOAD_FOLDER, pages="ALL")
            result_path = result[0]['output_jpgfiles'][0]
            print(result_path)
            im = Image.open(result_path)
            filename = filename.split(".pdf")[0] + ".jpg"
        
        else:
            return ("Error, invalid type of data, must be png, jpeg, jpg or pdf") 
 
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    im.save(filepath)
    upload_file(filepath, bucket, filename, args)

    if filename and allowed_file(filename):
        preds, jsonParsed = get_predictions(filepath, args)
        preds = json.loads(preds)
        #print(preds)
        format_preds = build_results.build_results(preds, jsonParsed, args)
        format_preds = json.loads(format_preds)
        return format_preds 
   
    return ("Error, image is missing") 