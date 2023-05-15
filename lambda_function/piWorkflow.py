#!/usr/bin/env python
# coding: utf-8

import boto3
import sys
import time
import re
import io
from PIL import Image, ImageDraw, ImageFont
import json
from botocore.retries import bucket
import pandas as pd
#from PyPDF2 import PdfFileReader
from PIL import Image
from nltk import ngrams
import datefinder
from tqdm import tqdm
#from pdf2image import convert_from_path, convert_from_bytes
from io import BytesIO
import tempfile
import os 

CONFIG_FILE = "variables.json"

class ProcessType:
    DETECTION = 1
    ANALYSIS = 2


class DocumentProcessor:
    
    args = {}

    with open(CONFIG_FILE) as config_file:
        data = json.load(config_file)    
        for k, v in data.items():
            args[k] = os.environ.get(k, v)

    jobId = ''
    textract = boto3.client('textract', region_name=args['region'])
    #boto3.set_stream_logger('')
    sqs = boto3.client('sqs', region_name=args['region'])
    sns = boto3.client('sns', region_name=args['region'])
    
    # roleArn = ''   
    # bucket = ''
    document = ''

    roleArn = args['roleArn']
    bucket = args['bucket']
    snsTopicArn = args['snsTopicArn']
    sqsQueueUrl = args['sqsQueueUrl']

    processType = ''


    def __init__(self, role, bucket, document):    
        self.roleArn = role
        self.bucket = bucket
        self.document = document    

    def analyzeImage(self):
        # Get the document from S3
        s3_connection = boto3.resource('s3')

        print(self.bucket, self.document)
        s3_object = s3_connection.Object(self.bucket, self.document)
        s3_response = s3_object.get()

        stream = io.BytesIO(s3_response['Body'].read())
        image = Image.open(stream)

        # Analyze the document
        client = boto3.client('textract')

        # Alternatively, process using S3 object
        response = client.analyze_document(
            Document={'S3Object': {'Bucket': self.bucket, 'Name': self.document}},
            FeatureTypes=["TABLES", "FORMS"])
            #FeatureTypes=["LINES"])

        return response    
 
    def ProcessDocument(self,type):

        jobFound = False
        response = ''
        
        self.processType=type
        validType=False

        #Determine which type of processing to perform
        if self.processType==ProcessType.DETECTION:
            response = self.textract.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket': self.bucket, 'Name': self.document}},
                    NotificationChannel={'RoleArn': self.roleArn, 'SNSTopicArn': self.snsTopicArn})
            print('Processing type: Detection')
            validType=True        

        
        if self.processType==ProcessType.ANALYSIS:
            response = self.textract.start_document_analysis(DocumentLocation={'S3Object': {'Bucket': self.bucket, 'Name': self.document}},
                FeatureTypes=["TABLES", "FORMS"],
                NotificationChannel={'RoleArn': self.roleArn, 'SNSTopicArn': self.snsTopicArn})
            print('Processing type: Analysis')
            validType=True    

        if validType==False:
            print("Invalid processing type. Choose Detection or Analysis.")
            return

        print("DATA: ", self.bucket, self.snsTopicArn, self.sqsQueueUrl, "DENTRO CLASS")
        print('Start Job Id: ' + response['JobId'])
        print(response)
        dotLine=0
        while jobFound == False:
            sqsResponse = self.sqs.receive_message(QueueUrl=self.sqsQueueUrl, MessageAttributeNames=['ALL'],
                                          MaxNumberOfMessages=10)
            #print(sqsResponse)
            if sqsResponse:
                if 'Messages' not in sqsResponse:
                    if dotLine<40:
                        print('.', end='')
                        dotLine=dotLine+1
                        #print(dotLine)
                    else:
                        print()
                        dotLine=0    
                    sys.stdout.flush()
                    time.sleep(5)
                    continue

                for message in sqsResponse['Messages']:
                    notification = json.loads(message['Body'])
                    textMessage = json.loads(notification['Message'])
                    print(textMessage['JobId'])
                    print(textMessage['Status'])
                    if str(textMessage['JobId']) == response['JobId']:
                        print('Matching Job Found:' + textMessage['JobId'])
                        jobFound = True
                        response = self.GetResults(textMessage['JobId'])
                        self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                       ReceiptHandle=message['ReceiptHandle'])
                    else:
                        print("Job didn't match:" +
                              str(textMessage['JobId']) + ' : ' + str(response['JobId']))
                    # Delete the unknown message. Consider sending to dead letter queue
                    self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                   ReceiptHandle=message['ReceiptHandle'])

        print('Done!')
        return(response)


    def CreateTopicandQueue(self):
      
        millis = str(int(round(time.time() * 1000)))

        #Create SNS topic
        #snsTopicName="AmazonTextractTopic" + millis
        snsTopicName="CubicFort" + millis

        topicResponse=self.sns.create_topic(Name=snsTopicName)
        self.snsTopicArn = topicResponse['TopicArn']
        print("Name of SNSTopicARN = "+str(self.snsTopicArn))

        #create SQS queue
        sqsQueueName="AmazonTextractQueue" + millis
        self.sqs.create_queue(QueueName=sqsQueueName)
        self.sqsQueueUrl = self.sqs.get_queue_url(QueueName=sqsQueueName)['QueueUrl']
 
        attribs = self.sqs.get_queue_attributes(QueueUrl=self.sqsQueueUrl,
                                                    AttributeNames=['QueueArn'])['Attributes']
                                        
        sqsQueueArn = attribs['QueueArn']
        print("Name of SQSQueueURL = "+str(self.sqsQueueUrl))

        # Subscribe SQS queue to SNS topic
        self.sns.subscribe(
            TopicArn=self.snsTopicArn,
            Protocol='sqs',
            Endpoint=sqsQueueArn)

        #Authorize SNS to write SQS queue 
        policy = """{{
  "Version":"2012-10-17",
  "Statement":[
    {{
      "Sid":"MyPolicy",
      "Effect":"Allow",
      "Principal" : {{"AWS" : "*"}},
      "Action": "SQS:SendMessage", 
      "Resource": {},
      "Condition":{{
        "ArnEquals":{{
          "aws:SourceArn": "{}"
        }}
      }}
    }}
  ]
}}""".format('"' + sqsQueueArn + '"' , self.snsTopicArn)
        print(policy)
        response = self.sqs.set_queue_attributes(
            QueueUrl = self.sqsQueueUrl,
            Attributes = {
                'Policy' : policy
            })
    
    def DeleteTopicandQueue(self):
        self.sqs.delete_queue(QueueUrl=self.sqsQueueUrl)
        self.sns.delete_topic(TopicArn=self.snsTopicArn)

    #Display information about a block
    def DisplayBlockInfo(self,block):
        
        print ("Block Id: " + block['Id'])
        print ("Type: " + block['BlockType'])
        if 'EntityTypes' in block:
            print('EntityTypes: {}'.format(block['EntityTypes']))

        if 'Text' in block:
            print("Text: " + block['Text'])

        if block['BlockType'] != 'PAGE':
            print("Confidence: " + "{:.2f}".format(block['Confidence']) + "%")

        print('Page: {}'.format(block['Page']))

        if block['BlockType'] == 'CELL':
            print('Cell Information')
            print('\tColumn: {} '.format(block['ColumnIndex']))
            print('\tRow: {}'.format(block['RowIndex']))
            print('\tColumn span: {} '.format(block['ColumnSpan']))
            print('\tRow span: {}'.format(block['RowSpan']))

            if 'Relationships' in block:
                print('\tRelationships: {}'.format(block['Relationships']))
    
        print('Geometry')
        print('\tBounding Box: {}'.format(block['Geometry']['BoundingBox']))
        print('\tPolygon: {}'.format(block['Geometry']['Polygon']))
        
        if block['BlockType'] == 'SELECTION_ELEMENT':
            print('    Selection element detected: ', end='')
            if block['SelectionStatus'] =='SELECTED':
                print('Selected')
            else:
                print('Not selected')  


    def GetResults(self, jobId):
        maxResults = 1000
        paginationToken = None
        finished = False

        while finished == False:

            response=None

            if self.processType==ProcessType.ANALYSIS:
                if paginationToken==None:
                    response = self.textract.get_document_analysis(JobId=jobId,
                        MaxResults=maxResults)
                else: 
                    response = self.textract.get_document_analysis(JobId=jobId,
                        MaxResults=maxResults,
                        NextToken=paginationToken)                           

            if self.processType==ProcessType.DETECTION:
                if paginationToken==None:
                    response = self.textract.get_document_text_detection(JobId=jobId,
                        MaxResults=maxResults)
                else: 
                    response = self.textract.get_document_text_detection(JobId=jobId,
                        MaxResults=maxResults,
                        NextToken=paginationToken)   

            if 'Blocks' in response:
                #blocks=response['Blocks']
                blocks = response
                #print ('Detected Document Text')
                #print ('Pages: {}'.format(response['DocumentMetadata']['Pages']))
            else:
                print(str(response))
                blocks = ''
        
            # Display block information
            # for block in blocks:
            #        self.DisplayBlockInfo(block)
            #        print()
            #        print()

            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True
        return blocks 

    def GetResultsDocumentAnalysis(self, jobId):
        maxResults = 1000
        paginationToken = None
        finished = False

        while finished == False:

            response=None
            if paginationToken==None:
                response = self.textract.get_document_analysis(JobId=jobId,
                                            MaxResults=maxResults)
            else: 
                response = self.textract.get_document_analysis(JobId=jobId,
                                            MaxResults=maxResults,
                                            NextToken=paginationToken)  
            

            #Get the text blocks
            blocks=response['Blocks']
            print(str(blocks))
            print ('Analyzed Document Text')
            print ('Pages: {}'.format(response['DocumentMetadata']['Pages']))
            # Display block information
            
            for block in blocks:
            #        self.DisplayBlockInfo(block)
            #        print()
            #        print()

                    if 'NextToken' in response:
                        paginationToken = response['NextToken']
                    else:
                        finished = True

def checkext(fname):   
    if re.search('\.jpg$',fname,flags=re.IGNORECASE):
        return('jpg')
    if re.search('\.jpeg$',fname,flags=re.IGNORECASE):
        return('jpg')
    if re.search('\.pdf$',fname,flags=re.IGNORECASE):
        return('pdf')
    if re.search('\.png$',fname,flags=re.IGNORECASE):
        return('png')
    return('skip')

def returnOCRTextract(document, args):
    
    #roleArn = 'arn:aws:iam::421351409481:role/TextractRole'
    #roleArn = 'arn:aws:iam::765372348836:role/service-role/cubic-lamba'
    roleArn = args['roleArn']
    bucket = args['bucket']
    print(roleArn, bucket, "returnOCR")
    #bucket = 'lambda-cubic-bucket1' #'buckettrainsciling2'

    extension = checkext(document)
    analyzer=DocumentProcessor(roleArn, bucket,document)
    
    #try:
    if extension == 'pdf':
        analyzer.CreateTopicandQueue()
        json = analyzer.ProcessDocument(ProcessType.ANALYSIS)
        analyzer.DeleteTopicandQueue()
        size = returnSizePDF(bucket, document)
        json[0]['sizeDoc'] = [str(size[0]),str(size[2]),str(size[1]),str(size[3])]
            
        return json
    elif extension in ['jpg','png']:
        #jsonPkt = analyzer.analyzeImage()
        jsonPkt = analyzer.ProcessDocument(ProcessType.DETECTION)
        size = returnSizeImage(bucket, document)
        time.sleep(5)
        jsonPkt['Blocks'][0]['sizeDoc'] = [0,str(size[0]),0,str(size[1])]
        #jsonPkt['Blocks'][0]['sizeDoc'] = [0,size[0],0,size[1]]
        return jsonPkt
    else:
        return False
    #except:
    #    return False
        
def returnSizePDF(bucket, document):

    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, document)
    fs = obj.get()['Body'].read()
    pdfFile = PdfFileReader(BytesIO(fs))
    size = pdfFile.getPage(0).mediaBox
    return size

def returnSizePDF(bucket, document):

    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, document)
    fs = obj.get()['Body'].read()
    pdfFile = PdfFileReader(BytesIO(fs))

    try:
        pdfFile.decrypt('')
        size = pdfFile.getPage(0).mediaBox
    except:
        size = pdfFile.getPage(0).mediaBox
    
    return size

def returnSizeImage(bucket, document):
    
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, document)
    fs = obj.get()['Body'].read()
    im=Image.open(BytesIO(fs))
    print(str(im.size))
    return im.size

def preprocess(json_dict):
    data = []
    df = pd.DataFrame(columns = ['words', 'coords', 'files', 'labels']) 
    j = 0
    if 'Blocks' in json_dict:
        blocks = json_dict['Blocks']
    else:
        blocks = json_dict        
    output_dict_page = [x for x in blocks if x['BlockType'] == 'PAGE']
    pageWidth = float(output_dict_page[0]['sizeDoc'][1])
    pageHeight = float(output_dict_page[0]['sizeDoc'][3])

    output_dict_line = [x for x in blocks if x['BlockType'] == 'LINE']
    for block in output_dict_line:

        words = block['Text']
        xmin = pageWidth * block['Geometry']['BoundingBox']['Left']
        ymin = pageHeight * block['Geometry']['BoundingBox']['Top']
        xmax = xmin + pageWidth * block['Geometry']['BoundingBox']['Width']
        ymax = ymin + pageHeight * block['Geometry']['BoundingBox']['Height']
        coords = [int(xmin), int(ymin), int(xmax), int(ymax)]
        files = str(j)
        labels = checkIfPattern(words)
        df.loc[j] = [words, coords, files, labels]
        j+=1
    
    return df

def checkIfPattern(string):
    pattern = re.compile("[A-Z][0-9]{7}(?:[0-9]|[A-Z])")
    patternIBAN = re.compile("^([A-Z]{2}[ \-]?[0-9]{2})(?=(?:[ \-]?[A-Z0-9]){9,30}$)((?:[ \-]?[A-Z0-9]{3,5}){2,7})([ \-]?[A-Z0-9]{1,3})?$")
    s = pattern.match(string)
    iban = patternIBAN.match(string)
    if s:
        return 1
    elif iban:
        return 2
    else:
        return 0
    
def ngrammer(tokens, length=2):
    for n in range(1,min(len(tokens)+1, length+1)):
        #print(n)
        for gram in ngrams(tokens,n):
            yield gram
            
def get_features(df):
    
    files = {}
    
    print('\nExtracting features...\n')
    
    for i, row in df.iterrows():
        
        if row['files'] not in files:
            files[row['files']] = {'lines': {'words':[], 'labels':[], 'ymin':[], 'ymax':[]},
                                 'xmin':sys.maxsize, 'ymin':sys.maxsize, 'xmax':0, 'ymax':0}
        
        tokens = row['words'].strip().split(' ')
        char_length = (row['coords'][2] - row['coords'][0]) / len(row['words'].strip())
        token_coords = [{'xmin': row['coords'][0],
                         'xmax': row['coords'][0] + (char_length * len(tokens[0]))}]
        for idx in range(1, len(tokens)):
            token_coords.append({'xmin': token_coords[-1]['xmax'] + char_length,
                                 'xmax': token_coords[-1]['xmax'] + (char_length * (len(tokens[idx])+1))})
            
        files[row['files']]['lines']['words'].append({'tokens': tokens, 'coords': token_coords})
        files[row['files']]['lines']['labels'].append(row['labels'])
        files[row['files']]['lines']['ymin'].append(row['coords'][1])
        files[row['files']]['lines']['ymax'].append(row['coords'][3])
        files[row['files']]['xmin'] = min(files[row['files']]['xmin'], row['coords'][0])
        files[row['files']]['ymin'] = min(files[row['files']]['ymin'], row['coords'][1])
        files[row['files']]['xmax'] = max(files[row['files']]['xmax'], row['coords'][2])
        files[row['files']]['ymax'] = max(files[row['files']]['ymax'], row['coords'][3])
        
    del df
    
    grams = {'raw_text': [],
             'processed_text': [],
             'text_pattern': [],
             'length': [],
             'line_size': [],
             'position_on_line': [],
             'has_digits': [],
             'bottom_margin': [],
             'top_margin': [],
             'left_margin': [],
             'right_margin': [],
             'page_width': [],
             'page_height': [],
             'parses_as_amount': [],
             'parses_as_date': [],
             'parses_as_number': [],
             'label': [],
             'closest_ngrams': []
             }

    label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    
    with tqdm(total=len(files)) as pbar:
        for key, value in files.items():
            num_ngrams = len(grams['raw_text'])
            page_height = value['ymax'] - value['ymin']
            page_width = value['xmax'] - value['xmin']
            for i in range(len(value['lines']['words'])):
                tokens = value['lines']['words'][i]['tokens']
                token_coords = value['lines']['words'][i]['coords']
                for ngram in ngrammer(tokens):
                    grams['parses_as_date'].append(0.0)
                    grams['parses_as_amount'].append(0.0)
                    grams['parses_as_number'].append(0.0)
                    processed_text = []
                    for word in ngram:
                        try:
                            finderDate = bool(list(datefinder.find_dates(word)))
                        except:
                            finderDate = False
                        if finderDate:
                            processed_text.append('date')
                            grams['parses_as_date'][-1] = 1.0
                        elif bool(re.search(r'\d\.\d', word)) or '$' in word:
                            processed_text.append('amount')
                            grams['parses_as_amount'][-1] = 1.0
                        elif word.isnumeric():
                            processed_text.append('number')
                            grams['parses_as_number'][-1] = 1.0
                        else:
                            processed_text.append(word.lower())
                    raw_text = ' '.join(ngram)
                    grams['raw_text'].append(raw_text)
                    grams['processed_text'].append(' '.join(processed_text))
                    grams['text_pattern'].append(re.sub('[a-z]', 'x', re.sub('[A-Z]', 'X', re.sub('\d', '0', re.sub(
                        '[^a-zA-Z\d\ ]', '?', raw_text)))))
                    grams['length'].append(len(' '.join(ngram)))
                    grams['line_size'].append(len(tokens))
                    grams['position_on_line'].append(tokens.index(ngram[0])/len(tokens))
                    grams['has_digits'].append(1.0 if bool(re.search(r'\d', raw_text)) else 0.0)
                    grams['left_margin'].append((token_coords[tokens.index(ngram[0])]['xmin'] - value['xmin']) / page_width)
                    grams['top_margin'].append((value['lines']['ymin'][i] - value['ymin']) / page_height)
                    grams['right_margin'].append((token_coords[tokens.index(ngram[-1])]['xmax'] - value['xmin']) / page_width)
                    grams['bottom_margin'].append((value['lines']['ymax'][i] - value['ymin']) / page_height)
                    grams['page_width'].append(page_width)
                    grams['page_height'].append(page_height)
                    grams['label'].append(label_dict[value['lines']['labels'][i]])

            for i in range(num_ngrams, len(grams['raw_text'])):
                grams['closest_ngrams'].append([-1] * 4)
                distance = [sys.maxsize] * 6
                for j in range(num_ngrams, len(grams['raw_text'])):
                    d = [grams['top_margin'][i] - grams['bottom_margin'][j],
                         grams['top_margin'][j] - grams['bottom_margin'][i],
                         grams['left_margin'][i] - grams['right_margin'][j],
                         grams['left_margin'][j] - grams['right_margin'][i],
                         abs(grams['left_margin'][i] - grams['left_margin'][j])]
                    if i == j:
                        continue
                    # If in the same line, check for closest ngram to left and right
                    if d[0] == d[1]:
                        if distance[2] > d[2] > 0:
                            distance[2] = d[2]
                            grams['closest_ngrams'][i][2] = j
                        if distance[3] > d[3] > 0:
                            distance[3] = d[3]
                            grams['closest_ngrams'][i][3] = j
                    # If this ngram is above current ngram
                    elif distance[0] > d[0] >= 0 and distance[4] > d[4]:
                        distance[0] = d[0]
                        distance[4] = d[4]
                        grams['closest_ngrams'][i][0] = j
                    # If this ngram is below current ngram
                    elif distance[1] > d[1] >= 0 and distance[5] > d[4]:
                        distance[1] = d[1]
                        distance[5] = d[4]
                        grams['closest_ngrams'][i][1] = j
            pbar.update(1)
            
           
    return pd.DataFrame(data=grams)

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def writeBB(document, where, text):

    bucket = 'lambda-cubic-bucket1' #'buckettrainsciling2' 
    bucketTemp = 'stelneronetemp'
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, document)
    fs = obj.get()['Body'].read()

    # CV2
    img = convert_from_bytes(fs)[0]
    converted = image_to_byte_array(img)
    im = Image.open(BytesIO(converted))
    rows = where.shape[0]

    draw = ImageDraw.Draw(im)
    sizeX = im.size[0]
    sizeY = im.size[1]

    for x in range(0, rows):

        left = sizeX * where[x][0]#left
        top = sizeY * where[x][1]#top
        width = left + sizeX * where[x][2]#width
        height = top + sizeY * where[x][3]#height

        draw.rectangle(((left, top),(width, height)), outline = 'red')
        draw.text((left, top+1), text[x], fill="blue")

    in_mem_file = io.BytesIO()
    im.save(in_mem_file, format='JPEG')
    data = in_mem_file.getvalue()

    key = 'tmp_RES/' + next(tempfile._get_candidate_names()) + '.jpg'
    file = s3.Object(bucketTemp, key).put(Body=data)

    # Get the service client with sigv4 configured
    s3 = boto3.client('s3')
    url = s3.generate_presigned_url(
    ClientMethod='get_object',
    Params={
        'Bucket': bucketTemp,
        'Key': key
        },
    ExpiresIn=300
    )
    return url

def returnResult(jsonRet):

    print(jsonRet)
    return None

def process_invoice(document):#i):
    
    # We are going to pass files to Textract by getting them out of Excel spreadsheet
    print('==========> Processing')
    print(str(document))

    # This returns the JSON object from Textract
    jsonParsed = returnOCRTextract(str(document))
    # In this call we generate the dataframe with data about the file

    jsonRet = {}
    jsonRet["global_attributes"] = {"file_id":  str(document)}
    jsonRet["fields"] = []
    jsonRet["text_boxes"] = []

    idN = 1
    for block in jsonParsed["Blocks"]:
        if block["BlockType"] == "PAGE":
            widthDoc = int(block["sizeDoc"][1])
            heightDoc = int(block["sizeDoc"][3])
        if block["BlockType"] == "WORD":
            text = block["Text"]
            x_min = widthDoc * block["Geometry"]["BoundingBox"]["Left"]
            x_max = x_min + widthDoc * block["Geometry"]["BoundingBox"]["Width"]
            y_min = heightDoc * block["Geometry"]["BoundingBox"]["Top"]
            y_max = y_min + heightDoc * block["Geometry"]["BoundingBox"]["Height"]
            jsonRet["text_boxes"].append({"id": idN, "bbox": [x_min, y_min, x_max, y_max], "text" : text})
            idN += 1

    returnResult(jsonRet)

def upload_file(file_name, bucket, key, args):
    """
    Function to upload a file to an S3 bucket
    """
    s3_client = boto3.client('s3', region_name=args['region'])
    response = s3_client.upload_file(file_name, bucket, key)

    return response

def process_invoice_local(document, filename, ind_to_class, args):  # i):

    # We are going to pass files to Textract by getting them out of Excel spreadsheet
    print('==========> Processing')

    # Let's upload the invoice to S3
    #bucket = 'lambda-cubic-bucket1' #'buckettrainsciling2' #buckettraini
    upload_file(document, args['bucket'], filename, args)

    # This returns the JSON object from Textract
    jsonParsed = returnOCRTextract(str(filename), args)
    # In this call we generate the dataframe with data about the file

    jsonRet = {}
    #jsonRet["global_attributes"] = {"file_id": str(document)}
    jsonRet["global_attributes"] = {"file_id": str(filename)}
    jsonRet["fields"] = []
    jsonRet["text_boxes"] = []

    key_unk = None
    for key, value in ind_to_class.items():
        jsonRet["fields"].append({"field_name": value, "key_id": [], "key_text":[], "value_id":[]})
        if value == 'unknown':
            key_unk = key

    idN = 1
    for block in jsonParsed["Blocks"]:
        try:
            if block["BlockType"] == "PAGE":
                widthDoc = int(block["sizeDoc"][1])
                heightDoc = int(block["sizeDoc"][3])

            if block["BlockType"] == "WORD":
                text = block["Text"]
                x_min = widthDoc * block["Geometry"]["BoundingBox"]["Left"]
                x_max = x_min + widthDoc * block["Geometry"]["BoundingBox"]["Width"]
                y_min = heightDoc * block["Geometry"]["BoundingBox"]["Top"]
                y_max = y_min + heightDoc * block["Geometry"]["BoundingBox"]["Height"]
                jsonRet["text_boxes"].append({"id": idN, "bbox": [x_min, y_min, x_max, y_max], "text": text})
                jsonRet["fields"][key_unk]["value_id"].append(idN)
                idN += 1
        except:
            print("BAD IMAGE: ", filename)
            with open('bad_images.txt', 'a') as f:
                f.write(str(filename)+'\n')

    return jsonRet, jsonParsed