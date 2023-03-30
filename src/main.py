import numpy as np
import cv2
from pathlib import Path
from IPython.display import Image
import torch

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import date
from geopy.geocoders import Nominatim

# Use a service account.
cred = credentials.Certificate('sdk.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()
model = torch.hub.load('ultralytics/yolov5', 'custom', source="local", path="models.pt", force_reload=True)
from datetime import datetime

global nesne
global kenevir
global sise
global kalem
global total
kenevir = 0
sise=0
kalem=0
total = 0


def totalCount():
    total = kenevir + sise + kalem
    data = {
    u'Sonuc': str(total)
    }
    db.collection(u'data').document(u'Node1').set(data)


def kenevirDetected():
    kenevir =+ 1
    current_date_and_time = datetime.now()
    loc = Nominatim(user_agent="GetLoc")
 
    getLoc = loc.geocode("Adana")
 
    print(getLoc.address)
    data = {
    u'nesne': "Kenevir",
    u'koordinat': str(getLoc.address),
    u'tarih': str(current_date_and_time),
    u'yer': "Adana"
    }

    db.collection(u'data').document(u'Node2').set(data)

    
def insanDetected():
    current_date_and_time = datetime.now()
    loc = Nominatim(user_agent="GetLoc")
    sise=+1
# entering the location name
    getLoc = loc.geocode("Adana")
 
# printing address
    print(getLoc.address)
    data = {
    u'nesne': "İnsan",
    u'koordinat': str(getLoc.address),
    u'tarih': str(current_date_and_time),
    u'yer': "Adana"
    }
    db.collection(u'data').document(u'Node3').set(data)




def kalemDetected():
    today = date.today()
    loc = Nominatim(user_agent="GetLoc")
    kalem=+1
# entering the location name
    getLoc = loc.geocode("Adana")
 
# printing address
    print(getLoc.address)
    data = {
    u'nesne': "Kalem",
    u'koordinat': str(getLoc.address),
    u'tarih': str(today),
    u'yer': "Adana"
    }
    db.collection(u'data').document(u'Node2').set(data)



cap = cv2.VideoCapture("video.mp4")
while True:

    ret, image_np = cap.read()
    image_np = cv2.resize(image_np, (816,416))
    results = model(image_np)
    df_result = results.pandas().xyxy[0]
    dict_result = df_result.to_dict()
    scores = list(dict_result["confidence"].values())
    labels = list(dict_result["name"].values())
    print(labels)
    nesne = labels
    if "kenevir" in nesne:
        print("kenevir found")
        kenevirDetected()
    if "insan" in nesne:
        print("insan found")
        insanDetected()
    if "kalem" in nesne:
        print("query found")
        kalemDetected()
    totalCount()
    list_boxes = list()
    for dict_item in df_result.to_dict('records'):
        list_boxes.append(list(dict_item.values())[:4])
    count = 0
    
    for xmin, ymin, xmax, ymax in list_boxes:
        image_np = cv2.rectangle(image_np, pt1=(int(xmin),int(ymin)), pt2=(int(xmax),int(ymax)), \
                                 color=(255,0, 0), thickness=2)
        cv2.putText(image_np, f"{labels[count]}: {round(scores[count], 2)}", (int(xmin), int(ymin)-10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        count = count + 1
        
    cv2.imshow('Object Detector', image_np);
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break