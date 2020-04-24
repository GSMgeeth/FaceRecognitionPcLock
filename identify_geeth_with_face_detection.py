#  Libraries For Object Detection and Image Prediction
from imageai.Detection import ObjectDetection

#  Libraries For Face Recognition
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

#  Libraries to take the picture
import pygame
import pygame.camera

#  Libraries to Lock the Computer in an intruder detection
import os
import subprocess

#  Library take an interval between every Camera Capture Analyzis
import time

#  Logging to log certain situations
import logging

#  Get logger instance for the program
logging.basicConfig(level=logging.INFO)

#  Getting the current working directory
execution_path = os.getcwd()

#  Maximum Threshold to be look like geeth
MAX_THRESHOLD_FOR_GEETH = 0.5

#  Moderate Threshold to be look like geeth
MODERATE_THRESHOLD_FOR_GEETH = 0.7

# Path To Program
PROGRAM_PATH = "/home/umadhg1/Documents/Workspace/my_tsflow/machine_learning_workspace/Testing Space"

# Owner Name
GEETH = "geeth"

#  geeth's model scores from different looks of geeth
model_scores_of_geeth = []

#  Setting up a variable to detect objects using YOLOv3 model
personDetector = ObjectDetection()
personDetector.setModelTypeAsYOLOv3()
personDetector.setModelPath( os.path.join(execution_path , "yolo.h5"))
personDetector.loadModel()

#  Setting the object types to be identified in the object detection process
custom_objects = personDetector.CustomObjects(person=True)

#  Load the ResNet50 model without pre-trained celebrity faces on top -> include_top=False
vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

#  Setting up the Face detector from Multi Tasked Convoluted Neural Network constructor
detector = MTCNN()

#  Extract faces (currently one face) from a given image into required size.
#  If needs to get every face in the image, append every face_array into another array inside an for loop after detecting faces.
def extract_face(image_path, required_size=(224, 224)):
    image = plt.imread(image_path)

    faces = detector.detect_faces(image)

    #  If there are no faces in the image return an empty array
    if len(faces) == 0:
        return []

    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height

    face_boundary = image[y1:y2, x1:x2]

    try:
        face_image = Image.fromarray(face_boundary)
    except ValueError:
        logging.warning("tile cannot extend outside image - %s", image_path)
        return []
    
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)

    return face_array

#  Calculate scores/attribute vectors/exbeddings of faces
def get_model_scores(image_paths):
    faces = [extract_face(image_paths)]

    #  If no face is extracted from the image return an empty array
    if (len(faces[0]) == 0):
        return []

    samples = asarray(faces, 'float32')

    samples = preprocess_input(samples, version=2)

    return vggface_model.predict(samples)

#  Check whether it's a photo of geeth or not
def is_geeth(image_paths):
    model_scores = get_model_scores(image_paths)

    #  Compare extracted face scores with geeth's face scores
    for stranger_score in model_scores:
        for geeth_score in model_scores_of_geeth:
            if cosine(stranger_score, geeth_score) <= MAX_THRESHOLD_FOR_GEETH:
                # logging.info("MAX")
                return True
            elif cosine(stranger_score, geeth_score) <= MODERATE_THRESHOLD_FOR_GEETH:
                # logging.info("MODERATE")
                return True
    
    return False

#  get the model score of a face
def get_model_score_of(image):
    return get_model_scores(image)

#  Save geeth's model_scores to compare with pictures to be taken
for i in range(9):
    model_scores_of_geeth.append(get_model_score_of(''.join([PROGRAM_PATH, '/', GEETH, '/', GEETH, '-', str(i + 1), '.jpg'])))

while True:
    #  Taking an interval before the next image processing
    time.sleep(4)
    
    #  Taking the picture to be processed
    pygame.camera.init()

    cam = pygame.camera.Camera("/dev/video0",(640,480))

    cam.start()
    img = cam.get_image()
    cam.stop()

    pygame.image.save(img,"webcam_image.jpg")

    #  Detecting people in the above taken picture
    personDetections, object_path = personDetector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "webcam_image.jpg"), output_image_path=os.path.join(execution_path , "image_detected.jpg"), minimum_percentage_probability=30, extract_detected_objects=True)

    #  Lock the computer if there are no people in front of the computer
    if len(personDetections) == 0:
        logging.info("No Person")
        # continue
        subprocess.run(["gnome-screensaver-command", "-l"])
        # raise SystemExit

    #  If there are people in the taken picture, giving time to crop and save the detected persons
    time.sleep(1)

    #  See if Geeth is there using Face Detection
    safe = False;
    for path in object_path:
        if is_geeth(path):
            safe = True;
            break
    
    #  If 'geeth' category has no people, then lock the computer for safety
    if safe == False:
        logging.info("Not Safe")
        subprocess.run(["gnome-screensaver-command", "-l"])
        # raise SystemExit
