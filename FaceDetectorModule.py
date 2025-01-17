import os.path

import cv2
import numpy as np
import dlib
from keras_facenet import FaceNet
from mtcnn import MTCNN


# Initialize the models
embedder = FaceNet()
detector = MTCNN()
dlib_face_rec_model = dlib.face_recognition_model_v1('model_v1.dat')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # For face alignment

VIDEO_PATH = '/home/alain/Documents/images/trialVD.mp4'  # Path to the video file
WANTED_PERSON_IMAGES = [
    '/home/alain/Documents/images/mugisha.jpg',
    '/home/alain/Documents/images/bella.jpeg'

]  # List of images of the wanted person
THRESHOLD = 0.6

def align_face(image,box,shape_predictor):

    x,y,w,h = box
    rect = dlib.rectangle(x,y,x+w,y+h)
    landmarks = shape_predictor(image,rect)
    aligned_face = dlib.get_face_chip(image,landmarks,size=150)
    return aligned_face

def compute_dlib_embedding(face,dlib_face_rec_model):
    return np.array(dlib_face_rec_model.compute_face_descriptor(face))


def process_known_images(image_paths, embedder, dlib_face_rec_model, shape_predictor):
    """Process and store embeddings of the wanted person."""
    known_embeddings = []



    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f'Failed to load image from {image_path}')
            continue

        person_name = os.path.splitext(os.path.basename(image_path))[0]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)

        for face in faces:
            aligned_face = align_face(rgb_image, face['box'], shape_predictor)

            # Compute embeddings using both FaceNet and Dlib
            cnn_embedding = embedder.embeddings([aligned_face])[0]
            dlib_embedding = compute_dlib_embedding(aligned_face, dlib_face_rec_model)

            known_embeddings.append((cnn_embedding, dlib_embedding,person_name))

    return known_embeddings

def is_match(test_embeddings,known_embeddings,threshold):

    for known_embedding in known_embeddings:
        cnn_dist=np.linalg.norm(test_embeddings[0]-known_embedding[0])
        dlib_dist=np.linalg.norm(test_embeddings[1]-known_embedding[1])

        if cnn_dist < threshold or dlib_dist < threshold:
            return known_embedding[2]
    return False


def process_video(video_path, known_embeddings, detector, embedder, dlib_face_rec_model, shape_predictor, threshold,frame_skip=50):
    """Process the video to detect and recognize the wanted person."""
    cap = cv2.VideoCapture(video_path)
    frame_count=0

    # Define the desired width and height for resizing the video window
    desired_width = 640
    desired_height = 480

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert the frame to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame using MTCNN
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            aligned_face = align_face(rgb_frame, face['box'], shape_predictor)

            # Compute embeddings using both FaceNet and Dlib
            cnn_embedding = embedder.embeddings([aligned_face])[0]
            dlib_embedding = compute_dlib_embedding(aligned_face, dlib_face_rec_model)

            # Check if the detected face matches the wanted person
            person_name = is_match((cnn_embedding, dlib_embedding), known_embeddings, threshold)
            x, y, w, h = face['box']
            if person_name:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)  # Green for match
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for non-match

        # Resize the frame to fit the desired window size
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        # Display the resized frame with annotations
        cv2.imshow('Video Frame', resized_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

known_embeddings = process_known_images(WANTED_PERSON_IMAGES,embedder,dlib_face_rec_model,shape_predictor)

process_video(VIDEO_PATH,known_embeddings,detector,embedder,dlib_face_rec_model,shape_predictor,THRESHOLD)
