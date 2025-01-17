import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN  # Import MTCNN for face detection

# Initialize the FaceNet model
embedder = FaceNet()

# Known images (use CNN for better accuracy)
known_images = ['/home/alain/Documents/images/smith.jpg','/home/alain/Documents/images/smith3.jpg','/home/alain/Documents/images/smith2.jpg','/home/alain/Documents/images/willSmith.jpg','/home/alain/Documents/images/willSmithFam.jpg']
known_embeddings = []
known_image_data = []

# Process known images to get embeddings
for image_path in known_images:
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load image from directory')
        continue

    known_image_data.append((image, image_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embeddings = embedder.embeddings([rgb_image])
    known_embeddings.append(embeddings[0])

# Load and process the test image
test_image = cv2.imread('/home/alain/Documents/images/willSmithFam2.jpg')
rgb_image_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Initialize MTCNN for face detection
detector = MTCNN()

# Detect faces in the test image
faces = detector.detect_faces(rgb_image_test)

# Process each detected face
for face in faces:
    x, y, w, h = face['box']  # Get the bounding box of the face
    face_roi = rgb_image_test[y:y+h, x:x+w]  # Crop the face region

    # Get the face embedding
    test_embedding = embedder.embeddings([face_roi])[0]

    match_found = False

    # Compare with known embeddings
    for known_embedding in known_embeddings:
        distance = np.linalg.norm(test_embedding - known_embedding)
        print(f'Distance: {distance}')

        if distance < 1:  # Adjust threshold as needed
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for match

            match_found = True
            break

    if not match_found:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for non-match

# Display known images
for i, (image, image_path) in enumerate(known_image_data):
    cv2.imshow(f'Known Image {i+1}', image)

# Display the test image with results
cv2.imshow('Test Image Results', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
