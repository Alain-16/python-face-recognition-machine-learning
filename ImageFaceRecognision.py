import dlib
import cv2
import numpy as np

# Load the pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('model_v1.dat')

# Load the images of known individuals
known_images = ['/home/alain/Documents/images/smith3.jpg','/home/alain/Documents/images/smith.jpg']
threshold = 1  # Set a reasonable threshold value

# Compute the face embeddings for known individuals
known_embeddings = []
known_images_data = []
for image_path in known_images:
    image = dlib.load_rgb_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    known_images_data.append((image, image_path))  # Keep the image data and path
    faces = face_detector(image)

    print(f"Detected {len(faces)} faces in {image_path}.")

    # Draw rectangles around all detected faces in the known image
    known_image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face in faces:
        shape = shape_predictor(image, face)  # Get the landmarks for the face
        face_chip = dlib.get_face_chip(image, shape)  # Crop and align the face
        embedding = face_recognizer.compute_face_descriptor(face_chip)
        known_embeddings.append(embedding)

        # Draw a rectangle around the face in the known image
        cv2.rectangle(known_image_cv2, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

# Load the test image and detect faces
test_image = dlib.load_rgb_image('/home/alain/Documents/images/willSmithFam.jpg')
faces = face_detector(test_image)

print(f"Detected {len(faces)} faces in test image.")

# Convert the test image to a format that OpenCV can display
test_image_cv2 = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Iterate over the detected faces in the test image
for face in faces:
    # Get the landmarks for the face in the test image
    shape = shape_predictor(test_image, face)

    # Align and crop the detected face using dlib.get_face_chip
    face_chip = dlib.get_face_chip(test_image, shape)

    # Compute the face embedding for the aligned and cropped face
    test_embedding = face_recognizer.compute_face_descriptor(face_chip)

    match_found = False  # Initialize match flag

    # Compare the test embedding with known embeddings
    for i, known_embedding in enumerate(known_embeddings):
        # Calculate Euclidean distance
        distance = np.linalg.norm(np.array(test_embedding) - np.array(known_embedding))
        print(f"Comparing with known face {i}, distance: {distance}")

        if distance < threshold:  # Replace `threshold` with an appropriate value
            print(f"Match found with person {i + 1}!")
            # Draw a green rectangle around the matching face in the test image
            cv2.rectangle(test_image_cv2, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            match_found = True
            break  # Exit the loop if a match is found

    if not match_found:
        # Draw a red rectangle around the non-matching face in the test image
        cv2.rectangle(test_image_cv2, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

# Display the known image with rectangles around faces
for i, (_, image_path) in enumerate(known_images_data):
    cv2.imshow(f'Known Image {i + 1}', cv2.cvtColor(dlib.load_rgb_image(image_path), cv2.COLOR_RGB2BGR))

# Display the test image with rectangles indicating matches and non-matches
cv2.imshow('Test Image Results', test_image_cv2)

cv2.waitKey(0)  # Wait for a key press to close the images
cv2.destroyAllWindows()  # Close all OpenCV windows
