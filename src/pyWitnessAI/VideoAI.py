import os
import pandas as pd
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from importlib.resources import files



class FrameAnalyzerMTCNNIndependent:
    def __init__(self, name="mtcnn_old"):
        self.detector = MTCNN()
        self.name = name
        #  Store detected faces as well as coordinates for transfer
        self.detected_faces = []

    def analyze_frame(self, frame):
        self.detected_faces = []
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)

        #  Faces and coordinates transfer
        frame_results = {
            'coordinates': [face['box'] for face in faces],
            'images': [frame[face['box'][1]:face['box'][1] + face['box'][3],
                             face['box'][0]:face['box'][0] + face['box'][2]]
                       for face in faces]
        }
        self.detected_faces.append(frame_results)
        confidence = self.get_confidence(faces)
        average_confidence = np.mean(confidence) if confidence else 0  # Calculate the average confidence
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        coordinates = self.get_face_coordinates(faces)

        return {
            # f'{self.name}_face_count': face_count,
            f'face_count': face_count,
            f'face_area': face_area,
            f'confidence': confidence,
            f'average_confidence': average_confidence,  # Add average confidence to the output
            f'coordinates': coordinates
        }

    def get_confidence(self, faces):
        confidence = []

        for face in faces:
            confidence.append(face['confidence'])

        return confidence

    def get_face_area(self, faces):
        face_area_sum = sum(face['box'][2] * face['box'][3] for face in faces)
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []

        for face in faces:
            coordinates.append(face['box'])

        return coordinates


class FrameAnalyzerOpenCVIndependent:
    def __init__(self,
                 cascade_path=str(files("pyWitnessAI.OpenCV_Models").joinpath("haarcascade_frontalface_alt.xml")),
                 name='opencv_old'):
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        self.name = name

    def analyze_frame(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        face_coordinates = self.get_face_coordinates(faces)
        confidence = 1.0 if faces else 0
        average_confidence = np.mean(confidence) if confidence else 0

        return {
            f'face_count': face_count,
            f'face_area': face_area,
            f'coordinates': face_coordinates,
            f'confidence': confidence,
            f'average_confidence': average_confidence
        }

    def get_face_area(self, faces):
        face_area_sum = sum([w * h for (x, y, w, h) in faces])
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []
        for (x, y, w, h) in faces:
            coordinates.append([x, y, w, h])
        return coordinates


class FrameAnalyzerDeepface:
    def __init__(self, detector_backend='mtcnn'):
        self.detect_backend = detector_backend
        self.name = detector_backend
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        average_confidence = np.mean(confidences) if confidences else 0

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'average_confidence': average_confidence,
            'coordinates': coordinates
        }


class FrameAnalyzerMTCNN:
    def __init__(self, name='mtcnn', detector_backend='mtcnn'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class FrameAnalyzerOpenCV:
    def __init__(self, name='opencv', detector_backend='opencv'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class FrameAnalyzerFastMTCNN:
    def __init__(self, name='fastmtcnn', detector_backend='fastmtcnn'):
        self.detect_backend = detector_backend
        self.name = name
        self.detected_faces = []

    def analyze_frame(self, frame):
        try:
            faces = DeepFace.extract_faces(frame, detector_backend=self.detect_backend, enforce_detection=False)
        except ValueError:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        if not faces or faces is None:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'coordinates': []
            }

        valid_faces = []
        confidences = []
        coordinates = []
        face_area_sum = 0
        frame_height, frame_width = frame.shape[:2]

        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                         face['facial_area']['w'], face['facial_area']['h']

            # Ignore faces that cover the entire frame or have zero confidence
            if (w == frame_width and h == frame_height) or face['confidence'] == 0:
                continue

            valid_faces.append(face)
            face_area = w * h
            face_area_sum += face_area
            confidence = face['confidence']
            confidences.append(confidence)
            coordinates.append([x, y, w, h])

        if not valid_faces:
            return {
                'face_count': 0,
                'face_area': 0,
                'confidence': [],
                'average_confidence': 0,
                'coordinates': []
            }

        return {
            'face_count': len(valid_faces),
            'face_area': face_area_sum,
            'confidence': confidences,
            'coordinates': coordinates
        }


class SimilarityAnalyzer:
    def __init__(self, lineup_faces, detector=None,
                 calculate_method='euclidean', model_name='Facenet', name='similarity'):
        self.name = name
        self.lineup_faces = lineup_faces
        self.calculate_method = calculate_method
        self.model_name = model_name   # The model to get the embedding

        #  Face detected from FrameAnalyzer used
        self.detector = detector

    def analyze_frame(self, frame):
        #  Use pre-detected faces for analysis
        frame_results = []

        # Access the detected faces from the current frame
        detected_faces_images = self.detector.detected_faces[0]['images']

        # for detected_face_info in detected_faces_info:
        for detected_face in detected_faces_images:
            face_comparisons = []

            # detected_face_np = self.preprocess_image(detected_face)
            embedding_results_detected = self.get_embedding(detected_face)
            emb_detected = np.array(embedding_results_detected[0]['embedding'])

            for lineup_face in self.lineup_faces:
                # lineup_face_np = self.preprocess_image(lineup_face)
                embedding_results_lineup = self.get_embedding(lineup_face)
                emb_lineup = np.array(embedding_results_lineup[0]['embedding'])

                if self.calculate_method == 'euclidean':
                    similarity_score = self.calculate_similarity_euclidean(emb_detected, emb_lineup)
                    face_comparisons.append(similarity_score)
                else:
                    raise ValueError(f"Unsupported detector backend: {self.calculate_method}")

            frame_results.append(face_comparisons)

        return {
            'facenet_distance': frame_results
        }

    def get_embedding(self, face, model_name='Facenet'):
        model_name = self.model_name

        #  Generate embedding using FaceNet
        embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)
        return np.array(embedding)

    def calculate_similarity_euclidean(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    @staticmethod
    def calculate_similarity_cosine(self, emb1, emb2):
        #  return np.linalg.norm(emb1 - emb2) #  L2 norm
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        return similarity

    def preprocess_image(self, image_np):
        #  Check if the image is in PIL format, convert to numpy array if so
        # if isinstance(image_np, Image.Image):
        #     image_np = np.array(image_np)

        #  Ensure image has 3 color channels (RGB)
        if image_np.shape[2] == 4:  # If the image has 4 x`channels (RGBA)
            image_np = image_np[:, :, :3]  # Drop the alpha channel

        #  Resize the image to 160x160 pixels using OpenCV
        resized_image = cv.resize(image_np, (160, 160))

        # Convert color from RGB to BGR
        resized_image = resized_image[:, :, ::-1]
        return resized_image


class LineupLoader:
    def __init__(self, image_paths=None, directory_path=None, target_size=(160, 160), image_number=0):
        self.directory_path = directory_path
        self.target_size = target_size
        self.lineup_images = []
        # Add or remove file types as needed
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        self.image_paths = image_paths
        self.file_names = []

        if image_paths is not None:
            self.image_paths = image_paths
            if image_number == 0:
                self.number = len(self.image_paths)
            else:
                self.number = image_number
        elif directory_path is not None:
            self.directory_path = directory_path
            # self.image_paths = self._load_paths_from_directory()
        else:
            raise ValueError("Either image paths or directory path must be provided.")

    def is_image_file(self, filename):
        return any(filename.endswith(ext) for ext in self.image_extensions)

    def preprocess_image(self, image):
        image = cv.resize(image, self.target_size)
        return image

    def load_image(self):
        count = 0
        loaded_images = []

        #  Loading images when paths are specified
        if self.image_paths is not None:
            for path in self.image_paths:
                if count >= self.number:
                    break
                image = cv.imread(path)
                processed_image = self.preprocess_image(image)
                self.lineup_images.append(processed_image)
                count += 1
            return self.lineup_images

        #  Loading images from directory
        # Loading images from directory
        if self.directory_path is not None:
            # Retrieve and sort the list of filenames
            filenames = os.listdir(self.directory_path)
            self.file_names = filenames
            # print(self.file_names)
            # Sort filenames alphabetically, case-insensitive
            sorted_filenames = sorted(self.file_names, key=lambda x: x.lower())

            for filename in sorted_filenames:
                if self.is_image_file(filename):
                    full_path = os.path.join(self.directory_path, filename)
                    image = cv.imread(full_path)
                    if image is not None:
                        processed_image = self.preprocess_image(image)
                        self.lineup_images.append(processed_image)
            return self.lineup_images

    def compare_faces(self, target_faces, filler_faces, detector='opencv', model_name='Facenet512', calculate_method='euclidean'):
        #  Use pre-detected faces for analysis
        frame_results = []
        model_name = model_name

        for target_face in target_faces:
            face_comparisons = []

            embedding_results_target = self.get_embedding(target_face, detector, model_name)
            emb_target = np.array(embedding_results_target[0]['embedding'])

            for j, filler_face in enumerate(filler_faces):
                embedding_results_filler = self.get_embedding(filler_face, detector, model_name)
                emb_filler = np.array(embedding_results_filler[0]['embedding'])

                if calculate_method == 'euclidean':
                    similarity_score = self.calculate_similarity_euclidean(emb_target, emb_filler)
                    face_comparisons.append({f'similarity_{j}': similarity_score})
                else:
                    raise ValueError(f"Unsupported detector backend: {calculate_method}")

            frame_results.append(face_comparisons)

        return frame_results

    def get_embedding(self, face, detector, model_name):
        #  Generate embedding using FaceNet
        embedding = DeepFace.represent(face, model_name=model_name, detector_backend = detector, enforce_detection=False)
        return np.array(embedding)

    def calculate_similarity_euclidean(self, emb1, emb2):
        # Apply L2 normalization to both embeddings
        emb1_normalized = emb1 / np.linalg.norm(emb1)
        emb2_normalized = emb2 / np.linalg.norm(emb2)

        # Compute the Euclidean distance between normalized embeddings
        return np.linalg.norm(emb1_normalized - emb2_normalized)

    def save(self, data, directory='results', label='', column_name=None):
        if not os.path.exists(directory):
            os.makedirs(directory)

        transposed_data = {}

        # Extract data
        for i,m in zip(range(len(data[0])), self.file_names):  # Assuming all sublists have the same length
            transposed_data[f'similarity_{m}'] = [data[j][i][f'similarity_{i}'] for j in range(len(data))]

        # Create DataFrame from the dictionary
        df = pd.DataFrame(transposed_data)

        # Transpose the DataFrame to get the desired structure
        df_transposed = df.T

        # Rename the columns to 0, 1, 2, ...
        # df_transposed.columns = [str(i) for i in range(len(data))]
        df_transposed.columns = [str(i) for i in column_name]

        # Save to CSV
        csv_filename = f"{label}_similarity_scores.csv"

        # Save to CSV
        df_transposed.to_csv(os.path.join(directory, csv_filename), index=True)
        print(f'Data saved to {csv_filename}.')
