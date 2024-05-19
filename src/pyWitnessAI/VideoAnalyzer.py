import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from .Constants import legend_colors, line_styles
from keras.models import load_model
from importlib.resources import files
from .DataFlattener import *
from PIL import Image
import heapq
from deepface import DeepFace
from retinaface import RetinaFace
import dlib
from deepface.commons import functions
# You should also load the path of cascade, similarity_model, lineup_images before using the analyzer


class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_count = []
        self.frame_width = None
        self.frame_height = None
        self.frame_area = None
        self.average_pixel_values = []
        self.average_value = 0  # This is for the mean of the average pixel values of the whole video

        self.frame_processor = {}
        self.frame_analyzer = {}
        self.frame_analyzer_output = {}
        self.frame_analyzed = 0
        self.frame_total = 0

        self.top_frames = None  # An attribute to get the best quality frame

    def add_analyzer(self, analyzer):
        #  Add an external frame analyzer
        self.frame_analyzer[analyzer.name] = analyzer
        self.frame_analyzer_output[analyzer.name] = []

    def add_processor(self, processor):
        self.frame_processor[processor.name] = processor

    def release_resources(self):
        self.cap.release()
        for processor in self.frame_processor.values():
            if hasattr(processor, 'release'):
                processor.release()

    def get_frame_info(self):
        #  Retrieve frame information
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_area = self.frame_width * self.frame_height

    def process_video(self, frame_start=0, frame_end=1000000000):
        #  Process the video frame between frame_start and frame_end
        frame_analyzed = 0

        for frame_count in range(frame_start, frame_end + 1):
            ret, frame = self.cap.read()

            if not ret:
                break

            if frame_count == frame_start:
                self.frame_total = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
                self.get_frame_info()

            self.frame_count.append(frame_count)
            average_pixel_value = int(frame.mean())
            self.average_pixel_values.append(average_pixel_value)

            for k in self.frame_processor:
                frame = self.frame_processor[k].process_frame(frame)

            for k in self.frame_analyzer:
                self.frame_analyzer_output[k].append(self.frame_analyzer[k].analyze_frame(frame))

            frame_analyzed += 1

        self.average_value = np.mean(self.average_pixel_values)
        self.frame_analyzed = frame_analyzed
        self.release_resources()
        cv.destroyAllWindows()

    def get_analysis_info(self):
        #  Get the number of analyzed frame and total frames
        return {
            'frame_analyzed': self.frame_analyzed,
            'frame_total': self.frame_total
        }

    def run(self, frame_start=0, frame_end=100000):
        self.process_video(frame_start, frame_end)

    def find_probe_frames(self, top_n=1, log_file='probe_frames_log.txt'):
        if 'mtcnn' not in self.frame_analyzer_output:
            print("MTCNN analyzer is not added.")
            return []

        frames_metric = []
        for i, frame_data in enumerate(self.frame_analyzer_output['mtcnn']):
            face_area = frame_data.get('face_area', 0)
            average_confidence = frame_data.get('average_confidence', 0)
            metric = face_area * average_confidence  # Combined metric
            frames_metric.append((metric, self.frame_count[i], face_area, average_confidence))

        # Get the top N frames with the highest average confidence
        top_frames = heapq.nlargest(top_n, frames_metric, key=lambda x: x[0])

        self.top_frames = top_frames
        with open(log_file, 'w') as f:
            for metric, frame_num, face_area, avg_conf in top_frames:
                log_message = (f"Probe frame at frame number: {frame_num} with metric: {metric} "
                               f"(face_area: {face_area}, avg_confidence: {avg_conf})\n")
                print(log_message.strip())
                f.write(log_message)

        return top_frames

    def print_probe_frames(self, top_frames):
        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        if self.top_frames is not None:
            top_frames = top_frames
        else:
            top_frames = top_frames

        for i, (_, frame_number) in enumerate(top_frames):
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                cv.imshow(f"Probe Frame {i+1}", frame)
                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                print(f"Failed to retrieve frame at frame number: {frame_number}")

    def save_probe_frames(self, top_frames, save_directory='probe_frames'):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        for i, (_, frame_number, _, _) in enumerate(top_frames):
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                save_path = os.path.join(save_directory, f'probe_frame_{i+1}.jpg')
                cv.imwrite(save_path, frame)
                print(f"Probe frame {i+1} saved at {save_path}")
            else:
                print(f"Failed to retrieve frame at frame number: {frame_number}")

    def print_frame(self, frame_number, window_name="Frame"):
        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            cv.imshow(window_name, frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print(f"Failed to retrieve frame at frame number: {frame_number}")

    def plot_face_counts(self):
        #  Plots the number of faces against frame numbers
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity":
                continue
            # print(k)  # mtcnn, opencv, similarity
            # print(output)  #  Their output
            face_counts = []
            for data in output:
                #  print("Data:", data)
                if 'face_count' in data:
                    face_counts.append(data['face_count'])
                else:
                    face_counts.append(0)
            plt.plot(self.frame_count, face_counts, label=k, linestyle=line_styles[k], color=legend_colors[k])

        plt.xlabel('Frame')
        plt.ylim(0, 5)
        plt.ylabel('Number of Faces')
        plt.title('Number of Faces vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_face_areas(self):
        #  Plots the face area recognized by the classifiers against frame numbers
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity":
                continue
            face_areas = []
            for data in output:
                if 'face_area' in data:
                    face_areas.append(data['face_area'] / self.frame_area)
                else:
                    face_areas.append(0)
            plt.plot(self.frame_count, face_areas, label=k, linestyle=line_styles[k], color=legend_colors[k])

        plt.xlabel('Frame')
        plt.ylim(0, 0.5)
        plt.ylabel('Face Area Ratio')
        plt.title('Face Area Ratio vs Frame Number')
        plt.legend()
        plt.grid(True)

    def plot_average_pixel_values(self):
        #  Plot the average pixel values of the video
        plt.plot(self.frame_count, self.average_pixel_values, color=legend_colors['general'])
        plt.axhline(y=self.average_value, color=legend_colors['mean'], linestyle='--', label='Average value')
        plt.xlabel('Frame')
        plt.ylim(self.average_value-50, self.average_value+50)
        plt.ylabel('Average pixel value')
        plt.title('Pixel Intensity Trend across the Video')
        plt.legend()
        plt.grid(True)

    def plot_mtcnn_confidence_histogram(self):
        #  Plot the confidence histogram of mtcnn
        if 'mtcnn' in self.frame_analyzer_output:
            confidences = []
            for data in self.frame_analyzer_output['mtcnn']:
                if 'confidence' in data:
                    confidences.append(data['confidence'])

            flattened_confidences = []
            for sublist in confidences:
                for item in sublist:
                    flattened_confidences.append(item)

            plt.hist(flattened_confidences, bins=30, edgecolor='k')
            plt.xlabel('confidence')
            plt.ylabel('Frequency')
            plt.title('MTCNN Confidence Histogram')
            plt.grid(True)

        else:
            print('MTCNN data are not found in the output.')

    def save_data(self, directory='results', prefix='analyzed'):
        #  Save the analyzed data results to .csv files.
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {
            'frame': self.frame_count,
            'avg_pixel_value': self.average_pixel_values
        }

        for analyzer_name, results_list in self.frame_analyzer_output.items():
            # if all entries are dictionaries
            if all(isinstance(entry, dict) for entry in results_list):
                for key in results_list[0].keys():
                    data[f'{analyzer_name}_{key}'] = [entry[key] for entry in results_list]
                    #print(1)
            else:
                # if the results list contains non-dictionary entries (like lists)
                # If the entry is a list, save its length (For simplicity).
                data[f'{analyzer_name}_length'] = [len(entry) if isinstance(entry, list) else None for entry in
                                                   results_list]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)

    def save_data_flattened(self, directory='results', prefix='analyzed_flattened'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        #  Initialize data structures for flattened data
        flattened_data = []
        #  Iterate through each frame's analyzed output
        for i, frame in enumerate(self.frame_count):
            #  Initialize a dictionary for each frame
            frame_data = {'frame': frame}

            #  Flatten and merge all analyzer data into frame_data
            for analyzer_name, results_list in self.frame_analyzer_output.items():
                result = results_list[i]  # Get the result for the current frame
                flat_result = flatten_data(result)  # Flatten the result
                flat_keys = flatten_keys(result)  # Flatten the keys

                # Pair flattened keys and values, then merge into frame_data
                for key, value in zip(flat_keys, flat_result):
                    frame_data[f"{analyzer_name}_{key}"] = value

            #  Add the average pixel value for the frame
            frame_data['avg_pixel_value'] = self.average_pixel_values[i]

            #  Append the frame data to the flattened data list
            flattened_data.append(frame_data)

        #  Create a DataFrame from the flattened data
        df = pd.DataFrame(flattened_data)

        #  Define a preferred column order
        preferred_order = ['frame', 'avg_pixel_value']
        analyzer_keys = set()

        #  Collect keys for each analyzer type, assuming they start with the analyzer's name
        for column in df.columns:
            if column.startswith(tuple(self.frame_analyzer.keys())):
                analyzer_keys.add(column.split('_')[0])  # Get the analyzer name prefix

        #  Add analyzer data to preferred order, grouped by analyzer
        for analyzer in analyzer_keys:
            preferred_order.extend([col for col in df.columns if col.startswith(analyzer)])

        #  Ensure all columns are included by adding any remaining columns at the end
        remaining_columns = [col for col in df.columns if col not in preferred_order]
        preferred_order.extend(remaining_columns)
        # preferred_order[2:] = preferred_order[2:][::-1]  # Reverse the results of analyzers

        #  Reorder the DataFrame according to the preferred order
        df = df[preferred_order]

        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)


class FrameProcessorCropper:
    def __init__(self, x1, x2, y1, y2, name='cropper'):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.name = name

    def process_frame(self, frame):
        #  Crop the frame using the provided coordinates
        cropped_frame = frame[self.y1:self.y2, self.x1:self.x2]

        return cropped_frame


class FrameProcessorMonochrome:
    def __init__(self, name='monochrome'):
        self.name = name

    def process_frame(self, frame):
        #  Convert frame to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #  Convert the grayscale frame back to BGR for video writer and other stuffs
        bgr_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

        return bgr_frame


class FrameProcessorNormalizer:
    def __init__(self, set_min, set_max, name='normalizer'):
        self.set_min = set_min
        self.set_max = set_max
        self.name = name

    def process_frame(self, frame):
        frame_min = frame.min()
        frame_max = frame.max()
        set_range = self.set_max - self.set_min

        #  Convert the frame to float for computation
        frame = frame.astaype('float32')

        #  Create a matrix filled with minimum pixel value of the frame
        matrix_frame_min = np.full(frame.shape, frame_min, dtype='float32')

        numerator = cv.subtract(frame, matrix_frame_min)
        denominator = frame_max - frame_min

        if denominator == 0:
            denominator = 1

        normalized_frame = set_range * (numerator / denominator) + self.set_min

        return normalized_frame.astype('uint8')


class FrameProcessorRemover:
    def __init__(self, boxes, name='remover'):
        self.boxes = boxes  # Each box is a tuple (x1, x2, y1, y2)
        self.color_gray = (128, 128, 128)
        self.name = name

    def process_frame(self, frame):
        # Replace specified rectangular area with gray color
        height, width, _ = frame.shape

        for (x1, x2, y1, y2) in self.boxes:
            #  Ensure the rectangle coordinates are within frame dimensions
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            frame[y1:y2, x1:x2] = self.color_gray

        return frame


class FrameProcessorVideoWriter:
    def __init__(self, output_path, name='video_writer'):
        self.output_path = output_path
        self.out_video_writer = None
        self.name = name

    def process_frame(self, frame):
        if self.out_video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out_video_writer = cv.VideoWriter(self.output_path, fourcc, fps, (width, height))

        self.out_video_writer.write(frame)

        # return the frame for potential further processing or display
        return frame

    def release(self):
        if self.out_video_writer:
            self.out_video_writer.release()


class FrameProcessorDisplayer:
    def __init__(self, window_name='processed Video', name='displayer', box = 'False'):
        self.window_name = window_name
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        self.name = name
        self.bounding_box = box

    def plot_rectangle(self, frame, faces):
        # There are 4 values in face array, x,y,h(eight),w(idth)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
        return frame

    def process_frame(self, frame):
        if self.bounding_box == 'True':
            detector_Haar = cv.CascadeClassifier(
                'E:/Project.Pycharm/FaceDetection/Face_detection/Models/haarcascade_frontalface_alt.xml')
            GRAY = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Face detection
            faces_Haar = detector_Haar.detectMultiScale(GRAY, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            img_bbox = self.plot_rectangle(frame.copy(), faces_Haar)
            cv.imshow(self.window_name, img_bbox)
        else:
            cv.imshow(self.window_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            raise Exception('User interrupted video display.')

        return frame

    def release(self):
        cv.destroyWindow(self.window_name)


class FrameAnalyzerMTCNN:
    def __init__(self, name="mtcnn"):
        self.detector = MTCNN()
        self.name = name
        #  Store detected faces as well as coordinates for transfer
        self.detected_faces = []

    def analyze_frame(self, frame):
        self.detected_faces = []
        faces = self.detector.detect_faces(frame)

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


class FrameAnalyzerRetinaface:
    def __init__(self, name='retinaface'):
        self.name = name

    def analyze_frame(self, frame):
        faces = RetinaFace.detect_faces(frame)
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        confidence = self.get_confidence(faces)
        avg_confidence = np.mean(confidence) if confidence else 0  # Calculate the average confidence
        coordinates = self.get_face_coordinates(faces)
        return{
            f'face_count': face_count,
            f'face_area': face_area,
            f'confidence': confidence,
            f'average_confidence': avg_confidence,
            f'coordinate': coordinates,
        }

    def get_face_area(self, faces):
        face_area_sum = sum(face['facial_area'][2] * face['facial_area'][3] for face in faces)
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []

        for face in faces:
            coordinates.append(face['facial_area'])

        return coordinates

    def get_confidence(self, faces):
        confidence = []

        for face in faces:
            confidence.append(face['score'])

        return confidence


class FrameAnalyzerOpenCV:
    def __init__(self,
                 cascade_path=str(files("pyWitnessAI.OpenCV_Models").joinpath("haarcascade_frontalface_alt.xml")),
                 name='opencv'):
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        self.name = name

    def analyze_frame(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
        face_area = self.get_face_area(faces)
        face_coordinates = self.get_face_coordinates(faces)

        return {
            f'face_count': face_count,
            f'face_area': face_area,
            f'coordinates': face_coordinates
        }

    def get_face_area(self, faces):
        face_area_sum = sum([w * h for (x, y, w, h) in faces])
        return face_area_sum

    def get_face_coordinates(self, faces):
        coordinates = []
        for (x, y, w, h) in faces:
            coordinates.append([x, y, w, h])
        return coordinates


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

    def compare_faces(self, target_faces, filler_faces, model_name='Facenet', calculate_method='euclidean'):
        #  Use pre-detected faces for analysis
        frame_results = []
        model_name = model_name

        for target_face in target_faces:
            face_comparisons = []

            embedding_results_target = self.get_embedding(target_face, model_name)
            emb_target = np.array(embedding_results_target[0]['embedding'])

            for j, filler_face in enumerate(filler_faces):
                embedding_results_filler = self.get_embedding(filler_face, model_name)
                emb_filler = np.array(embedding_results_filler[0]['embedding'])

                if calculate_method == 'euclidean':
                    similarity_score = self.calculate_similarity_euclidean(emb_target, emb_filler)
                    face_comparisons.append({f'similarity_{j}': similarity_score})
                else:
                    raise ValueError(f"Unsupported detector backend: {calculate_method}")

            frame_results.append(face_comparisons)

        return frame_results

    def get_embedding(self, face, model_name):
        #  Generate embedding using FaceNet
        embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)
        return np.array(embedding)

    def calculate_similarity_euclidean(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    def save(self, data, directory='results', column_name=None):
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
        csv_filename = 'similarity_scores.csv'

        # Save to CSV
        df_transposed.to_csv(os.path.join(directory, csv_filename), index=True)
        print(f'Data saved to {csv_filename}.')

    













