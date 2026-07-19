import os
import pandas as pd
import cv2 as cv
import numpy as np

from .VideoAnalysis import FrameAnalysisResult, normalize_box_xywh


class FrameProcessorCropper:
    def __init__(self, x1, x2, y1, y2, name='cropper'):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.name = name
        self.stage = 'pre'

    def process_frame(self, frame):
        #  Crop the frame using the provided coordinates
        cropped_frame = frame[self.y1:self.y2, self.x1:self.x2]

        return cropped_frame

class FrameProcessorMonochrome:
    def __init__(self, name='monochrome'):
        self.name = name
        self.stage = 'pre'

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
        self.stage = 'pre'

    def process_frame(self, frame):
        frame_min = frame.min()
        frame_max = frame.max()
        set_range = self.set_max - self.set_min

        #  Convert the frame to float for computation
        frame = frame.astype('float32')

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
        self.stage = 'post'

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


class FrameProcessorFaceMasker:
    def __init__(self, keep=None, remove=None, face_table=None,
                 analyzer_name='identity', color_gray=(128, 128, 128),
                 padding=0, remove_unlabeled=False, mask_style='gray',
                 mosaic_blocks=12, blur_kernel=31, name='face_masker'):
        if keep is not None and remove is not None:
            raise ValueError("keep and remove cannot be used together.")
        self.keep = set(keep) if keep is not None else None
        self.remove = set(remove) if remove is not None else None
        self.face_table = self._coerce_face_table(face_table)
        self.analyzer_name = analyzer_name
        self.color_gray = color_gray
        self.padding = int(padding)
        self.remove_unlabeled = bool(remove_unlabeled)
        self.mask_style = str(mask_style).lower()
        if self.mask_style not in {'gray', 'grey', 'mosaic', 'blur'}:
            raise ValueError("mask_style must be 'gray', 'mosaic', or 'blur'.")
        self.mosaic_blocks = max(1, int(mosaic_blocks))
        self.blur_kernel = max(3, int(blur_kernel))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.name = name
        self.stage = 'post'

    def process_frame(self, frame, context=None):
        frame_index = None if context is None else context.get('frame_index', context.get('frame'))
        faces = self._faces_for_frame(frame_index, context)
        for face in faces:
            label = face.get('label')
            if pd.isna(label):
                label = None
            if self._should_remove(label):
                self._mask_box(frame, face['bbox'])
        return frame

    def _should_remove(self, label):
        if self.keep is not None:
            return label not in self.keep
        if self.remove is not None:
            return label in self.remove or (label is None and self.remove_unlabeled)
        return True

    def _faces_for_frame(self, frame_index, context):
        if self.face_table is not None:
            return self._faces_from_table(frame_index)
        return self._faces_from_context(context)

    def _faces_from_context(self, context):
        if not context:
            return []
        analysis = context.get('analysis', {}) or {}
        result = analysis.get(self.analyzer_name)
        if result is None:
            result = next((value for value in analysis.values() if isinstance(value, dict) and 'labels' in value), None)
        if not isinstance(result, dict):
            return []
        frame_result = FrameAnalysisResult.from_dict(result, frame_index=context.get('frame_index'))
        return [
            {'label': detection.label, 'bbox': detection.bbox}
            for detection in frame_result.detections
        ]

    def _faces_from_table(self, frame_index):
        if frame_index is None or self.face_table.empty:
            return []
        df = self.face_table[self.face_table['frame'] == frame_index]
        if self.analyzer_name and 'analyzer' in df.columns:
            named = df[df['analyzer'] == self.analyzer_name]
            if not named.empty:
                df = named
        faces = []
        for _, row in df.iterrows():
            faces.append({
                'label': row.get('label'),
                'bbox': [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']],
            })
        return faces

    def _mask_box(self, frame, box):
        x, y, w, h = normalize_box_xywh(box, frame.shape)
        if self.padding:
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(frame_w, x + w + self.padding)
            y2 = min(frame_h, y + h + self.padding)
        else:
            x1, y1, x2, y2 = x, y, x + w, y + h
        region = frame[y1:y2, x1:x2]
        if not region.size:
            return
        if self.mask_style in {'gray', 'grey'}:
            region[:] = self.color_gray
        elif self.mask_style == 'mosaic':
            region_h, region_w = region.shape[:2]
            small_w = max(1, region_w // self.mosaic_blocks)
            small_h = max(1, region_h // self.mosaic_blocks)
            pixelated = cv.resize(region, (small_w, small_h), interpolation=cv.INTER_AREA)
            region[:] = cv.resize(pixelated, (region_w, region_h), interpolation=cv.INTER_NEAREST)
        else:
            region[:] = cv.GaussianBlur(region, (self.blur_kernel, self.blur_kernel), 0)

    @staticmethod
    def _coerce_face_table(face_table):
        if face_table is None:
            return None
        if isinstance(face_table, pd.DataFrame):
            return face_table.copy()
        if isinstance(face_table, str):
            return pd.read_csv(face_table)
        return pd.DataFrame(face_table)


class FrameProcessorFaceRemover(FrameProcessorFaceMasker):
    """Backward-compatible name for :class:`FrameProcessorFaceMasker`."""

    def __init__(self, *args, name='face_remover', **kwargs):
        super().__init__(*args, name=name, **kwargs)

class FrameProcessorVideoWriter:
    def __init__(self, output_path, fps=None, codec='mp4v', name='video_writer'):
        self.output_path = output_path
        self.out_video_writer = None
        self.name = name
        self.fps = fps
        self.codec = codec
        self.frame_size = None
        self.stage = 'post'

    def process_frame(self, frame, context=None):
        if self.out_video_writer is None:
            height, width, _ = frame.shape
            output_directory = os.path.dirname(os.path.abspath(self.output_path))
            os.makedirs(output_directory, exist_ok=True)
            fourcc = cv.VideoWriter_fourcc(*self.codec)
            context_fps = None if context is None else context.get('fps')
            fps = float(self.fps or context_fps or 30)
            self.frame_size = (width, height)
            self.out_video_writer = cv.VideoWriter(self.output_path, fourcc, fps, (width, height))
            if not self.out_video_writer.isOpened():
                self.out_video_writer.release()
                self.out_video_writer = None
                raise RuntimeError(f"Cannot open output video for writing: {self.output_path}")

        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                f"Frame size changed from {self.frame_size} to {(width, height)} while writing."
            )

        self.out_video_writer.write(frame)

        # return the frame for potential further processing or display
        return frame

    def release(self):
        if self.out_video_writer is not None:
            self.out_video_writer.release()
            self.out_video_writer = None
        self.frame_size = None

    def reset(self):
        self.release()

class FrameProcessorDisplayer:
    def __init__(self, window_name='processed Video', name='displayer', box='False'):
        self.window_name = window_name
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        self.name = name
        self.bounding_box = box if isinstance(box, bool) else str(box).lower() == 'true'
        self.detector_Haar =  cv.CascadeClassifier(
                'E:/Project.Pycharm/FaceDetection/Face_detection/Models/haarcascade_frontalface_alt.xml')
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        self.stage = 'post'

    def plot_rectangle(self, frame, faces):
        # There are 4 values in face array, x,y,h(eight),w(idth)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
        return frame

    def process_frame(self, frame):
        if self.bounding_box:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.detector_Haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            frame = self.plot_rectangle(frame.copy(), faces)
            cv.imshow(self.window_name, frame)
        else:
            cv.imshow(self.window_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            raise Exception('User interrupted video display.')

        return frame

    def release(self):
        cv.destroyWindow(self.window_name)

# Enhance the contrast of the image
class FrameProcessorHistogramEqualization:
    def __init__(self, name='histogram_equalization'):
        self.name = name
        self.stage = 'pre'

    def process_frame(self, frame):
        img_yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
        img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        return img_output

# Reduce noise that might interfere with face detection
class FrameProcessorNoiseReduction:
    def __init__(self, name='noise_reduction'):
        self.name = name
        self.stage = 'pre'

    def process_frame(self, frame):
        return cv.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

# Adjust the brightness of the image
class FrameProcessorGammaCorrection:
    def __init__(self, gamma=1.0, name='gamma_correction'):
        self.gamma = gamma
        self.name = name
        self.inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.stage = 'pre'

    def process_frame(self, frame):
        return cv.LUT(frame, self.table)

# Enhance the edges in the image, making faces more distinguishable
class FrameProcessorSharpening:
    def __init__(self, name='sharpening'):
        self.name = name
        self.stage = 'pre'

    def process_frame(self, frame):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv.filter2D(frame, -1, kernel)
