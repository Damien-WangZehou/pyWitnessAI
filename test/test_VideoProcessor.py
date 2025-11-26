from pyWitnessAI.Video import *
from pyWitnessAI.VideoAI import *
from pyWitnessAI.VideoProcessor import *


def exclude_test():
    #  Input the path and arguments needed
    video_path = "D:/MscPsy/Data/Colloff2021/Eyewitness video.wmv"
    #  cascade_path = 'E:/Project.Pycharm/FaceDetection/Face_detection/Models/haarcascade_frontalface_alt.xml'
    a = VideoAnalyzer(video_path)
    # lineup_number = 2

    mt = FrameAnalyzerMTCNN()
    op = FrameAnalyzerOpenCV()
    # a.add_analyzer(mt)
    # a.add_analyzer(op)

    cropper = FrameProcessorCropper(10, 500, 500, 10)
    a.add_processor(cropper)

    display_processor = FrameProcessorDisplayer(box='True')
    a.add_processor(display_processor)

    writer = FrameProcessorVideoWriter("D:/MscPsy/Data/Colloff2021/cropped")
    a.add_processor(writer)
    # #  Load the lineup
    # lineup_loader = LineupLoader(directory_path='D:/MscPsy/Data/Colloff2021/TargetFace')
    # lineup = lineup_loader.load_image()
    #
    # #  Comparison of lineup faces with faces in the video
    # similarity_analyzer = SimilarityAnalyzer(lineup_faces=lineup, detector=mt)
    # a.add_analyzer(similarity_analyzer)

    a.run(0, 10)

    # print(a.get_analysis_info())
    #
    # a.save_data_flattened()
    #
    plt.figure(figsize=(9, 6))
    plt.suptitle("Video analysis", fontsize=24, fontweight='bold')
    plt.subplot(2, 2, 1)
    a.plot_face_counts()
    plt.subplot(2, 2, 2)
    a.plot_face_areas()
    plt.subplot(2, 2, 3)
    a.plot_average_pixel_values()
    plt.subplot(2, 2, 4)
    a.plot_mtcnn_confidence_histogram()

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()
