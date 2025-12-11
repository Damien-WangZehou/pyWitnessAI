from pyWitnessAI import Images, ImagesAI
from pytest import approx

# ---------- Georgia State ----------
def test_ImagesAI_similarity_georgia_pipeline():
    """
    Test to reproduce the similarity scores from Kleider-Offutt et al. (2024)
    using the dedicated process_georgia_pipeline method.
    """
    # Load images as specified in the notebook's code cell for the Georgia pipeline
    WH_column_images_Georgia = Images("./data/01_Georgia_State_Video1/Video1_ProbeImage.png")
    WH_row_images_Georgia = Images([
        "./data/01_Georgia_State_Video1/*Mugshot*",
        "./data/01_Georgia_State_Video1/Video1_Perpetrator.png"
    ])

    image_analyzer_Georgia = ImagesAI(
        column_images=WH_column_images_Georgia,
        row_images=WH_row_images_Georgia,
    )

    image_analyzer_Georgia.process_georgia_pipeline()
    df = image_analyzer_Georgia.dataframe()

    # Expected values updated from the notebook
    assert df.loc["Video1_Perpetrator", "Video1_ProbeImage"] == approx(0.8759, abs=1e-5)
    assert df.loc["Video1_Mugshot2", "Video1_ProbeImage"] == approx(1.2342, abs=1e-5)
    assert df.loc["Video1_Mugshot3", "Video1_ProbeImage"] == approx(1.3474, abs=1e-5)
    assert df.loc["Video1_Mugshot4", "Video1_ProbeImage"] == approx(1.2398, abs=1e-5)
    assert df.loc["Video1_Mugshot5", "Video1_ProbeImage"] == approx(1.2717, abs=1e-5)
    assert df.loc["Video1_Mugshot6", "Video1_ProbeImage"] == approx(1.2998, abs=1e-5)
    assert df.loc["Video1_Mugshot7", "Video1_ProbeImage"] == approx(1.2906, abs=1e-5)


# ---------- different distance metrics ----------
def test_ImagesAI_similarity_mtcnn_facenet_euclidean_l2():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        align=False,
        enforce_detection=False,
        model="Facenet",
        backend="mtcnn",
        distance_metric="euclidean_l2",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(1.0869, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(1.2359, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(1.2768, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(1.3133, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(1.1511, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(1.2442, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_facenet_cosine():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="mtcnn",
        distance_metric="cosine",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(0.5906, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(0.7637, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(0.8151, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(0.8624, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(0.6626, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(0.7740, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="mtcnn",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(13.0366, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(14.7332, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(15.2573, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(15.6377, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.4157, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(14.8153, abs=1e-3)


# ---------- different detection backends ----------
def test_ImagesAI_similarity_opencv_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="opencv",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(11.6449, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(14.3952, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.1869, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(14.9362, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.5124, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(14.2616, abs=1e-3)


def test_ImagesAI_similarity_fastmtcnn_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="fastmtcnn",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(13.0325, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(14.7854, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(15.0674, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(15.9633, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.4093, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(14.8043, abs=1e-3)


def test_ImagesAI_similarity_ssd_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="ssd",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(12.7515, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(15.0852, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.4758, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(15.2980, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.4211, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(14.8717, abs=1e-3)


def test_ImagesAI_similarity_dlib_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="dlib",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(12.1419, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(15.4491, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.0794, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(14.8101, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.8467, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(15.0814, abs=1e-3)


def test_ImagesAI_similarity_retinaface_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="retinaface",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(14.0947, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(15.2000, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.5374, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(14.6287, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(12.7990, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(14.7612, abs=1e-3)


def test_ImagesAI_similarity_yunet_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="yunet",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(12.5380, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(15.0752, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.7540, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(15.5789, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.1358, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(15.3376, abs=1e-3)


def test_ImagesAI_similarity_centerface_facenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet",
        backend="centerface",
        distance_metric="euclidean",
        normalization="Facenet"
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(12.6939, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(15.1955, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(14.4030, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(15.2244, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(13.4164, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(15.1770, abs=1e-3)


# ---------- different models ----------
def test_ImagesAI_similarity_mtcnn_vggface_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="VGG-Face",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(1.2198, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(1.3102, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(1.2750, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(1.2426, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(1.2195, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(1.2934, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_facenet512_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Facenet512",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(24.0495, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(22.9667, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(25.2377, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(26.8565, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(23.8400, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(28.4808, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_openface_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="OpenFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(0.8041, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(0.9918, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(0.9186, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(0.8266, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(0.8583, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(0.8997, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_deepid_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="DeepID",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(53.9863, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(62.8030, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(64.1543, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(58.0046, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(93.2651, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(55.7571, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_arcface_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="ArcFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(5.5961, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(5.3795, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(5.7088, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(5.9477, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(5.2541, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(5.9396, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_dlib_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="Dlib",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(0.6366, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(0.6068, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(0.6362, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(0.6377, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(0.6038, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(0.6575, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_sface_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="SFace",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(11.0124, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(9.3277, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(11.1553, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(11.6845, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(8.1307, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(11.6240, abs=1e-3)


def test_ImagesAI_similarity_mtcnn_ghostfacenet_euclidean():
    WH_column_images = Images("./data/01_Georgia_State_Video1/Video1_Perpetrator.png")
    WH_row_images = Images("./data/01_Georgia_State_Video1/*Mugshot*")

    image_analyzer = ImagesAI(
        column_images=WH_column_images,
        row_images=WH_row_images,
        model="GhostFaceNet",
        backend="mtcnn",
        distance_metric="euclidean",
    )

    image_analyzer.process()
    df = image_analyzer.dataframe()

    assert df["Video1_Perpetrator"][0] == approx(39.3741, abs=1e-3)
    assert df["Video1_Perpetrator"][1] == approx(40.6091, abs=1e-3)
    assert df["Video1_Perpetrator"][2] == approx(41.3193, abs=1e-3)
    assert df["Video1_Perpetrator"][3] == approx(42.9959, abs=1e-3)
    assert df["Video1_Perpetrator"][4] == approx(40.3540, abs=1e-3)
    assert df["Video1_Perpetrator"][5] == approx(49.0519, abs=1e-3)
