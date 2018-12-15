class AlignDlib():
    def __init__(self, facePredictor):
        assert facePredictor is not None # Verifica se o caminho definido é válido
        self.detector = dlib.get_frontal_face_detector() # Seleciona o detector do caminho de facePredictor definido (normalmente seleciona entre os landmark predictors de 68 ou 5 pontos)
        self.predictor = dlib.shape_predictor(facePredictor)

    def allBoudingBoxesInFace(self, rgbImg):
