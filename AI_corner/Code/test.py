from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'

detector = Predictor(config)

pth = ".\\output\\add_1.jpg"
img = Image.open(pth)
s = detector.predict(img,return_prob=True)
print(s)
