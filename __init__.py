from flask.app import Flask
from PIL import Image
import torch
from torch import nn
from torchvision import models

from torchvision import transforms
import faulthandler

faulthandler.enable()


def read_model():
    model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                           **{'topN': 6, 'device': 'cpu', 'num_classes': 200})

    model.load_state_dict(torch.load("./bird-model.pth"))

    model.eval()
    return model


class MLFlask(Flask):
    model = read_model()

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        super(MLFlask, self).run(host=host, port=port,
                                 debug=debug, load_dotenv=load_dotenv, **options)


app = MLFlask(__name__)
app.run()
