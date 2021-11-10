from flask.app import Flask
import torch


def read_model():
    model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                           **{'topN': 6, 'device': 'cpu', 'num_classes': 200})

    model.load_state_dict(torch.load("./bird-model.pth"))

    model.eval()
    return model


class MLFlask(Flask):
    model = read_model()

    def run(self, host="127.0.0.1", port=9876, debug=None,  **options):
        super(MLFlask, self).run(host=host, port=port,
                                 debug=debug,  **options)
