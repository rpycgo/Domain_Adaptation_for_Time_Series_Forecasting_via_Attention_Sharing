from layers.encoder import PrivateEncoders

import json



if __name__ == '__main__':
    with open('./config/model.json') as file:
        config = json.load(file)
    PrivateEncoders(config)