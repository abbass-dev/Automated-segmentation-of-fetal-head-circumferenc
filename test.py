from utils import parse_configuration
from models.SegModel import SegModel
config = parse_configuration("config.json")
number_of_epoch = config['train_params']['number_of_epoch']
#transformer = create_transformer(config['transformer_params'])
print('initializing Dataset')
#train,val = create_dataset(transformer,**config["train_dataset_params"])
print(number_of_epoch)
print('initializing fsModel')
model = SegModel(config['model_params'])