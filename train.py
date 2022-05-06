from pyexpat import model
from utils import parse_configuration
from models.SegModel import SegModel
from datasets import create_dataset, create_transformer


config = parse_configuration("congefig.json")
number_of_epoch = config['train_params']['number_of_epoch']
transformer = create_transformer(config['transformer_params'])
print('initializing Dataset')
train,val = create_dataset(transformer,**config["train_dataset_params"])
print(number_of_epoch)
print('initializing fsModel')
model = SegModel(config['model_params'])
data_size = len(train.dataset)
print('start training')
for epoch in range(number_of_epoch):
    running_loss = 0
    for data in train:
        model.set_input(data)
        model.forward()
        model.backward()
        model.optimize_parameters()
        running_loss += model.get_current_losses()['final'].item()
    print(running_loss/data_size)    
    model.update_learning_rate()
