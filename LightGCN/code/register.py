import sys
sys.path.append('../')
from LightGCN.code import world
from LightGCN.code import dataloader
from LightGCN.code import model
from LightGCN.code import utils
from pprint import pprint
import os

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','cloth_sport', 'phone_elec', 'game_video', 'music_movie']:
    # print(os.getcwd())
    dataset = dataloader.Loader(path="../LightGCN/data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

# print('===========config================')
# pprint(world.config)
# print("cores for test:", world.CORES)
# print("comment:", world.comment)
# print("tensorboard:", world.tensorboard)
# print("LOAD:", world.LOAD)
# print("Weight path:", world.PATH)
# print("Test Topks:", world.topks)
# print("using bpr loss")
# print('===========end===================')

# print(model)
MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}