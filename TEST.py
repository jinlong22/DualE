import neuralkg
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import wandb
from neuralkg.utils import setup_parser
from neuralkg.utils.tools import *
from neuralkg.data.Sampler import *
import yaml

from models import *

parser = neuralkg.setup_parser()
args = parser.parse_args(args=[])

config_path = "DualE.yaml"
args = load_config(args, config_path)
# print(args.lmbda)
args.data_path = "/Users/jinlong/NeuralKG/dataset/FB15K237"
train_sampler = neuralkg.import_class("neuralkg.data.UniSampler")(args)
test_sampler = neuralkg.import_class("neuralkg.data.TestSampler")(train_sampler)
kgdata = neuralkg.import_class("neuralkg.data.KGDataModule")(args, train_sampler, test_sampler)
model = neuralkg.import_class("models.DualE")(args)
lit_model = neuralkg.import_class("neuralkg.lit_model.KGELitModel")(model, args)

trainer = pl.Trainer.from_argparse_args(args)
# trainer.fit(lit_model, datamodule=kgdata)