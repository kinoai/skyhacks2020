# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# yaml
import yaml

# wandb
from pytorch_lightning.loggers import WandbLogger

# custom utils
from sky_utils.lightning_wrapper import LitModel
from sky_utils.data_modules import *
from sky_utils.callbacks import ExampleCallback, SaveOnnxToWandbCallback


def init_wandb(config, model, dataloader):
    wandb_logger = WandbLogger(
        project=config["loggers"]["wandb"]["project"],
        job_type=config["loggers"]["wandb"]["job_type"],
        tags=config["loggers"]["wandb"]["tags"],
        entity=config["loggers"]["wandb"]["team"],
        id=config["resume"]["wandb_run_id"] if config["resume"]["resume_from_ckpt"] else None,
        log_model=config["loggers"]["wandb"]["log_model"],
        offline=False
    )
    wandb_logger.watch(model.model, log=None)
    wandb_logger.log_hyperparams({
        "model_name": model.model.__class__.__name__,
        "dataset_name": dataloader.__class__.__name__,
        "optimizer": model.configure_optimizers().__class__.__name__,
        "train_size": len(dataloader.data_train),
        "val_size": len(dataloader.data_val),
        "test_size": len(dataloader.data_test),
        "input_dims": dataloader.input_dims,
    })
    # download model from a specific wandb run
    # wandb.restore('model-best.h5', run_path="kino/some_project/a1b2c3d")
    return wandb_logger


def main(config):

    # Init data module
    datamodule = SkyDataModule(batch_size=config["hparams"]["batch_size"])
    datamodule.prepare_data()
    datamodule.setup()

    # Init our model
    model = LitModel(config)

    # Init wandb logger
    # wandb_logger = []
    wandb_logger = init_wandb(config, model, datamodule)

    # Init callbacks
    callbacks = [
        EarlyStopping(
            monitor=config["callbacks"]["early_stop"]["monitor"],
            patience=config["callbacks"]["early_stop"]["patience"],
            mode=config["callbacks"]["early_stop"]["mode"],
        ),
        ModelCheckpoint(
            monitor=config["callbacks"]["checkpoint"]["monitor"],
            save_top_k=config["callbacks"]["checkpoint"]["save_top_k"],
            mode=config["callbacks"]["checkpoint"]["mode"],
            save_last=config["callbacks"]["checkpoint"]["save_last"],
        ),
        # ExampleCallback(),
        # LearningRateMonitor(),
        # SaveOnnxToWandbCallback(dataloader=datamodule.train_dataloader(), wandb_save_dir=wandb_logger.save_dir)
    ]

    # Init trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        gpus=config["num_of_gpus"],
        max_epochs=config["hparams"]["max_epochs"],
        resume_from_checkpoint=config["resume"]["ckpt_path"] if config["resume"]["resume_from_ckpt"] else None,
        accumulate_grad_batches=config["hparams"]["accumulate_grad_batches"],
        gradient_clip_val=config["hparams"]["gradient_clip_val"],
        progress_bar_refresh_rate=config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if config["printing"]["profiler"] else None,
        weights_summary=config["printing"]["weights_summary"],
        # fast_dev_run=True,
        # min_epochs=10,
        # limit_train_batches=0.05,
        # limit_val_batches=0.05,
        # limit_test_batches=0.01
        # auto_scale_batch_size="power",
        # amp_backend='apex',
        # precision=16,
    )

    # Test before training
    # trainer.test(model=model, datamodule=datamodule)
    # trainer.save_checkpoint("example.ckpt")

    # Train the model ⚡
    trainer.fit(model=model, datamodule=datamodule)


def load_config():
    with open("config.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    main(config=load_config())
