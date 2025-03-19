import argparse

from config import Config
from lndp import make as model_factory
from task import make as task_factory, MultiTask
from trainer import make as trainer_factory

import equinox as eqx
import jax.random as jr


def load_config():
    default_config = Config()
    parser = argparse.ArgumentParser()
    for arg, value in default_config.__dict__.items():
        parser.add_argument(f"--{arg}", type=type(value), default=value)
    args = vars(parser.parse_args())
    return Config(**args)


if __name__ == '__main__':

    config = load_config()
    key_model, key_train = jr.split(jr.key(config.seed))

    if "," not in config.env_name:
        mdl = model_factory(config, key_model)
        params, statics = eqx.partition(mdl, eqx.is_array)
        task = task_factory(config, statics)
    else:
        env_names = config.env_name.split(",")
        statics = []
        tsks = []
        for env_name in env_names:
            _cfg = config._replace(env_name=env_name)
            mdl = model_factory(_cfg, key_model)
            params, statics = eqx.partition(mdl, eqx.is_array)
            task = task_factory(_cfg, statics)
            tsks.append(task)
        task = MultiTask(tsks)

    trainer = trainer_factory(config, task, params)  # type:ignore

    for seed in range(config.n_seeds):
        key_train, _ktrain = jr.split(key_train)
        if config.log:
            trainer.logger.init(config.project, config._replace(seed=config.seed + seed)._asdict())  # type:ignore
        trainer.init_and_train_(_ktrain)
        if config.log:
            trainer.logger.finish()  # type:ignore
