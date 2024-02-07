import hydra
from omegaconf import DictConfig, OmegaConf

from training import training


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    training(cfg)
    


if __name__ == "__main__":
    main()
