import hydra
from omegaconf import DictConfig

from rewam.runtime import train


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
