from pathlib import Path


ROOT_PATH: Path = \
    Path('/ds') if Path('/ds').exists() else \
    Path(__file__).parent.parent.parent

