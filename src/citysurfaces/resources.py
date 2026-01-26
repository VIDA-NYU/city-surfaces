import os
from pathlib import Path


class Directory:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        self.parent = instance
        return self

    def __fspath__(self):
        return os.fspath(self.path)

    @property
    def path(self) -> Path:
        if hasattr(self, "_root"):
            return Path(self._root)
        return self.parent.path / self.name


class File(Directory):
    def __init__(self, extension: str):
        self.extension = extension

    @property
    def path(self) -> Path:
        return super().path.with_suffix(self.extension)


class Weights(Directory):
    block_c_10classes = File(".pth")
    hrnetv2_w48_imagenet_pretrained = File(".pth")

    @staticmethod
    def download(url: str):
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("gdown is required to download weights") from exc
        Project.resources.assets.weights.path.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=url, quiet=False, output=Project.resources.assets.weights.__fspath__())


class Assets(Directory):
    weights = Weights()


class Resources(Directory):
    assets = Assets()


class Project(Directory):
    resources = Resources()

    def __init__(self, root: Path):
        self._root = root


Project = Project(Path(__file__).resolve().parents[2])
