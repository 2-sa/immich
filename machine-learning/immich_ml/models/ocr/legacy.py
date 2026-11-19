from pathlib import Path
from shutil import copy2

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from immich_ml.config import log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.schemas import ModelFormat, ModelSource, ModelType, Options

ARABIC_REC_HF_REPO = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec_onnx"


def extract_charset_from_paddle_yml(yml_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    characters = config.get("PostProcess", {}).get("character_dict")
    if not isinstance(characters, list) or not characters:
        raise ValueError(f"Invalid PaddleOCR character_dict in {yml_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(char) for char in characters), encoding="utf-8")


class TextModel[O: Options](InferenceModel[O]):
    sources = (ModelSource.PADDLE,)

    @property
    def _model_format_default(self) -> ModelFormat:
        if settings.legacy_models or self.model_name.startswith("ARABIC__"):
            return ModelFormat.ONNX
        return super()._model_format_default

    def _download_arabic(self) -> None:
        if self.model_type == ModelType.DETECTION:
            base_model = self.model_name.split("__", 1)[1]
            log.info(f"Downloading {self.model_type} model '{self.model_name}' from '{base_model}' to {self.model_dir}.")
            snapshot_download(
                f"{settings.model_organization}/{base_model}",
                revision="v2",
                cache_dir=self.cache_dir,
                local_dir=self.cache_dir,
                allow_patterns=["detection/*"],
                ignore_patterns=["*.armnn", "*.rknn"],
            )
            if not self.cached:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            return

        log.info(f"Downloading {self.model_type} model '{self.model_name}' from '{ARABIC_REC_HF_REPO}' to {self.model_dir}.")
        onnx_file = Path(hf_hub_download(ARABIC_REC_HF_REPO, "inference.onnx", cache_dir=self.cache_dir))
        yml_file = Path(hf_hub_download(ARABIC_REC_HF_REPO, "inference.yml", cache_dir=self.cache_dir))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        copy2(onnx_file, self.model_path)
        extract_charset_from_paddle_yml(yml_file, self.model_dir / "charset.txt")

    def download(self) -> None:
        if self.cached:
            return
        if self.model_name.startswith("ARABIC__"):
            self._download_arabic()
            return
        if not settings.legacy_models:
            return super().download()
        # here, as only an install still on the older exports has a use for it
        from rapidocr.inference_engine.base import FileInfo, InferSession
        from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
        from rapidocr.utils.typings import EngineType, LangDet, LangRec, OCRVersion, TaskType
        from rapidocr.utils.typings import ModelType as RapidModelType

        detects = self.model_type == ModelType.DETECTION
        language = self.model_name.split("__")[0] if "__" in self.model_name else "CH"
        version, size = self.model_name.split("__")[-1].rsplit("_", 1)
        model_info = InferSession.get_model_url(
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion(version),
                task_type=TaskType.DET if detects else TaskType.REC,
                lang_type=LangDet.CH if detects else LangRec[language],
                model_type=RapidModelType(size),
            )
        )
        log.info(
            f"Downloading {self.model_type} model '{self.model_name}' to {self.model_path}. This may take a while."
        )
        DownloadFile.run(
            DownloadFileInput(
                file_url=model_info["model_dir"], sha256=model_info["SHA256"], save_path=self.model_path, logger=log
            )
        )
