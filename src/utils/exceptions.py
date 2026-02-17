class DAMOException(Exception):
    pass


class ModelNotFoundError(DAMOException):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class VRAMExhaustedError(DAMOException):
    def __init__(self, required_gb: float, available_gb: float):
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(
            f"Not enough VRAM: need {required_gb:.1f}GB, "
            f"available {available_gb:.1f}GB"
        )


class InvalidParametersError(DAMOException):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"Invalid parameters: {detail}")


class DownloadError(DAMOException):
    def __init__(self, model_id: str, reason: str):
        self.model_id = model_id
        super().__init__(f"Failed to download {model_id}: {reason}")
