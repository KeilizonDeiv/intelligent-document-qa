from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str = "ok"
    has_api: bool
