from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for application errors that map to a known HTTP status."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnsupportedFileTypeError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyDocumentError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class NoDocumentsError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class FileTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class DocumentNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error occurred"},
    )


def register_exception_handlers(app) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
