import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from controllers import image_controller, models_controller  
from output_messages.output import BasicMessage
import logging

app = FastAPI()

app.include_router(image_controller.router, tags=["Images"])
app.include_router(models_controller.router, tags=["AI Models"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title = "Muret Classification Server",
        version = "1.5",
        description = "Muret Classification Server official documentation",
        routes=app.routes)

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get('/ping', tags=["Healthcheck"])
async def ping():
    return BasicMessage(message=f'pong')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)