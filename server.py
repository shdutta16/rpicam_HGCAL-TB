from fastapi import FastAPI
from take_snap import run

app = FastAPI()

@app.get("/")
async def root():
    return run()