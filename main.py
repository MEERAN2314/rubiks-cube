from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from cube_solver import RubiksCube

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.rubiks_solver

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/solve", response_class=HTMLResponse)
async def solve_cube(request: Request, scrambled_state: str = Form(...)):
    cube = RubiksCube(state=scrambled_state)
    solution_steps = cube.solve()
    return templates.TemplateResponse("index.html", {"request": request, "solution": "\n".join(solution_steps)})
