from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
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
    return templates.TemplateResponse("index.html", {"request": request, "solved_cube_state": 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'})

from fastapi.responses import JSONResponse

@app.get("/scramble", response_class=JSONResponse)
async def get_scramble(request: Request):
    cube = RubiksCube()
    scrambled_state, scramble_sequence = cube.generate_scramble()
    return {
        "scrambled_state": scrambled_state,
        "scramble_sequence": " ".join(scramble_sequence)
    }

@app.post("/solve", response_class=HTMLResponse)
async def solve_cube(request: Request, scrambled_state: str = Form(...)):
    solution_steps = []
    error_message = None
    try:
        cube = RubiksCube(state=scrambled_state)
        solution_steps = cube.solve()
        
        print(f"Scrambled State Received: {scrambled_state}")
        print(f"Solution Steps Generated: {solution_steps}")

        # Store the scrambled state and solution in MongoDB
        try:
            db.solutions.insert_one({
                "scrambled_state": scrambled_state,
                "solution": solution_steps,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")
            error_message = f"Database error: {e}" # Inform user about DB error if it occurs
            
    except ValueError as e:
        error_message = str(e)
        print(f"Input validation error: {e}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"Unexpected error during solving: {e}")

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "solution": "\n".join(solution_steps) if solution_steps else None,
        "error": error_message,
        "scrambled_state_input": scrambled_state # Pass back the input to pre-fill the form
    })
