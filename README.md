# Rubik's Cube Solver (Hackathon Project)

This project aims to implement an algorithm to solve a standard 3x3 Rubik's Cube from any scrambled state, mimicking real-world solving logic. It uses Python with FastAPI for the web interface, Jinja2 for templating, and MongoDB Atlas for potential state storage (though not fully utilized in the current solver logic). Langchain is included as an optional tech stack for future enhancements, possibly for natural language interaction or AI-driven move suggestions.

## Tech Stack

*   **Python**: Core programming language.
*   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
*   **Jinja2**: A modern and designer-friendly templating language for Python.
*   **Langchain**: (Optional/Future) A framework for developing applications powered by language models. Could be used for AI-assisted solving or user interaction.
*   **PyMongo**: Python driver for MongoDB.
*   **MongoDB Atlas**: Cloud-hosted NoSQL database for storing cube states, solutions, or user data.
*   **python-dotenv**: For managing environment variables.
*   **uvicorn**: ASGI server for running FastAPI applications.

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── cube_solver.py          # Contains Rubik's Cube representation and move logic
├── templates/
│   └── index.html          # Frontend HTML for the web interface
├── .env                    # Environment variables (e.g., MongoDB URI)
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MEERAN2314/rubiks-cube.git
    cd rubiks-cube
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MongoDB Atlas:**
    *   Create a free cluster on [MongoDB Atlas](https://cloud.mongodb.com/).
    *   Get your connection string.
    *   Open the `.env` file and replace `"YOUR_MONGODB_ATLAS_CONNECTION_STRING"` with your actual connection string:
        ```
        MONGO_URI="mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority"
        ```

## Running the Application

To start the FastAPI application:

```bash
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## How to Use

1.  Open your web browser and navigate to `http://127.0.0.1:8000`.
2.  Enter a scrambled cube state in the input field. The state should be a 54-character string representing the colors of the stickers.
    *   **Solved State Example**: `WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB`
        (U: White, R: Red, F: Green, D: Yellow, L: Orange, B: Blue)
3.  Click "Solve Cube".

## Current State of the Solver

The `cube_solver.py` currently includes:
*   A `RubiksCube` class for representing the cube's state.
*   A robust `apply_move` method that correctly simulates standard Rubik's Cube rotations (R, R', R2, L, L', L2, U, U', U2, D, D', D2, F, F', F2, B, B', B2).
*   A placeholder `solve` method. **The core Rubik's Cube solving algorithm needs to be implemented here.** This is the primary challenge for the hackathon.

## Next Steps / Areas for Improvement

1.  **Implement the Solving Algorithm**: This is the most critical part. Consider algorithms like:
    *   **Layer-by-Layer Method**: Simpler to implement, but might not be the most efficient.
    *   **Fridrich Method (CFOP)**: More advanced, widely used by speedcubers.
    *   **Kociemba's Algorithm**: A two-phase algorithm known for finding solutions in a small number of moves. This would require building a large lookup table (pruning table) or using a pre-computed one.
    *   **IDA* (Iterative Deepening A*)**: A search algorithm that can find optimal solutions.
2.  **State Representation**: While the current string representation works for `apply_move`, consider if a more abstract representation (e.g., using permutations of corners and edges) would be more efficient for the solving algorithm.
3.  **Visual Simulation/Cube UI**: (Bonus) Integrate a 3D visualization of the cube using a JavaScript library (e.g., Three.js) to enhance the "Wow factor". This would involve sending cube state updates from FastAPI to the frontend.
4.  **Scalability**: (Optional Bonus) Extend the `cube_solver.py` to handle different cube sizes (2x2, 4x4, etc.).
5.  **Langchain Integration**: Explore using Langchain to allow users to describe a scrambled cube or ask for hints in natural language, and have the LLM translate that into cube states or moves.
6.  **Error Handling and Validation**: Add more robust input validation for the scrambled state.
7.  **Testing**: Implement unit tests for the `RubiksCube` class and the `apply_move` method to ensure correctness.

This project provides a solid foundation for building an award-winning Rubik's Cube solver. The `apply_move` logic is a significant step, and the remaining challenge lies in implementing an efficient and intelligent solving algorithm.
