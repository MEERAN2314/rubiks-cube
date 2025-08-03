# Rubik's Cube Solver (Hackathon Project)

This project provides a web-based interface for solving a standard 3x3 Rubik's Cube. It allows users to input a scrambled cube state and then visualizes the solution steps. The project leverages a combination of Python, FastAPI, and JavaScript to create an interactive and user-friendly experience. While the core solving algorithm is not yet fully implemented, the project provides a solid foundation for building a complete Rubik's Cube solver.

## Tech Stack

*   **Python**: The core programming language used for implementing the solver logic and the web application.
*   **FastAPI**: A modern, high-performance web framework for building APIs in Python. It's used to create the web interface and handle user requests.
*   **Jinja2**: A templating engine for Python that allows dynamic generation of HTML pages. It's used to render the user interface with the current cube state and solution steps.
*   **Langchain**: (Optional/Future) A framework for developing applications powered by language models. Could be used for AI-assisted solving or user interaction.
*   **PyMongo**: The official Python driver for MongoDB, used for interacting with the MongoDB database.
*   **MongoDB Atlas**: A cloud-hosted NoSQL database used for storing cube states, solutions, and potentially user data.
*   **python-dotenv**: A library for loading environment variables from a `.env` file, used to manage sensitive information such as the MongoDB connection URI.
*   **uvicorn**: An ASGI server for running FastAPI applications in production.

## Project Structure

```
.
├── main.py                 # FastAPI application entry point: Handles web requests and orchestrates the solving process.
├── cube_solver.py          # Contains Rubik's Cube representation and move logic: Implements the core cube data structure and move operations.
├── templates/
│   └── index.html          # Frontend HTML for the web interface: Provides the user interface for interacting with the solver.
├── .env                    # Environment variables (e.g., MongoDB URI): Stores sensitive configuration data.
└── requirements.txt        # Python dependencies: Lists all the required Python packages for the project.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MEERAN2314/rubiks-cube.git
    cd rubiks-cube
    ```

2.  **Create a virtual environment (recommended):**
    It's highly recommended to create a virtual environment to isolate the project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux or macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    Install the required Python packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MongoDB Atlas (Optional):**
    If you want to store cube states and solutions, you can configure a MongoDB Atlas database.
    *   Create a free cluster on [MongoDB Atlas](https://cloud.mongodb.com/).
    *   Get your connection string from the Atlas dashboard.
    *   Open the `.env` file and replace `"YOUR_MONGODB_ATLAS_CONNECTION_STRING"` with your actual connection string.
        ```
        MONGO_URI="mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority"
        ```
        **Note:** Replace `<username>`, `<password>`, and `<cluster-url>` with your actual MongoDB credentials.

## Running the Application

To start the FastAPI application:

```bash
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## How to Use

1.  **Access the Web Interface:** Open your web browser and navigate to `http://127.0.0.1:8000`.
2.  **Input the Scrambled Cube State:** Enter the scrambled cube state in the designated input field. The cube state must be a 54-character string, where each character represents the color of a sticker on the cube.
    *   The colors are represented by the following letters:
        *   `U`: White (Up face)
        *   `R`: Red (Right face)
        *   `F`: Green (Front face)
        *   `D`: Yellow (Down face)
        *   `L`: Orange (Left face)
        *   `B`: Blue (Back face)
    *   The order of the characters in the string matters and corresponds to the following facelet order: UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
    *   **Solved State Example**: `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`
3.  **Solve the Cube:** Click the "Solve Cube" button to submit the scrambled state to the solver.

## Current State of the Solver

The `cube_solver.py` currently provides the following functionality:

*   **RubiksCube Class:** A Python class that represents the state of a 3x3 Rubik's Cube. The cube's state is stored as a string of 54 characters, where each character represents the color of a sticker.
*   **apply_move Method:** A robust method that correctly simulates standard Rubik's Cube rotations. This method takes a move (e.g., "R", "U'", "F2") as input and updates the cube's state accordingly. It supports all standard rotations (R, R', R2, L, L', L2, U, U', U2, D, D', D2, F, F', F2, B, B', B2).
*   **solve Method (Placeholder):** A placeholder method for implementing the core Rubik's Cube solving algorithm. **This method is currently not implemented and needs to be filled in with a solving algorithm.**

## Next Steps / Areas for Improvement

The following is a list of suggested improvements, ordered by priority:

1.  **Implement the Solving Algorithm**: This is the most critical part. The `solve` method in `cube_solver.py` is currently a placeholder and needs to be implemented with a concrete solving algorithm. Consider the following algorithms:
    *   **Layer-by-Layer Method**: A relatively simple algorithm to implement, but may not be the most efficient in terms of the number of moves required.
    *   **Fridrich Method (CFOP)**: A more advanced algorithm commonly used by speedcubers. This algorithm requires learning a set of algorithms for different cases.
    *   **Kociemba's Algorithm**: A two-phase algorithm known for finding solutions in a relatively small number of moves. This algorithm typically involves using a precomputed lookup table (pruning table).
    *   **IDA* (Iterative Deepening A*)**: A search algorithm that can be used to find optimal solutions (i.e., solutions with the fewest number of moves).
2.  **Visual Simulation/Cube UI**: Enhance the user experience by integrating a 3D visualization of the cube. This could be achieved using a JavaScript library such as Three.js. The visualization would need to be updated in real-time to reflect the cube's state as the solution is being applied.
3.  **State Representation**: Explore alternative state representations that may be more efficient for the solving algorithm. The current string representation is easy to work with for applying moves, but may not be the most efficient for searching for solutions. Consider using a more abstract representation based on permutations of corners and edges.
4.  **Error Handling and Validation**: Implement more robust input validation to ensure that the user provides a valid 54-character string representing the cube state. This would involve checking that the string contains only valid characters (U, R, F, D, L, B) and that each character appears the correct number of times.
5.  **Testing**: Implement unit tests for the `RubiksCube` class and the `apply_move` method to ensure that the code is working correctly and that the moves are being applied as expected.
6.  **Langchain Integration**: Explore using Langchain to allow users to describe a scrambled cube or ask for hints in natural language. The LLM could then translate the natural language input into cube states or moves.
7.  **Scalability**: (Optional Bonus) Extend the `cube_solver.py` to handle different cube sizes (e.g., 2x2, 4x4, 5x5).

This project provides a solid foundation for building an award-winning Rubik's Cube solver. The `apply_move` logic is a significant step, and the remaining challenge lies in implementing an efficient and intelligent solving algorithm.
