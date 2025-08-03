class RubiksCube:
    def __init__(self, state=None):
        # Standard Kociemba representation: UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
        self.solved_state = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
        
        self.face_map = {
            'U': (0, 9), 'R': (9, 18), 'F': (18, 27),
            'D': (27, 36), 'L': (36, 45), 'B': (45, 54)
        }
        # Center sticker indices for each face
        self.center_indices = {
            'U': 4, 'R': 13, 'F': 22, 'D': 31, 'L': 40, 'B': 49
        }
        # Expected center faces
        self.center_faces = {
            'U': 'U', 'R': 'R', 'F': 'F', 'D': 'D', 'L': 'L', 'B': 'B'
        }

        if state:
            self._validate_state(state)
            self.current_state = list(state)
        else:
            self.current_state = list(self.solved_state)
            
    def _validate_state(self, state_str):
        """Validates the input scrambled state string using face names."""
        if len(state_str) != 54:
            raise ValueError("Input state must be 54 characters long.")

        valid_faces = {'U', 'R', 'F', 'D', 'L', 'B'}
        face_counts = {face: 0 for face in valid_faces}

        for char in state_str:
            if char not in valid_faces:
                raise ValueError(f"Invalid face '{char}' found in state. Must be one of {', '.join(valid_faces)}.")
            face_counts[char] += 1
        
        for face, count in face_counts.items():
            if count != 9:
                raise ValueError(f"Incorrect number of '{face}' stickers. Expected 9, got {count}.")

        # Validate center stickers
        current_state_list = list(state_str)
        for face_char, center_idx in self.center_indices.items():
            if current_state_list[center_idx] != self.center_faces[face_char]:
                raise ValueError(f"Center sticker for face '{face_char}' must be '{self.center_faces[face_char]}'.")

    def _get_face_colors(self, face_char):
        start, end = self.face_map[face_char]
        return self.current_state[start:end]

    def _set_face_colors(self, face_char, colors):
        start, end = self.face_map[face_char]
        self.current_state[start:end] = colors

    def apply_move(self, move, state_str=None):
        """Applies a move to the cube state. If state_str is provided, applies to that state, otherwise to self.current_state."""
        if state_str:
            state_list = list(state_str)
        else:
            state_list = list(self.current_state)

        def rotate_face(face_indices, direction=1):
            """Rotates a 3x3 face (9 stickers) clockwise (1) or counter-clockwise (-1)."""
            face = [state_list[i] for i in face_indices]
            if direction == 1: # Clockwise
                new_face = [
                    face[6], face[3], face[0],
                    face[7], face[4], face[1],
                    face[8], face[5], face[2]
                ]
            else: # Counter-clockwise
                new_face = [
                    face[2], face[5], face[8],
                    face[1], face[4], face[7],
                    face[0], face[3], face[6]
                ]
            for i, val in zip(face_indices, new_face):
                state_list[i] = val

        def swap_edges(indices_sets, direction=1):
            """Swaps edge stickers between faces."""
            if direction == 1: # Clockwise
                temp = [state_list[i] for i in indices_sets[-1]]
                for i in range(len(indices_sets) - 1, 0, -1):
                    for j in range(len(indices_sets[i])):
                        state_list[indices_sets[i][j]] = state_list[indices_sets[i-1][j]]
                for j in range(len(indices_sets[0])):
                    state_list[indices_sets[0][j]] = temp[j]
            else: # Counter-clockwise
                temp = [state_list[i] for i in indices_sets[0]]
                for i in range(len(indices_sets) - 1):
                    for j in range(len(indices_sets[i])):
                        state_list[indices_sets[i][j]] = state_list[indices_sets[i+1][j]]
                for j in range(len(indices_sets[-1])):
                    state_list[indices_sets[-1][j]] = temp[j]

        # Define sticker indices for each face
        U = [0,1,2,3,4,5,6,7,8]
        R = [9,10,11,12,13,14,15,16,17]
        F = [18,19,20,21,22,23,24,25,26]
        D = [27,28,29,30,31,32,33,34,35]
        L = [36,37,38,39,40,41,42,43,44]
        B = [45,46,47,48,49,50,51,52,53]

        # Define affected edge stickers for each move
        moves = {
            'R': {
                'face': R,
                'edges': [
                    [F[2], F[5], F[8]], # F-right edge
                    [D[2], D[5], D[8]], # D-right edge
                    [B[6], B[3], B[0]], # B-left edge (reversed for R move)
                    [U[2], U[5], U[8]]  # U-right edge
                ]
            },
            'R\'': {
                'face': R,
                'edges': [
                    [F[2], F[5], F[8]],
                    [U[2], U[5], U[8]],
                    [B[6], B[3], B[0]],
                    [D[2], D[5], D[8]]
                ],
                'direction': -1
            },
            'R2': {
                'face': R,
                'edges': [
                    [F[2], F[5], F[8]],
                    [D[2], D[5], D[8]],
                    [B[6], B[3], B[0]],
                    [U[2], U[5], U[8]]
                ],
                'double': True
            },
            'L': {
                'face': L,
                'edges': [
                    [F[0], F[3], F[6]],
                    [U[0], U[3], U[6]],
                    [B[8], B[5], B[2]],
                    [D[0], D[3], D[6]]
                ],
                'direction': -1 # L is counter-clockwise relative to R
            },
            'L\'': {
                'face': L,
                'edges': [
                    [F[0], F[3], F[6]],
                    [D[0], D[3], D[6]],
                    [B[8], B[5], B[2]],
                    [U[0], U[3], U[6]]
                ]
            },
            'L2': {
                'face': L,
                'edges': [
                    [F[0], F[3], F[6]],
                    [U[0], U[3], U[6]],
                    [B[8], B[5], B[2]],
                    [D[0], D[3], D[6]]
                ],
                'double': True
            },
            'U': {
                'face': U,
                'edges': [
                    [F[0], F[1], F[2]],
                    [R[0], R[1], R[2]],
                    [B[0], B[1], B[2]],
                    [L[0], L[1], L[2]]
                ]
            },
            'U\'': {
                'face': U,
                'edges': [
                    [F[0], F[1], F[2]],
                    [L[0], L[1], L[2]],
                    [B[0], B[1], B[2]],
                    [R[0], R[1], R[2]]
                ],
                'direction': -1
            },
            'U2': {
                'face': U,
                'edges': [
                    [F[0], F[1], F[2]],
                    [R[0], R[1], R[2]],
                    [B[0], B[1], B[2]],
                    [L[0], L[1], L[2]]
                ],
                'double': True
            },
            'D': {
                'face': D,
                'edges': [
                    [F[6], F[7], F[8]],
                    [L[6], L[7], L[8]],
                    [B[6], B[7], B[8]],
                    [R[6], R[7], R[8]]
                ],
                'direction': -1 # D is counter-clockwise relative to U
            },
            'D\'': {
                'face': D,
                'edges': [
                    [F[6], F[7], F[8]],
                    [R[6], R[7], R[8]],
                    [B[6], B[7], B[8]],
                    [L[6], L[7], L[8]]
                ]
            },
            'D2': {
                'face': D,
                'edges': [
                    [F[6], F[7], F[8]],
                    [L[6], L[7], L[8]],
                    [B[6], B[7], B[8]],
                    [R[6], R[7], R[8]]
                ],
                'double': True
            },
            'F': {
                'face': F,
                'edges': [
                    [U[6], U[7], U[8]],
                    [R[0], R[3], R[6]],
                    [D[0], D[1], D[2]],
                    [L[2], L[5], L[8]]
                ]
            },
            'F\'': {
                'face': F,
                'edges': [
                    [U[6], U[7], U[8]],
                    [L[2], L[5], L[8]],
                    [D[0], D[1], D[2]],
                    [R[0], R[3], R[6]]
                ],
                'direction': -1
            },
            'F2': {
                'face': F,
                'edges': [
                    [U[6], U[7], U[8]],
                    [R[0], R[3], R[6]],
                    [D[0], D[1], D[2]],
                    [L[2], L[5], L[8]]
                ],
                'double': True
            },
            'B': {
                'face': B,
                'edges': [
                    [U[0], U[1], U[2]],
                    [L[0], L[3], L[6]],
                    [D[6], D[7], D[8]],
                    [R[2], R[5], R[8]]
                ],
                'direction': -1 # B is counter-clockwise relative to F
            },
            'B\'': {
                'face': B,
                'edges': [
                    [U[0], U[1], U[2]],
                    [R[2], R[5], R[8]],
                    [D[6], D[7], D[8]],
                    [L[0], L[3], L[6]]
                ]
            },
            'B2': {
                'face': B,
                'edges': [
                    [U[0], U[1], U[2]],
                    [L[0], L[3], L[6]],
                    [D[6], D[7], D[8]],
                    [R[2], R[5], R[8]]
                ],
                'double': True
            }
        }

        if move not in moves:
            print(f"Invalid move: {move}")
            return "".join(state_list) # Return current state if move is invalid

        move_info = moves[move]
        face_indices = move_info['face']
        edge_sets = move_info['edges']
        direction = move_info.get('direction', 1)
        is_double = move_info.get('double', False)

        # Apply face rotation
        rotate_face(face_indices, direction)
        if is_double:
            rotate_face(face_indices, direction) # Apply twice for double moves

        # Apply edge swaps
        swap_edges(edge_sets, direction)
        if is_double:
            swap_edges(edge_sets, direction) # Apply twice for double moves

        new_state_str = "".join(state_list)
        if not state_str: # Only update self.current_state if not a temporary state for get_neighbors
            self.current_state = list(new_state_str)
        return new_state_str

    def is_solved(self, state_str):
        """Checks if the cube is in a solved state."""
        return state_str == self.solved_state.replace(" ", "")

    def get_neighbors(self, current_state_str):
        """Generates all possible next states from the current state by applying a single move."""
        neighbors = []
        all_moves = ['R', 'R\'', 'R2', 'L', 'L\'', 'L2', 'U', 'U\'', 'U2',
                     'D', 'D\'', 'D2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2']
        
        for move in all_moves:
            new_state_str = self.apply_move(move, state_str=current_state_str)
            neighbors.append((new_state_str, move))
        
        return neighbors

    def solve(self):
        """Solves the Rubik's Cube using Kociemba's Algorithm."""
        current_state_str = "".join(self.current_state)
        
        # First, check if the cube is already solved.
        if self.is_solved(current_state_str):
            return ["Cube is already solved."]

        import kociemba

        try:
            # The state is already in the format expected by the kociemba library.
            solution = kociemba.solve(current_state_str)
            
            # Split the solution string into a list of moves
            solution_steps = solution.split(" ")

            # Apply the solution to the cube
            for move in solution_steps:
                self.apply_move(move)
            
            return solution_steps
        except Exception as e:
            # Re-raise the exception with a more specific message for the main app to catch.
            # This indicates an unsolvable or geometrically impossible cube state.
            raise ValueError(f"The provided cube state is invalid or unsolvable. A valid cube must have 9 stickers of each color and represent a possible physical configuration. Original error: {e}")

    def generate_scramble(self, num_moves=20):
        """Generates a random scramble by applying a specified number of random moves."""
        import random
        
        moves = ['R', 'R\'', 'R2', 'L', 'L\'', 'L2', 'U', 'U\'', 'U2',
                 'D', 'D\'', 'D2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2']
        
        # Start with the solved state as a list
        scrambled_state_list = list(self.solved_state)
        scramble_sequence = []
        
        for _ in range(num_moves):
            move = random.choice(moves)
            scramble_sequence.append(move)
            
            # Apply the move to the list representation of the state
            # The apply_move function returns a string, so we convert it back to a list
            scrambled_state_list = list(self.apply_move(move, state_str="".join(scrambled_state_list)))
            
        # Return the final scrambled state as a string
        return "".join(scrambled_state_list), scramble_sequence

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Solved state: UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
    # Using W, R, G, Y, O, B for colors
    solved_cube_str = 'WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB'
    
    cube = RubiksCube(state=solved_cube_str)
    print(f"Initial state: {''.join(cube.current_state)}")
    print(f"Is solved: {cube.is_solved(''.join(cube.current_state))}")

    # Test a single move (R)
    print("\nApplying R move:")
    new_state_r = cube.apply_move('R')
    print(f"State after R: {new_state_r}")
    
    # Test R'
    print("\nApplying R' move:")
    new_state_rp = cube.apply_move('R\'') # Now it applies to self.current_state
    print(f"State after R': {new_state_rp}")
    print(f"Is solved (should be true if R and R' cancel out): {cube.is_solved(new_state_rp)}")

    # Test a scramble and then attempt to "solve" with BFS
    scrambled_state_example = 'WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB' # Start from solved
    temp_cube = RubiksCube(state=scrambled_state_example)
    
    # Apply a few moves to scramble it
    temp_cube.apply_move('R')
    temp_cube.apply_move('U')
    temp_cube.apply_move('F')
    
    print(f"\nScrambled state for testing: {''.join(temp_cube.current_state)}")
    print(f"Is solved: {temp_cube.is_solved(''.join(temp_cube.current_state))}")

    print("\nAttempting to solve scrambled cube with BFS:")
    solution = temp_cube.solve()
    print("Solution steps:")
    for step in solution:
        print(step)
    
    # Verify the solution by applying the steps
    # The solve method already leaves the cube in the solved state if successful
    # So we just need to check the current_state of temp_cube
    print(f"Final state after applying solution: {''.join(temp_cube.current_state)}")
    print(f"Is solved: {temp_cube.is_solved(''.join(temp_cube.current_state))}")

    # --- Additional simple test case for BFS ---
    print("\n--- Testing a simple scramble for BFS ---")
    # A simple scramble: F'
    # Solved: WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB
    # F' applied to solved: WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB
    # U[6,7,8] -> L[2,5,8] -> D[0,1,2] -> R[0,3,6] -> U[6,7,8]
    # U: W W W W W W G G G
    # R: Y R R Y R R Y R R
    # F: W W W W W W W W W
    # D: G G G Y Y Y Y Y Y
    # L: O O W O O W O O W
    # B: B B B B B B B B B

    # Let's create a cube that is one move away from solved (e.g., F)
    one_move_scramble_state = 'WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB'
    test_cube_one_move = RubiksCube(state=one_move_scramble_state)
    test_cube_one_move.apply_move('F') # Scramble it with one F move
    
    print(f"\nScrambled state (one move F): {''.join(test_cube_one_move.current_state)}")
    print(f"Is solved: {test_cube_one_move.is_solved(''.join(test_cube_one_move.current_state))}")

    print("\nAttempting to solve with BFS (expecting F'):")
    solution_one_move = test_cube_one_move.solve()
    print("Solution steps:")
    for step in solution_one_move:
        print(step)
    
    print(f"Final state after applying solution: {''.join(test_cube_one_move.current_state)}")
    print(f"Is solved: {test_cube_one_move.is_solved(''.join(test_cube_one_move.current_state))}")

    # --- Test case for invalid input ---
    print("\n--- Testing invalid input ---")
    try:
        invalid_cube = RubiksCube(state="WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBBX") # Invalid length
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        invalid_cube = RubiksCube(state="WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBA") # Invalid color
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        invalid_cube = RubiksCube(state="WWWWWWWWWRRRRRRRRRGGGGGGGGGYWWWWWWWWWOOOOOOOOOBBBBBBBBB") # Incorrect center color (D face)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        invalid_cube = RubiksCube(state="WWWWWWWWWRRRRRRRRRGGGGGGGGGYWWWWWWWWWOOOOOOOOOBBBBBBBBB") # Incorrect center color (D face)
        # A state with incorrect counts, e.g., 10 W's and 8 R's
        invalid_counts_state = 'WWWWWWWWWW RRRRRRRR GGGGGGGGG YYYYYYYYY OOOOOOOOO BBBBBBBBB'
        invalid_cube_counts = RubiksCube(state=invalid_counts_state.replace(" ", ""))
    except ValueError as e:
        print(f"Caught expected error: {e}")
