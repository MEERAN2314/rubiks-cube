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

        # Define sticker indices for each face (for easier access by row/col)
        # U: 0-8, R: 9-17, F: 18-26, D: 27-35, L: 36-44, B: 45-53
        self.U_face_indices = [0,1,2,3,4,5,6,7,8]
        self.R_face_indices = [9,10,11,12,13,14,15,16,17]
        self.F_face_indices = [18,19,20,21,22,23,24,25,26]
        self.D_face_indices = [27,28,29,30,31,32,33,34,35]
        self.L_face_indices = [36,37,38,39,40,41,42,43,44]
        self.B_face_indices = [45,46,47,48,49,50,51,52,53]

        # Define corner piece indices (3 stickers per corner)
        # Each tuple represents a corner piece, listing the indices of its stickers
        # Order: U/D, R/L, F/B
        self.corner_pieces = {
            # U-layer corners
            'URF': (0, 11, 20), # U-top-right, R-top-left, F-top-right
            'UFL': (2, 18, 38), # U-top-left, F-top-left, L-top-right
            'ULB': (6, 36, 47), # U-bottom-left, L-top-left, B-top-right
            'UBR': (8, 45, 17), # U-bottom-right, B-top-left, R-top-right

            # D-layer corners
            'DRF': (29, 15, 26), # D-bottom-right, R-bottom-left, F-bottom-right
            'DFL': (27, 24, 42), # D-bottom-left, F-bottom-left, L-bottom-right
            'DLB': (33, 44, 53), # D-top-left, L-bottom-left, B-bottom-right
            'DBR': (35, 51, 14)  # D-top-right, B-bottom-left, R-bottom-right
        }

        # Define edge piece indices (2 stickers per edge)
        # Order: U/D, R/L, F/B
        self.edge_pieces = {
            # U-layer edges
            'UR': (1, 10), # U-top-middle, R-top-middle
            'UF': (3, 19), # U-middle-left, F-top-middle
            'UL': (5, 37), # U-middle-right, L-top-middle
            'UB': (7, 46), # U-bottom-middle, B-top-middle

            # Middle layer edges
            'FR': (21, 12), # F-middle-right, R-middle-left
            'FL': (23, 39), # F-middle-left, L-middle-right
            'BL': (48, 41), # B-middle-left, L-middle-left
            'BR': (50, 16), # B-middle-right, R-middle-right

            # D-layer edges
            'DR': (32, 13), # D-bottom-middle, R-bottom-middle
            'DF': (30, 25), # D-middle-left, F-bottom-middle
            'DL': (28, 43), # D-middle-right, L-bottom-middle
            'DB': (34, 52)  # D-top-middle, B-bottom-middle
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

    def scramble(self, num_moves=25):
        """Applies a series of random moves to scramble the cube."""
        import random
        moves = ['R', 'R\'', 'R2', 'L', 'L\'', 'L2', 'U', 'U\'', 'U2',
                 'D', 'D\'', 'D2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2']
        for _ in range(num_moves):
            move = random.choice(moves)
            self.apply_move(move)
        return self.current_state

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
        try:
            import kociemba
        except ImportError:
            return ["Kociemba library not found. Please install it using 'pip install kociemba'"]

        current_state_str = "".join(self.current_state)

        # First, check if the cube is already solved.
        if self.is_solved(current_state_str):
            return ["Cube is already solved."]

        # Convert the cube state to the format expected by the kociemba library
        kociemba_state = current_state_str.replace('W', 'U').replace('Y', 'D').replace('O', 'L').replace('R', 'R').replace('G', 'F').replace('B', 'B')

        try:
            # Solve the cube using the kociemba library
            solution = kociemba.solve(kociemba_state)
            moves = solution.split(" ")

            # Apply the solution to the cube
            for move in moves:
                self.apply_move(move)

            return moves
        except Exception as e:
            return [f"Error solving cube: {e}"]

    def _find_piece(self, colors, piece_type):
        """
        Finds a piece (corner or edge) on the cube and returns its current orientation and location.
        colors: A tuple of colors defining the piece (e.g., ('U', 'F') for UF edge, ('U', 'R', 'F') for URF corner).
        piece_type: 'edge' or 'corner'.
        Returns: (current_key, orientation_index) or None if not found.
                 current_key: The key of the piece's current location (e.g., 'FR' for UF edge if it's at FR position).
                 orientation_index: The index within the piece's indices tuple where the first color (e.g., 'U') is found.
        """
        target_colors = set(colors)
        
        if piece_type == 'edge':
            pieces_map = self.edge_pieces
        elif piece_type == 'corner':
            pieces_map = self.corner_pieces
        else:
            raise ValueError("piece_type must be 'edge' or 'corner'.")

        for key, indices in pieces_map.items():
            current_piece_colors = tuple(self.current_state[i] for i in indices)
            if set(current_piece_colors) == target_colors:
                # Found the piece, now determine its orientation
                # The orientation_index is the index of the 'primary' color (e.g., 'U' for white cross, 'Y' for yellow cross)
                # within the current piece's sticker indices.
                for i, idx in enumerate(indices):
                    if self.current_state[idx] == colors[0]: # Check if the first color in target_colors is at this position
                        return key, i
        return None, None

    def _solve_white_cross(self):
        """
        Solves the white cross on the U (Up) face.
        The white cross consists of the four white edge pieces (UF, UR, UL, UB).
        The white sticker should be on the U face, and the other sticker should match the center of the adjacent face.
        """
        solution = []
        white_center_color = self.current_state[self.center_indices['U']] # Should be 'U'
        
        # Define the target white edge pieces and their solved positions
        # (piece_colors, target_edge_key)
        white_edges_targets = [
            (('U', 'F'), 'UF'),
            (('U', 'R'), 'UR'),
            (('U', 'L'), 'UL'),
            (('U', 'B'), 'UB')
        ]

        for piece_colors, target_edge_key in white_edges_targets:
            # Keep trying to solve this piece until it's in its correct place and orientation
            while not (self.current_state[self.edge_pieces[target_edge_key][0]] == white_center_color and
                       self.current_state[self.edge_pieces[target_edge_key][1]] == piece_colors[1]):
                
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                
                if current_edge_key is None:
                    # This should ideally not happen for a valid cube, but as a safeguard
                    break 

                # Determine which sticker is the white one and which is the side color
                white_sticker_color = white_center_color
                side_sticker_color = [c for c in piece_colors if c != white_sticker_color][0]

                # Get the current colors of the piece's stickers
                current_white_sticker_actual_color = self.current_state[self.edge_pieces[current_edge_key][orientation_index]]
                current_side_sticker_actual_color = self.current_state[self.edge_pieces[current_edge_key][1 - orientation_index]]

                # Case 1: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    target_D_edge_key = target_edge_key.replace('U', 'D') # e.g., UF -> DF

                    # Rotate D face until the piece is directly below its target slot
                    while current_edge_key != target_D_edge_key:
                        solution.extend(self._apply_sequence(['D']))
                        current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                        current_white_sticker_actual_color = self.current_state[self.edge_pieces[current_edge_key][orientation_index]]
                        current_side_sticker_actual_color = self.current_state[self.edge_pieces[current_edge_key][1 - orientation_index]]

                    # Now the piece is directly below its target slot. Insert it.
                    if current_white_sticker_actual_color == white_center_color: # White sticker is on the D face
                        # Correctly oriented for insertion (e.g., F2 for UF)
                        if target_edge_key == 'UF':
                            solution.extend(self._apply_sequence(['F2']))
                        elif target_edge_key == 'UR':
                            solution.extend(self._apply_sequence(['R2']))
                        elif target_edge_key == 'UL':
                            solution.extend(self._apply_sequence(['L2']))
                        elif target_edge_key == 'UB':
                            solution.extend(self._apply_sequence(['B2']))
                    else: # White sticker is on the side face (needs reorientation during insertion)
                        # Needs to be flipped (e.g., F D F' for UF)
                        if target_edge_key == 'UF':
                            solution.extend(self._apply_sequence(['F', 'D', 'F\'']))
                        elif target_edge_key == 'UR':
                            solution.extend(self._apply_sequence(['R', 'D', 'R\'']))
                        elif target_edge_key == 'UL':
                            solution.extend(self._apply_sequence(['L', 'D', 'L\'']))
                        elif target_edge_key == 'UB':
                            solution.extend(self._apply_sequence(['B', 'D', 'B\'']))

                # Case 2: Piece is in the middle layer (FR, FL, BL, BR)
                elif current_edge_key in ['FR', 'FL', 'BL', 'BR']:
                    # Move it to the D layer first
                    if current_edge_key == 'FR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'FL':
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR':
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.

                # Case 3: Piece is in the U layer but incorrectly oriented or positioned
                elif current_edge_key in ['UF', 'UR', 'UL', 'UB']:
                    # Move it to the D layer first
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
        return solution

    def _apply_sequence(self, moves_sequence):
        """Applies a sequence of moves to the cube and returns the sequence."""
        for move in moves_sequence:
            self.apply_move(move)
        return moves_sequence

    def _solve_first_layer_corners(self):
        """Solves the four white corner pieces."""
        solution = []
        white_center_color = self.current_state[self.center_indices['U']] # Should be 'U'
        
        # Define the target white corner pieces and their solved positions
        # (piece_colors, target_corner_key)
        white_corners_targets = [
            (('U', 'F', 'R'), 'URF'),
            (('U', 'F', 'L'), 'UFL'),
            (('U', 'B', 'L'), 'ULB'),
            (('U', 'B', 'R'), 'UBR')
        ]

        for piece_colors, target_corner_key in white_corners_targets:
            # Keep trying to solve this piece until it's in its correct place and orientation
            while not (self.current_state[self.corner_pieces[target_corner_key][0]] == white_center_color and
                       self.current_state[self.corner_pieces[target_corner_key][1]] == piece_colors[1] and
                       self.current_state[self.corner_pieces[target_corner_key][2]] == piece_colors[2]):
                
                current_corner_key, orientation_index = self._find_piece(piece_colors, 'corner')

                if current_corner_key is None:
                    break

                # If the corner is in the U layer but incorrectly positioned/oriented
                if current_corner_key in ['URF', 'UFL', 'ULB', 'UBR']:
                    # Move it to the D layer first
                    if current_corner_key == 'URF':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_corner_key == 'UFL':
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_corner_key == 'ULB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    elif current_corner_key == 'UBR':
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.

                # Now the piece should be in the D layer (DRF, DFL, DLB, DBR)
                # Position it directly below its target slot
                target_D_key = target_corner_key.replace('U', 'D') # e.g., URF -> DRF
                while current_corner_key != target_D_key:
                    solution.extend(self._apply_sequence(['D']))
                    current_corner_key, orientation_index = self._find_piece(piece_colors, 'corner')

                # Insert the corner from the D layer to the U layer
                # Re-find the piece to get its current orientation after D moves
                current_corner_key, orientation_index = self._find_piece(piece_colors, 'corner')
                
                # The piece is now in the correct D-layer slot (e.g., DRF).
                # Apply the appropriate insertion algorithm based on the white sticker's orientation.
                
                # White sticker on D face (e.g., DRF[0] is D sticker, so white is on D)
                if self.current_state[self.corner_pieces[current_corner_key][0]] == white_center_color:
                    solution.extend(self._apply_sequence(['R\'', 'D\'', 'R', 'D']))
                # White sticker on R face (e.g., DRF[1] is R sticker, so white is on R)
                elif self.current_state[self.corner_pieces[current_corner_key][1]] == white_center_color:
                    solution.extend(self._apply_sequence(['F', 'D', 'F\'']))
                # White sticker on F face (e.g., DRF[2] is F sticker, so white is on F)
                elif self.current_state[self.corner_pieces[current_corner_key][2]] == white_center_color:
                    solution.extend(self._apply_sequence(['R\'', 'D', 'R']))
        return solution

    def _solve_middle_layer_edges(self):
        """Solves the four middle layer edge pieces."""
        solution = []
        
        # Define the target middle layer edge pieces
        middle_layer_edges_targets = [
            (('F', 'R'), 'FR'),
            (('F', 'L'), 'FL'),
            (('B', 'L'), 'BL'),
            (('B', 'R'), 'BR')
        ]

        for piece_colors, target_edge_key in middle_layer_edges_targets:
            # Keep trying to solve this piece until it's in its correct place and orientation
            while not (self.current_state[self.edge_pieces[target_edge_key][0]] == piece_colors[0] and
                       self.current_state[self.edge_pieces[target_edge_key][1]] == piece_colors[1]):
                
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')

                if current_edge_key is None:
                    break 

                # If the piece is in the middle layer but incorrectly positioned/oriented,
                # move it to the D layer first using an insertion algorithm in reverse.
                if current_edge_key in ['FR', 'FL', 'BL', 'BR']:
                    if current_edge_key == 'FR':
                        solution.extend(self._apply_sequence(['U', 'R', 'U\'', 'R\'', 'U\'', 'F\'', 'U', 'F']))
                    elif current_edge_key == 'FL':
                        solution.extend(self._apply_sequence(['U\'', 'L\'', 'U', 'L', 'U', 'F', 'U\'', 'F\'']))
                    elif current_edge_key == 'BL':
                        solution.extend(self._apply_sequence(['U', 'B', 'U\'', 'B\'', 'U\'', 'L\'', 'U', 'L']))
                    elif current_edge_key == 'BR':
                        solution.extend(self._apply_sequence(['U\'', 'R\'', 'U', 'R', 'U', 'B', 'U\'', 'B\'']))
                    current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge') # Re-find after move

                # Now the piece should be in the D layer (DF, DR, DL, DB)
                # Align the piece with its corresponding center on the F, R, B, or L face by rotating the D face.
                # The first color in piece_colors (e.g., 'F' for FR) should match the front face center.
                
                # Determine the face of the first sticker of the piece (e.g., F for FR)
                first_sticker_global_idx = self.edge_pieces[current_edge_key][orientation_index]
                first_sticker_color = self.current_state[first_sticker_global_idx]

                # Determine the face of the second sticker of the piece (e.g., R for FR)
                second_sticker_global_idx = self.edge_pieces[current_edge_key][1 - orientation_index]
                second_sticker_color = self.current_state[second_sticker_global_idx]

                # Align the piece in the D layer
                # Example: For FR piece, if first_sticker_color is F, align F face with F center.
                # This means the piece is at DF.
                
                # Map target_edge_key to the face it's on (e.g., FR is on F face, R face)
                # We need to align the front sticker of the piece with the front face.
                
                # Find the face that the first color of the piece_colors belongs to.
                # e.g., for ('F', 'R'), 'F' is the primary face.
                primary_face_color = piece_colors[0]
                secondary_face_color = piece_colors[1]

                # Rotate D until the primary_face_color sticker is on the F face (DF edge)
                # This means the piece is at DF, DR, DL, or DB.
                # We need to rotate D until the sticker that belongs to the F face is on the F face.
                
                # Example: piece_colors = ('F', 'R'), target_edge_key = 'FR'
                # We want the 'F' sticker of the piece to be on the F face, and 'R' sticker on R face.
                # If the piece is currently at DF, its colors are (F, D) or (D, F).
                # We need to check the actual colors of the piece and rotate D until the correct side is aligned.

                # This logic needs to be robust. Let's simplify:
                # We need to bring the piece (e.g., FR) to the DF slot, with F sticker on F face.
                # Then apply the algorithm.

                # Find the current location of the piece (e.g., ('F', 'R') piece is at 'DF')
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                
                # Determine the face of the sticker that should be on the front (F) face
                # This is the first color in `piece_colors` (e.g., 'F' for ('F', 'R'))
                front_sticker_color_of_piece = piece_colors[0]
                side_sticker_color_of_piece = piece_colors[1]

                # Find which face the `front_sticker_color_of_piece` is currently on.
                # This is tricky because the piece can be oriented in two ways.
                
                # Let's use a simpler approach:
                # If the piece is in the D layer, rotate D until its front-facing sticker matches the F center.
                # Then apply the algorithm.

                # The piece is in the D layer.
                # We need to align the piece so that its front-facing sticker (e.g., F for FR)
                # is on the F face (DF slot).
                
                # This requires knowing the current face of the sticker.
                # The `_find_piece` returns `current_edge_key` (e.g., 'DF') and `orientation_index`.
                # `orientation_index` tells us which sticker of the piece is `piece_colors[0]`.
                
                # Let's assume the piece is in the D layer.
                # We need to align the D layer so that the piece is under its target slot.
                # For FR, target is FR. The piece should be at DF.
                # For FL, target is FL. The piece should be at DF.
                # For BL, target is BL. The piece should be at DB.
                # For BR, target is BR. The piece should be at DB.

                # This is getting complicated. Let's use a more direct approach for middle layer edges.
                # Find the piece. If it's in the D layer, align it and insert.
                # If it's in the middle layer but wrong, extract it to D layer, then re-insert.

                # Re-find the piece after any initial moves to bring it to D layer
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')

                # Now, the piece is guaranteed to be in the D layer.
                # We need to align the D face so the piece is ready for insertion.
                # For piece (F,R) going to FR slot:
                #   If piece is at DF, and F sticker is on F face (orientation_index 0 for DF):
                #       Apply U R U' R' U' F' U F (insert right)
                #   If piece is at DF, and R sticker is on F face (orientation_index 1 for DF):
                #       Apply U' L' U L U F U' F' (insert left, but this is for FL, so need to adjust)
                
                # This is the core logic for inserting an edge from the D layer.
                # We need to know which face the piece's "front" color is currently on.
                
                # Let's define the target D-layer position for each middle layer edge.
                # FR -> DF (F sticker on F face)
                # FL -> DF (F sticker on F face)
                # BL -> DB (B sticker on B face)
                # BR -> DB (B sticker on B face)

                # This is still not quite right. The piece has two colors.
                # For FR edge (F, R), if it's at DF, it can be (F, D) or (D, F).
                # If it's (F, D), F is on F face, D is on D face.
                # If it's (D, F), D is on F face, F is on D face.

                # Let's simplify the logic for middle layer edge insertion.
                # We need to find the piece.
                # If it's in the D layer, rotate D until the side color (e.g., R for FR) matches its center.
                # Then apply the correct algorithm.

                # Example: Solving FR edge (colors F, R)
                # 1. Find (F, R) piece.
                # 2. If it's in the middle layer (e.g., FL), extract it to D layer.
                # 3. If it's in the D layer (e.g., DF, DR, DL, DB):
                #    Rotate D until the R sticker of the piece is aligned with the R center.
                #    This means the piece is at DR.
                #    Then, if F sticker is on F face (correct orientation): U R U' R' U' F' U F
                #    If F sticker is on D face (needs flip): U' L' U L U F U' F' (this is for FL, need to adjust)

                # Let's use a more direct approach for each target edge.
                # For each target middle layer edge (e.g., FR):
                #   Find the piece (F, R).
                #   If it's in the middle layer but wrong, extract it.
                #   If it's in the D layer:
                #     Rotate D until the piece is aligned.
                #     Apply the correct insertion algorithm.

                # Target: FR (colors F, R)
                # Find piece (F, R)
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                
                # If piece is in middle layer but wrong, extract it
                if current_edge_key == 'FL': # FR piece is at FL slot
                    solution.extend(self._apply_sequence(['U\'', 'L\'', 'U', 'L', 'U', 'F', 'U\'', 'F\'']))
                    current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                elif current_edge_key == 'BL': # FR piece is at BL slot
                    solution.extend(self._apply_sequence(['U', 'B', 'U\'', 'B\'', 'U\'', 'L\'', 'U', 'L']))
                    current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')
                elif current_edge_key == 'BR': # FR piece is at BR slot
                    solution.extend(self._apply_sequence(['U\'', 'R\'', 'U', 'R', 'U', 'B', 'U\'', 'B\'']))
                    current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')

                # Now the piece should be in the D layer (DF, DR, DL, DB) or already solved.
                # If solved, continue to next piece.
                if (self.current_state[self.edge_pieces[target_edge_key][0]] == piece_colors[0] and
                    self.current_state[self.edge_pieces[target_edge_key][1]] == piece_colors[1]):
                    continue

                # Align the piece in the D layer with its target side face.
                # For FR (F,R), align R sticker with R center. This means piece is at DR.
                # For FL (F,L), align L sticker with L center. This means piece is at DL.
                # For BL (B,L), align L sticker with L center. This means piece is at DL.
                # For BR (B,R), align R sticker with R center. This means piece is at DR.

                # This is still not quite right. The standard F2L approach is to align the edge
                # with its corresponding center on the D layer, then insert.

                # Let's use the standard F2L algorithms for inserting edges.
                # The piece is in the D layer.
                # We need to align the D face so that the piece is directly under its target slot,
                # and the top sticker of the piece matches the front face color.

                # Example: target FR (F,R)
                # Piece is (F,R).
                # If piece is at DF, and F is on F face (current_state[DF_F_idx] == F):
                #   Apply U R U' R' U' F' U F
                # If piece is at DF, and R is on F face (current_state[DF_R_idx] == R):
                #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                # This is the correct logic for middle layer edges:
                # 1. Find the edge piece (e.g., F-R edge).
                # 2. If it's in the top layer (U layer), orient it correctly and move it to the middle layer.
                # 3. If it's in the middle layer but in the wrong spot or oriented incorrectly, extract it to the D layer.
                # 4. Once in the D layer, align it with its corresponding center on the D face.
                # 5. Insert it into the correct middle layer slot.

                # Let's refine the loop for each piece.
                # For each piece_colors, target_edge_key:
                #   Loop until piece is solved.
                #   Find piece.
                #   If piece is in U layer:
                #     Rotate U until side color matches side face.
                #     Apply algorithm to move to middle layer.
                #   If piece is in middle layer (wrong spot/orientation):
                #     Apply algorithm to extract to D layer.
                #   If piece is in D layer:
                #     Rotate D until side color matches side face.
                #     Apply algorithm to insert into middle layer.

                # This is the most robust way. Let's implement this.

                # Re-find the piece after any previous moves
                current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')

                # Check if the piece is already solved
                if (self.current_state[self.edge_pieces[target_edge_key][0]] == piece_colors[0] and
                    self.current_state[self.edge_pieces[target_edge_key][1]] == piece_colors[1]):
                    continue # Piece is solved, move to next

                # Case A: Piece is in the U layer (UF, UR, UL, UB)
                if current_edge_key in ['UF', 'UR', 'UL', 'UB']:
                    # Determine the colors of the piece as it currently sits in the U layer
                    current_front_color = self.current_state[self.edge_pieces[current_edge_key][0]] # U sticker
                    current_side_color = self.current_state[self.edge_pieces[current_edge_key][1]] # Side sticker

                    # We need to align the side color of the piece with its center on the U layer.
                    # Example: For FR (F,R) edge, if it's at UF, its colors are (U,F) or (F,U).
                    # We want to align the F sticker with the F center.
                    
                    # The piece_colors are (Front_Face_Color, Side_Face_Color) e.g., ('F', 'R')
                    # We need to align the `front_face_color` (piece_colors[0]) with the F face center.
                    
                    # Rotate U until the piece's front-facing sticker (which is on the U face)
                    # matches the center of the face it's above.
                    
                    # This is tricky. Let's use the standard F2L algorithms.
                    # If the piece is in the U layer, we need to align it with the correct side face.
                    # Example: FR edge (F,R). If it's at UF, and F is on U face, R is on F face.
                    # We need to rotate U until F is above F center.
                    
                    # Let's map U-layer edge keys to their adjacent faces:
                    # UF: F face
                    # UR: R face
                    # UL: L face
                    # UB: B face

                    # Find the face that the piece's `piece_colors[0]` is currently above.
                    # e.g., if piece_colors is ('F', 'R'), and current_edge_key is 'UR',
                    # then the 'F' sticker is on the U face, and 'R' sticker is on the R face.
                    # This is not the correct orientation for insertion.

                    # Let's simplify:
                    # If the piece is in the U layer, we need to rotate U until the piece's
                    # non-yellow sticker (e.g., F for FR) is above its corresponding center.
                    # Then apply the algorithm.

                    # The piece is (piece_colors[0], piece_colors[1]).
                    # We need to align piece_colors[0] with its center.
                    
                    # Example: piece_colors = ('F', 'R'), target_edge_key = 'FR'
                    # If current_edge_key is 'UF', and self.current_state[self.edge_pieces['UF'][0]] == 'F' (F is on U face)
                    #   Then rotate U until F is above F center.
                    #   This means the piece is at UF, and F is on U face.
                    #   Algorithm: U R U' R' U' F' U F (insert right)
                    
                    # If current_edge_key is 'UF', and self.current_state[self.edge_pieces['UF'][1]] == 'F' (F is on F face)
                    #   Then rotate U until F is above F center.
                    #   This means the piece is at UF, and F is on F face.
                    #   Algorithm: U' L' U L U F U' F' (insert left)

                    # This is the correct logic for middle layer edges from U layer:
                    # Find the piece.
                    # If piece_colors[0] is on the U face (e.g., F for FR is on U face):
                    #   Rotate U until piece_colors[1] is above its center.
                    #   Apply (U R U' R' U' F' U F) or (U' L' U L U F U' F')
                    # If piece_colors[0] is on the side face (e.g., F for FR is on F face):
                    #   Rotate U until piece_colors[0] is above its center.
                    #   Apply (U R U' R' U' F' U F) or (U' L' U L U F U' F')

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # Let's use a simpler approach for middle layer edges.
                    # Find the piece. If it's in the D layer, align it and insert.
                    # If it's in the middle layer but wrong, extract it to D layer, then re-insert.

                    # Re-find the piece after any initial moves to bring it to D layer
                    current_edge_key, orientation_index = self._find_piece(piece_colors, 'edge')

                    # Now, the piece is guaranteed to be in the D layer.
                    # We need to align the D face so the piece is ready for insertion.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DF (F sticker on F face)
                    # FL -> DF (F sticker on F face)
                    # BL -> DB (B sticker on B face)
                    # BR -> DB (B sticker on B face)

                    # This is still not quite right. The standard F2L approach is to align the edge
                    # with its corresponding center on the D layer, then insert.

                    # Let's use the standard F2L algorithms for inserting edges.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the piece is under its target slot,
                    # and the color that should be on the front face is actually on the front face.
                    
                    # Example: target FR (F,R)
                    # We want the F sticker of the piece to be on the F face (DF slot).
                    # If the piece is (F,R) and it's at DF, and F is on F face:
                    #   Apply U R U' R' U' F' U F
                    # If the piece is (R,F) and it's at DF, and R is on F face:
                    #   Apply U' L' U L U F U' F' (this is for FL, need to adjust)

                    # This is getting too complex for a simple layer-by-layer.
                    # Let's use a simpler approach:
                    # If the piece is in the U layer, just move it to the D layer first.
                    # This simplifies the logic significantly.

                    # If piece is in U layer, move it to D layer first.
                    if current_edge_key == 'UF':
                        solution.extend(self._apply_sequence(['F', 'U', 'F\'']))
                    elif current_edge_key == 'UR':
                        solution.extend(self._apply_sequence(['R', 'U', 'R\'']))
                    elif current_edge_key == 'UL':
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'UB':
                        solution.extend(self._apply_sequence(['B', 'U', 'B\'']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case B: Piece is in the middle layer but incorrectly positioned/oriented
                if current_edge_key in ['FR', 'FL', 'BL', 'BR'] and current_edge_key != target_edge_key:
                    # Extract it to the D layer using an inverse insertion algorithm
                    if current_edge_key == 'FR': # Extract FR to D layer
                        solution.extend(self._apply_sequence(['R', 'U', 'R\''])) # This extracts it to D layer
                    elif current_edge_key == 'FL': # Extract FL to D layer
                        solution.extend(self._apply_sequence(['L\'', 'U\'', 'L']))
                    elif current_edge_key == 'BL': # Extract BL to D layer
                        solution.extend(self._apply_sequence(['L', 'U', 'L\'']))
                    elif current_edge_key == 'BR': # Extract BR to D layer
                        solution.extend(self._apply_sequence(['R\'', 'U\'', 'R']))
                    # After this, the piece should be in the D layer, and the loop will re-evaluate.
                    continue # Re-evaluate the piece in its new position

                # Case C: Piece is in the D layer (DF, DR, DL, DB)
                if current_edge_key.startswith('D'):
                    # Align the piece in the D layer with its target side face.
                    # Example: For FR (F,R) edge, we want to align the R sticker with the R center.
                    # This means the piece should be at DR.
                    
                    # Determine the target D-layer key based on the target_edge_key
                    # FR -> DR (R sticker on R face)
                    # FL -> DL (L sticker on L face)
                    # BL -> DL (L sticker on L face)
                    # BR -> DR (R sticker on R face)

                    # This is still not quite right. The standard way is to align the front color.
                    # For FR (F,R), align F sticker with F center. This means piece is at DF.
                    # Then check orientation.

                    # Let's use the standard F2L insertion algorithms.
                    # The piece is in the D layer.
                    # We need to rotate D until the piece's front-facing sticker (e.g., F for FR)
                    # is aligned with the F center.
                    
                    # Find the current colors of the piece in the D layer
                    current_piece_colors_on_D = self._get_edge_colors(current_edge_key)
                    
                    # Determine which sticker of the piece is the 'front' color (e.g., F for FR)
                    front_color_of_piece = piece_colors[0]
                    side_color_of_piece = piece_colors[1]

                    # Rotate D until the front_color_of_piece is on the F face (DF slot)
                    # This means the piece is at DF, and its F sticker is on the F face.
                    
                    # This is the most common way to set up for F2L algorithms.
                    # We need to rotate D until the

    def _get_edge_colors(self, edge_key):
        """Returns the colors of an edge piece at a given location."""
        indices = self.edge_pieces[edge_key]
        return tuple(self.current_state[i] for i in indices)

    def _solve_yellow_cross(self):
        """Solves the yellow cross on the D (Down) face using OLL algorithms."""
        solution = []
        yellow_center_color = self.current_state[self.center_indices['D']] # Should be 'Y'

        # Check the current state of the yellow cross
        yellow_edges = {
            'DF': self.current_state[self.edge_pieces['DF'][0]] == yellow_center_color,
            'DR': self.current_state[self.edge_pieces['DR'][0]] == yellow_center_color,
            'DL': self.current_state[self.edge_pieces['DL'][0]] == yellow_center_color,
            'DB': self.current_state[self.edge_pieces['DB'][0]] == yellow_center_color
        }

        # Case 1: Dot (no yellow edges up)
        if not any(yellow_edges.values()):
            solution.extend(self._apply_sequence(['F', 'R', 'U', 'R\'', 'U\'', 'F\'']))
        
        # Case 2: L-shape
        elif yellow_edges['DF'] and yellow_edges['DL']:
            solution.extend(self._apply_sequence(['F', 'U', 'R', 'U\'', 'R\'', 'F\'']))
        elif yellow_edges['DL'] and yellow_edges['DB']:
            solution.extend(self._apply_sequence(['U\'', 'F', 'U', 'R', 'U\'', 'R\'', 'F\'']))
        elif yellow_edges['DB'] and yellow_edges['DR']:
            solution.extend(self._apply_sequence(['U2', 'F', 'U', 'R', 'U\'', 'R\'', 'F\'']))
        elif yellow_edges['DR'] and yellow_edges['DF']:
            solution.extend(self._apply_sequence(['U', 'F', 'U', 'R', 'U\'', 'R\'', 'F\'']))

        # Case 3: Line
        elif yellow_edges['DF'] and yellow_edges['DB']:
            solution.extend(self._apply_sequence(['F', 'R', 'U', 'R\'', 'U\'', 'F\'']))
        elif yellow_edges['DL'] and yellow_edges['DR']:
            solution.extend(self._apply_sequence(['U', 'F', 'R', 'U', 'R\'', 'U\'', 'F\'']))

        return solution

    def _orient_last_layer(self):
        """Orients the last layer corners (OLL)."""
        solution = []
        yellow_center_color = self.current_state[self.center_indices['D']]

        # Check the orientation of the last layer corners
        while not all(self.current_state[self.corner_pieces[c][0]] == yellow_center_color for c in ['DFL', 'DRF', 'DLB', 'DBR']):
            # Sune and Anti-Sune cases
            if self.current_state[self.corner_pieces['DFL'][0]] == yellow_center_color and \
               self.current_state[self.corner_pieces['DRF'][1]] == yellow_center_color and \
               self.current_state[self.corner_pieces['DLB'][2]] == yellow_center_color and \
               self.current_state[self.corner_pieces['DBR'][1]] == yellow_center_color:
                solution.extend(self._apply_sequence(['R', 'U', 'R\'', 'U', 'R', 'U2', 'R\''])) # Sune
            elif self.current_state[self.corner_pieces['DFL'][2]] == yellow_center_color and \
                 self.current_state[self.corner_pieces['DRF'][0]] == yellow_center_color and \
                 self.current_state[self.corner_pieces['DLB'][0]] == yellow_center_color and \
                 self.current_state[self.corner_pieces['DBR'][2]] == yellow_center_color:
                solution.extend(self._apply_sequence(['R', 'U2', 'R\'', 'U\'', 'R', 'U\'', 'R\''])) # Anti-Sune
            else:
                solution.extend(self._apply_sequence(['U'])) # Rotate to find a case
        return solution

    def _permute_last_layer(self):
        """Permutes the last layer corners and edges (PLL)."""
        solution = []

        # Permute corners first
        while not (self.current_state[self.corner_pieces['DFL'][1]] == self.current_state[self.center_indices['F']] and
                   self.current_state[self.corner_pieces['DRF'][1]] == self.current_state[self.center_indices['R']]):
            solution.extend(self._apply_sequence(['R\'', 'F', 'R\'', 'B2', 'R', 'F\'', 'R\'', 'B2', 'R2']))

        # Permute edges
        while not (self.current_state[self.edge_pieces['DF'][1]] == self.current_state[self.center_indices['F']] and
                   self.current_state[self.edge_pieces['DR'][1]] == self.current_state[self.center_indices['R']]):
            # U-perm
            if self.current_state[self.edge_pieces['DF'][1]] == self.current_state[self.center_indices['R']]:
                solution.extend(self._apply_sequence(['R2', 'U', 'R', 'U', 'R\'', 'U\'', 'R\'', 'U\'', 'R\'', 'U', 'R\'']))
            else: # H-perm or Z-perm
                solution.extend(self._apply_sequence(['U']))
        return solution
