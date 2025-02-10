from typing import Any

import numpy as np
from matplotlib import pyplot as plt

type Coordinates = list[np.ndarray[tuple[int, ...], np.dtype]]
type Coordinate = tuple[int, int, int] | list[int]

import random
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO
from scipy.spatial.transform import Rotation


class Canvas:
    """A 3D canvas using matplotlib as the backend"""

    def __init__(self):
        self._fig: plt.Figure = plt.figure()
        self._ax: plt.Axes = self._fig.add_subplot(111, projection='3d')

    def clear(self) -> None:
        """Clear the canvas"""
        self._fig.clear()

    def draw(self, *points: Coordinate, label: str = None) -> None:
        """Plot a line on the canvas"""
        self._ax.plot(*points, label=label)

    def triangle(self, a: Coordinate, b: Coordinate, c: Coordinate) -> None:
        """Draw a triangle on the canvas using lines to connect vertices"""
        # List of vertices in sequence to create a closed loop (return to the first point)
        vertices = np.array([a, b, c, a])  # Repeating the first vertex to close the triangle
        self._ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], label='Triangle')

    def rect(self, a: Coordinate, b: Coordinate, c: Coordinate, d: Coordinate) -> None:
        """Draw a rectangle on the canvas by connecting four points"""
        self._ax.plot(*np.array([a, b, c, d, a]).T, label='Rectangle')

    def circle(self, center: Coordinate, radius: float, orientation: Coordinate, num_vertices: int = 25):
        orientation = np.array(orientation)
        norm = np.linalg.norm(orientation)

        if norm == 0:
            raise ValueError("Orientation vector cannot be zero")
        orientation = orientation / norm

        # Create a vector not parallel to the orientation vector
        if orientation[0] == 0 and orientation[1] == 0:
            not_parallel = np.array([1, 0, 0])
        else:
            not_parallel = np.array([0, 0, 1])

        # For some weird reason I get an 'unreachable' error and it screws up linting. idk. this is a hack to allow the linter to work
        if eval('True'):
            # noinspection PyUnreachableCode
            tangent1 = np.cross(orientation, not_parallel)
            tangent1 /= np.linalg.norm(tangent1)
            # noinspection PyUnreachableCode
            tangent2 = np.cross(orientation, tangent1)

        # Define circle in terms of tangent vectors
        theta = np.linspace(0, 2 * np.pi, num_vertices)
        circle_points = (radius * np.cos(theta)[:, None] * tangent1 +
                         radius * np.sin(theta)[:, None] * tangent2 +
                         np.array(center))

        self.draw(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], label='Circle')

    def show(self):
        self._fig.show()

    @property
    def coordinates(self) -> Coordinates:
        """Convert the plot to a numpy array representation of the drawn structure"""
        line_data: Coordinates = []

        for line in self._ax.lines:
            x, y, z = line.get_data_3d()
            coordinates = np.array([x, y, z])
            line_data.append(coordinates)

        return line_data

    def draw_function(self, equation: Any, width: float, height: float, center: tuple = (0, 0, 0), num_vertices: int = 50, orientation: tuple = (0, 0, 1)):
        """
        Plots a mathematical relation (curve) on the 3D canvas based on the given equation,
        ensuring that it fits within the specified width and height.

        :param equation: A string representing a mathematical equation in terms of 'x' and/or 'y'.
        :param width: The total width of the plot.
        :param height: The total height of the plot.
        :param center: The center point around which the equation is plotted.
        :param num_vertices: Number of points to approximate the curve.
        :param orientation: Direction of the curve in 3D space.
        """
        import numpy as np
        from sympy import symbols, parse_expr, lambdify, Eq, solve, N, Float

        # Define symbolic variables
        x, y = symbols('x y')
        expr = parse_expr(equation) if isinstance(equation, str) else equation

        # Check if the equation contains both x and y (implicit function)
        if y in expr.free_symbols:
            eq = Eq(expr, 0)  # Convert to equation form
            y_vals = np.linspace(-height / 2, height / 2, num_vertices)
            x_vals = []

            for yi in y_vals:
                yi_sympy = Float(yi)  # Ensure yi is a SymPy Float
                solutions = solve(eq.subs(y, yi_sympy), x)  # Solve for x in terms of y

                # Convert solutions to float and handle multiple solutions per y
                numeric_solutions = [float(N(sol)) for sol in solutions if sol.is_real]
                if numeric_solutions:
                    x_vals.append(numeric_solutions[0])  # Choose one solution (e.g., positive branch)
                else:
                    x_vals.append(np.nan)  # Handle cases where no real solution exists

            x_vals = np.array(x_vals)

        else:  # Explicit function y = f(x)
            func = lambdify(x, expr, 'numpy')
            x_vals = np.linspace(-width / 2, width / 2, num_vertices)
            y_vals = func(x_vals)

        # Normalize y-values to fit within the specified height
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
        if y_max - y_min > 0:
            y_vals = height * (y_vals - y_min) / (y_max - y_min) - height / 2
        else:
            y_vals = np.zeros_like(y_vals)

        # Adjust points based on orientation
        orientation = np.array(orientation)
        norm = np.linalg.norm(orientation)
        if norm == 0:
            raise ValueError("Orientation vector cannot be zero")

        orientation = orientation / norm  # Normalize orientation vector

        # Default perpendicular vector
        if orientation[0] == 0 and orientation[1] == 0:
            perp_vector = np.array([1, 0, 0])
        else:
            perp_vector = np.array([0, 0, 1])

        tangent = np.cross(orientation, perp_vector)
        tangent /= np.linalg.norm(tangent)

        curve_points = np.array([
            center + x_vals[i] * tangent + y_vals[i] * orientation
            for i in range(num_vertices) if not np.isnan(x_vals[i])
        ])

        # Plot the curve
        self.draw(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], label=f'Plot: {equation}')

    def save(self, name: str, output_path: str = "structure.pdb", scale: float = 1.0, orientation: tuple = (0.0, 0.0, 0.0)):
        """
        Convert the drawn 3D structure into a PDB structure and save it to a file.
        Each line is treated as a separate polypeptide chain, and all molecules within
        a line are connected. The constituent molecules are randomized.

        :param name: Name of the structure
        :param output_path: Path to save the PDB file
        :param scale: Scaling factor for coordinates
        :param orientation: Rotation angles (in degrees) around x, y, and z axes
        """
        structure = Structure.Structure(name)
        model = Model.Model(0)
        structure.add(model)

        rotation = Rotation.from_euler(seq='xyz', angles=orientation, degrees=True)
        atom_serial = 1

        possible_residues = ['ALA', 'GLY', 'SER', 'THR', 'LEU', 'VAL', 'ILE', 'PRO', 'MET']

        with open(output_path, "w") as pdb_file:
            for chain_index, coordinates in enumerate(self.coordinates):
                chain_id = chr(65 + chain_index % 26)  # Use A-Z as chain identifiers
                chain = Chain.Chain(chain_id)
                model.add(chain)

                prev_atom_index = None

                for j in range(coordinates.shape[1]):
                    scaled_coord = coordinates[:, j] * scale  # Apply scaling factor
                    rotated_coord = rotation.apply(scaled_coord)  # Apply rotation

                    if np.isnan(rotated_coord).any():  # Exclude NaN values
                        continue

                    res_name = random.choice(possible_residues)  # Randomize residue name
                    residue = Residue.Residue((' ', atom_serial, ' '), res_name, '')
                    atom = Atom.Atom(name=f'C{atom_serial}',
                                     coord=rotated_coord,
                                     bfactor=1.0, occupancy=1.0, altloc=' ',
                                     fullname=f'C{atom_serial}', serial_number=atom_serial,
                                     element='C')
                    residue.add(atom)
                    chain.add(residue)
                    pdb_file.write(f"HETATM{atom_serial:5} C   {res_name} {chain_id}{atom_serial:4}    {rotated_coord[0]:8.3f}{rotated_coord[1]:8.3f}{rotated_coord[2]:8.3f}  1.00  0.00           C  \n")

                    if prev_atom_index is not None:
                        # Create a bond between the previous and current atom
                        pdb_file.write(f"CONECT {prev_atom_index:5} {atom_serial:5}\n")

                    prev_atom_index = atom_serial
                    atom_serial += 1

                pdb_file.write("TER\n")  # Terminate each chain properly

        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)
        print(f"Structure saved to {output_path}")
