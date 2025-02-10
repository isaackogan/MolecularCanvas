from MolecularCanvas.canvas import Canvas

if __name__ == '__main__':
    canvas = Canvas()

    diff = 20
    verts = 200

    width_1 = 84

    # Define heart shape using valid function expressions
    canvas.draw_function(
        'sqrt(1-x)',  # y in terms of x
        center=(0, -diff + (-22), -10),  # Adjust for positioning
        num_vertices=verts,
        orientation=(0, 0, -1),
        width=width_1,
        height=60
    )

    canvas.draw_function(
        'sqrt(1+x)',  # y in terms of x
        center=(0, +diff + (22), -10),  # Adjust for positioning
        num_vertices=verts,
        orientation=(0, 0, -1),
        width=width_1,
        height=60
    )

    p2_verts = 60
    width = 42
    diff_2 = 22

    h_2 = 36

    # Define heart shape using valid function expressions
    canvas.draw_function(
        'x**2',  # y in terms of x
        center=(0, diff_2, h_2),  # Adjust for positioning
        num_vertices=p2_verts,
        orientation=(0, 0, -1),
        width=width,
        height=25,
    )

    canvas.draw_function(
        'x**2',  # y in terms of x
        center=(0, -diff_2, h_2),  # Adjust for positioning
        num_vertices=p2_verts,
        orientation=(0, 0, -1),
        width=width,
        height=25
    )

    # Save to a molecular structure file
    canvas.show()
    canvas.save(name='02/14', output_path='./outputs/0214.pdb', scale=0.1, orientation=(-90, 0, 0))
