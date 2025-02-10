import matplotlib.pyplot as plt
import pygame

from MolecularCanvas.canvas import Canvas

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DOT_DENSITY = 50  # Defines how many points to store per stroke

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Paintboard")
screen.fill(WHITE)
clock = pygame.time.Clock()

# Stroke storage
strokes = []
current_stroke = []

def save_strokes(fp: str = './paint.pdb'):
    """Convert strokes to 3D coordinates and visualize them."""
    canvas = Canvas()
    figure = canvas.fig

    # Select figure to plot
    plt.figure(figure.number)

    # Copy implementation of draw_matplotlib but to figure
    for stroke in strokes:
        x_vals, y_vals = zip(*stroke)
        plt.plot(x_vals, y_vals)

    canvas.save(name="paint", output_path=fp, scale=1, orientation=(-180, 0, 0))

def draw_matplotlib():
    """Visualize strokes using Matplotlib."""
    plt.figure()
    for stroke in strokes:
        x_vals, y_vals = zip(*stroke)
        plt.plot(x_vals, y_vals)
    plt.gca().invert_yaxis()  # Match pygame coordinate system

def redraw_canvas():
    """Redraw the canvas after undo or clear."""
    screen.fill(WHITE)
    for stroke in strokes:
        for x, y in stroke:
            pygame.draw.circle(screen, BLACK, (x, y), 2)

running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            current_stroke = []
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            if current_stroke:
                strokes.append(current_stroke)
                draw_matplotlib()
        elif event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            is_ctrl_pressed = mods & pygame.KMOD_CTRL  # Windows/Linux Ctrl
            is_cmd_pressed = mods & pygame.KMOD_META  # macOS Cmd

            if event.key == pygame.K_s:
                if is_ctrl_pressed or is_cmd_pressed:
                    save_strokes()
                else:
                    plt.show()
            elif event.key == pygame.K_z and (is_ctrl_pressed or is_cmd_pressed):  # Undo
                if strokes:
                    print("Undo")
                    strokes.pop()
                    redraw_canvas()
            elif event.key == pygame.K_r and (is_ctrl_pressed or is_cmd_pressed):  # Redo (Clear)
                print("Clear")
                strokes.clear()
                redraw_canvas()

    if drawing:
        x, y = pygame.mouse.get_pos()
        current_stroke.append((x, y))
        pygame.draw.circle(screen, BLACK, (x, y), 2)

    pygame.display.flip()
    clock.tick(240)

pygame.quit()
