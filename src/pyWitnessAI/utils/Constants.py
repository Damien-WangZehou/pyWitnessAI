import numpy as np


# Defining colors used below
colors = {
    'red': (1, 0, 0),
    'yellow': (1, 0.7, 0.1),
    'green': (0, 1, 0),
    'blue': (0, 0, 1),
    'magenta': (1, 0, 1),
    'cyan': (0, 1, 1),
    'white': (1, 1, 1),
    'black': (0, 0, 0),
    'gray': (0.49, 0.49, 0.49),
    'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
    'dark_gray': (0.2, 0.2, 0.2),
    'light_gray': (0.86, 0.86, 0.86)
}

legend_colors = {
    'general': colors['black'],
    'mean': colors['green'],
    'mtcnn': colors['blue'],
    'opencv': colors['red'],
    'fastmtcnn': colors['yellow'],
    'lineup': colors['gray']

}

line_styles = {
    'mtcnn': '-',
    'opencv': '--',
    'lineup': '-.',
    'fastmtcnn': '--',
}

def get_random_color():
    # Generate a random color in the form of an RGB tuple
    return tuple(np.random.rand(3).tolist())

def get_color_for_analyzer(analyzer_name):
    # Get color for the analyzer. If not defined, generate a random color
    if analyzer_name not in legend_colors:
        # Assign a random color if analyzer not found in legend_colors
        random_color = get_random_color()
        legend_colors[analyzer_name] = random_color
        return random_color
    return legend_colors[analyzer_name]

def get_style_for_analyzer(analyzer_name):
    # Get line style for the analyzer. If not defined, assign a default style
    if analyzer_name not in line_styles:
        # Assign default line style '-' if analyzer not found in line_styles
        line_styles[analyzer_name] = '-'
        return '-'
    return line_styles[analyzer_name]
