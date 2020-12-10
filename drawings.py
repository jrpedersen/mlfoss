import matplotlib.pyplot as plt
import numpy as np

def draw_straight_line(start, end, num_points = 100):
    """returns the x,y's for a straight line connecting two points."""
    #xs = np.linspace(start[0],end[0],num_points)
    #ys = np.linspace(start[1],end[1],num_points)
    return np.linspace(start[0],end[0],num_points), np.linspace(start[1],end[1],num_points)#xs,ys

def connect_corners(corners, num_points=100):
    """Corners need to be ordered"""
    lines_x = np.zeros((len(corners),num_points))
    lines_y = np.zeros((len(corners),num_points))
    for _c in range(len(corners)):
        lines_x[_c], lines_y[_c] = draw_straight_line(corners[_c-1], corners[_c], num_points)
    lines_x = np.ravel(lines_x)
    lines_y = np.ravel(lines_y)
    return lines_x, lines_y
 if __name__ == '__main__':
     # GOlden ratio (1+5^(1/2))/2=(a+b)/a = a/b with a>b
     marginfigure_size = (8,8)
     line_figure_size = (16,16)
     paper_figure_size = (24,24)
