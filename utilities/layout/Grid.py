import streamlit as graphical_interface
from typing import List, Optional
import markdown

from utilities.layout.Cell import Cell

# Grid class used in conjunction with a Cell class to allow a user to create grids,
# streamlit doe not have any native functionality for this, so I created it.

# I accomplished this by utilizing css grids and overriding streamlits (unsafe_allow_html)
# functionality.
class Grid:
    """A (CSS) Grid"""
    # init function, used to initilize column number, padding, background color and color.
    def __init__(
        self, 
        # Column number.
        template_columns="1 1 1", 
        # Padding.
        gap="10px", 
        # Background color.
        background_color="#fff", 
        # Color.
        color="#444"
    ):
        # Instantiate the variables.
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        # Creates an array of the cells to keep track of there content, potision.
        self.cells: List[Cell] = []

    # Function to return the current styles used by the grid.
    def _get_grid_style(self):
        # Return the style, in css.
        return f"""
            <style>
                .wrapper {{
                display: grid;
                grid-template-columns: {self.template_columns};
                grid-gap: {self.gap};
                background-color: {self.background_color};
                color: {self.color};
                }}
                .box {{
                background-color: {self.color};
                color: {self.background_color};
                border-radius: 5px;
                padding: 20px;
                font-size: 100%;
                }}
                table {{
                    color: {self.color}
                }}
            </style>
            """

    # Function to return the current styles used by the cells.
    def _get_cells_style(self):
        # Return the style, in css, along with the cells array which we iterate through.
        return (
            "<style>" + "\n".join([cell._to_style() for cell in self.cells]) + "</style>"
        )

    # Function to return the current styles used by the cells html.
    def _get_cells_html(self):
        # Return the style, in css, along with the cells array which we iterate through.
        return (
            '<div class="wrapper">'
            + "\n".join([cell.to_html() for cell in self.cells])
            + "</div>"
        )

    # See the Cell class for an explanation of each of the variables.
    def cell(
        self,
        # Class, the cells identifier.
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        # Adds the cells to the cells array.
        self.cells.append(cell)
        # Returns the cell.
        return cell

    # Default functions used when Instantiating and Destroying as cell.
    # Enter, used when Instantiating a new cell, returns the created cell.
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        graphical_interface.markdown(self._get_grid_style(), unsafe_allow_html=True)
        graphical_interface.markdown(self._get_cells_style(), unsafe_allow_html=True)
        graphical_interface.markdown(self._get_cells_html(), unsafe_allow_html=True)