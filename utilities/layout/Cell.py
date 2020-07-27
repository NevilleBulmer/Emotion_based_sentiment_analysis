from typing import List, Optional

import pandas as pd
import markdown

# Cell class used in conjunction with a Grid class to allow a user to create grids,
# streamlit doe not have any native functionality for this, so I created it.

# I accomplished this by utilizing css grids and overriding streamlits (unsafe_allow_html)
# functionality.
class Cell:
    # init function, used to initilize class (cells unique identifier), column / cell / grid padding
    # and any inner HTML.
    def __init__(
        self,
        # A cells class, unique identifier.
        class_: str = None,
        # The cells grid column start.
        grid_column_start: Optional[int] = None,
        # The cells grid column end.
        grid_column_end: Optional[int] = None,
        # The cells grid row start.
        grid_row_start: Optional[int] = None,
        # The cells grid row end.
        grid_row_end: Optional[int] = None,
    ):
        # Instantiate the variables.
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    # Used to return the style for the cell.
    def _to_style(self) -> str:
        # Returns the cells styl in css.
        return f"""
                    .{self.class_} {{
                        grid-column-start: {self.grid_column_start};
                        grid-column-end: {self.grid_column_end};
                        grid-row-start: {self.grid_row_start};
                        grid-row-end: {self.grid_row_end};
                    }}
                """
    # Used to create a cell which holds plain text.
    def cell_text(self, text: str = ""):
        self.inner_html = text

    # Used to create a cell which holds markdown text.
    def cell_markdown(self, text):
        self.inner_html = markdown.markdown(text)

    # Used to create a cell which holds a dataframe, for this we pass a dummy pd.DataFrame.
    def cell_dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()

    # Used to pass HTML to any cell, I..e enables the use of divs within cells.
    def to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""