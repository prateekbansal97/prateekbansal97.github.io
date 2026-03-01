from manim import *

class ManualColumnAdjust(Scene):
    def construct(self):
        m = Matrix([[1, 234, 3],[4, 5, 678]])
        
        # Access individual columns (0-indexed)
        cols = m.get_columns()
        
        # Make the second column wider
        cols[1].stretch_to_fit_width(2)
        
        # Re-arrange them to ensure proper spacing
        for i in range(1, len(cols)):
            cols[i].next_to(cols[i-1], RIGHT, buff=0.5)
        
        # Re-center the matrix
        m.center()
        self.add(m)