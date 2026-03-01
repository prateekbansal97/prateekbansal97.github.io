from manim import *

class MatrixAddition(Scene):
    def construct(self):
        
        elements_A = [
            [r"a_{11}",r"a_{12}", r"\cdots", r"a_{1n}"],
            [r"a_{21}",r"a_{22}", r"\cdots", r"a_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"a_{m1}", r"a_{m2}", r"\cdots", r"a_{mn}"]
        ]

        matrixA = Matrix(elements_A, v_buff=0.8, h_buff=1.0)
        eq_lhs_A = MathTex(r"\mathbf{A} = ")

        equation_group_A = VGroup(eq_lhs_A, matrixA).arrange(RIGHT)
        equation_group_A.shift(LEFT*3.2 + UP*2)

        elements_B = [
            [r"b_{11}",r"b_{12}", r"\cdots", r"b_{1n}"],
            [r"b_{21}",r"b_{22}", r"\cdots", r"b_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"b_{m1}", r"b_{m2}", r"\cdots", r"b_{mn}"]
        ]

        matrixB = Matrix(elements_B, v_buff=0.8, h_buff=1.0)
        eq_lhs_B = MathTex(r"\mathbf{B} = ")

        equation_group_B = VGroup(eq_lhs_B, matrixB).arrange(RIGHT)
        equation_group_B.shift(RIGHT*3.2 + UP*2)


        self.play(Write(eq_lhs_A), Write(eq_lhs_B))
        self.play(Create(matrixA.get_brackets()), Create(matrixB.get_brackets()))
        
        # Isolate the rows to animate them one by one (emphasizing element-wise addition)
        entriesA = matrixA.get_entries()
        row1A = entriesA[0:4]
        row2A = entriesA[4:8]
        row3A = entriesA[8:12]
        row4A = entriesA[12:16]

        entriesB = matrixB.get_entries()
        row1B = entriesB[0:4]
        row2B = entriesB[4:8]
        row3B = entriesB[8:12]
        row4B = entriesB[12:16]

        self.play(Write(row1A), Write(row1B), run_time=1.5)
        self.play(Write(row2A), Write(row2B), run_time=1.5)
        self.play(Write(row3A), Write(row3B), run_time=1)
        self.play(Write(row4A), Write(row4B), run_time=1.5)

        # 2. Setup the Equation and Matrix
        eq_lhs = MathTex(r"\mathbf{C} = \mathbf{A} + \mathbf{B} =")
        
        # Define the individual elements of the expanded matrix
        elements = [
            [r"a_{11} + b_{11}", r"a_{12} + b_{12}", r"\cdots", r"a_{1n} + b_{1n}"],
            [r"a_{21} + b_{21}", r"a_{22} + b_{22}", r"\cdots", r"a_{2n} + b_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"a_{m1} + b_{m1}", r"a_{m2} + b_{m2}", r"\cdots", r"a_{mn} + b_{mn}"]
        ]
        
        # Using a slightly larger h_buff to ensure the wide equation entries don't overlap
        matrix = Matrix(elements, v_buff=0.8, h_buff=3.0)
        
        # Group the left-hand side and the matrix, scale it down, and position it
        equation_group = VGroup(eq_lhs, matrix).arrange(RIGHT)
        equation_group.scale(0.7).shift(DOWN * 2)
        
        # 3. Animate the matrix building process
        self.play(Write(eq_lhs))
        self.play(Create(matrix.get_brackets()))
        
        # Isolate the rows to animate them one by one (emphasizing element-wise addition)
        entries = matrix.get_entries()
        row1 = entries[0:4]
        row2 = entries[4:8]
        row3 = entries[8:12]
        row4 = entries[12:16]

        self.play(Write(row1), run_time=1.5)
        self.play(Write(row2), run_time=1.5)
        self.play(Write(row3), run_time=1)
        self.play(Write(row4), run_time=1.5)
        
        self.wait(1)
        
        # 4. Pedagogical Highlight: Tie the matrix back to the text definition c_{ij} = a_{ij} + b_{ij}
        # Highlight the first element
        highlight_box = SurroundingRectangle(row1[0], color=YELLOW, buff=0.1)
        highlight_text = MathTex(r"c_{11} = a_{11} + b_{11}").scale(0.8).next_to(highlight_box, UP, buff=0.2).set_color(YELLOW)
        
        self.play(Create(highlight_box), FadeIn(highlight_text))
        self.wait(2)
        
        # Move the highlight to the generic 'mn' element
        highlight_box2 = SurroundingRectangle(row4[3], color=GREEN, buff=0.1)
        highlight_text2 = MathTex(r"c_{mn} = a_{mn} + b_{mn}").scale(0.8).next_to(highlight_box2, DOWN, buff=0.2).set_color(GREEN)
        
        self.play(
            Transform(highlight_box, highlight_box2),
            Transform(highlight_text, highlight_text2)
        )
        self.wait(3)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = MatrixAddition()
    scene.render()
