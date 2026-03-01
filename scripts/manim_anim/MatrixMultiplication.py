from manim import *

class MatrixMultiplication(Scene):
    def construct(self):
        # 1. Setup Matrix A
        elements_A = [
            [r"a_{11}", r"a_{12}", r"\cdots", r"a_{1n}"],
            [r"a_{21}", r"a_{22}", r"\cdots", r"a_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"a_{m1}", r"a_{m2}", r"\cdots", r"a_{mn}"]
        ]

        matrixA = Matrix(elements_A, v_buff=0.8, h_buff=1.0)
        eq_lhs_A = MathTex(r"\mathbf{A} = ")

        equation_group_A = VGroup(eq_lhs_A, matrixA).arrange(RIGHT)
        equation_group_A.shift(LEFT * 3.5 + UP * 2)

        # 2. Setup Matrix B
        elements_B = [
            [r"b_{11}", r"b_{12}", r"\cdots", r"b_{1p}"],
            [r"b_{21}", r"b_{22}", r"\cdots", r"b_{2p}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"b_{n1}", r"b_{n2}", r"\cdots", r"b_{np}"]
        ]

        matrixB = Matrix(elements_B, v_buff=0.8, h_buff=1.0)
        eq_lhs_B = MathTex(r"\mathbf{B} = ")

        equation_group_B = VGroup(eq_lhs_B, matrixB).arrange(RIGHT)
        equation_group_B.shift(RIGHT * 3.5 + UP * 2)

        # Draw A and B
        self.play(Write(eq_lhs_A), Write(eq_lhs_B))
        self.play(Create(matrixA.get_brackets()), Create(matrixB.get_brackets()))
        
        entriesA = matrixA.get_entries()
        entriesB = matrixB.get_entries()

        # Write rows of A and B
        for i in range(4):
            rowA = entriesA[i*4 : (i+1)*4]
            rowB = entriesB[i*4 : (i+1)*4]
            self.play(Write(rowA), Write(rowB), run_time=1.0)

        # 3. Setup Matrix C (The Product)
        eq_lhs_C = MathTex(r"\mathbf{C} = \mathbf{A}\mathbf{B} =")
        
        # Fixed the elements to represent proper matrix multiplication
        elements_C = [
            [r"a_{11}b_{11} + a_{12}b_{21} + \cdots + a_{1n}b_{n1}", r"a_{11}b_{12} + a_{12}b_{22} + \cdots + a_{1n}b_{n2}", r"\cdots", r"a_{11}b_{1p} + a_{12}b_{2p} + \cdots + a_{1n}b_{np}"],
            [r"a_{21}b_{11} + a_{22}b_{21} + \cdots + a_{2n}b_{n1}", r"a_{22}b_{12} + a_{22}b_{22} + \cdots + a_{2n}b_{n2}", r"\cdots", r"a_{21}b_{1p} + a_{22}b_{2p} + \cdots + a_{2n}b_{np}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"a_{m1}b_{11} + a_{m2}b_{21} + \cdots + a_{mn}b_{n1}", r"a_{m2}b_{12} + a_{m2}b_{22} + \cdots + a_{mn}b_{n2}", r"\cdots", r"a_{m1}b_{1p} + a_{m2}b_{2p} + \cdots + a_{mn}b_{np}"]
        ]
        
        # Adjusted h_buff so elements don't overlap, scale down to fit screen
        matrixC = Matrix(elements_C, v_buff=0.8, h_buff=7)

        cols = matrixC.get_columns() 
        cols[2].stretch_to_fit_width(0.3)
        for i in range(1, len(cols)):
            cols[i].next_to(cols[i-1], RIGHT, buff=0.5)
        bracketCright = matrixC.get_brackets()[1]
        bracketCright.next_to(cols[-1], RIGHT, buff=0.5)

        equation_group_C = VGroup(eq_lhs_C, matrixC).arrange(RIGHT)
        equation_group_C.scale(0.5).shift(DOWN * 2)
        
        # Draw C
        self.play(Write(eq_lhs_C))
        self.play(Create(matrixC.get_brackets()))
        
        entriesC = matrixC.get_entries()
        for i in range(4):
            rowC = entriesC[i*4 : (i+1)*4]
            self.play(Write(rowC), run_time=1.0)
        
        self.wait(3)
        
        # 4. Pedagogical Highlight
        # Grab the specific a_11 and b_11 entries in the top matrices

        a11_in_A = entriesA[0]
        b11_in_B = entriesB[0]
        
        # Grab the first entry of C: "a_{11}b_{11} + ..."
        c11_entry = entriesC[0]
        
        # In Manim, c11_entry[0] contains the rendered character paths (glyphs).
        # "a_{11}" takes up the first 3 glyphs (indices 0, 1, 2)
        # "b_{11}" takes up the next 3 glyphs (indices 3, 4, 5)
        a11_in_C = c11_entry[0][0:3]
        b11_in_C = c11_entry[0][3:6]
        
        # Simultaneously highlight the target variables
        self.play(
            a11_in_A.animate.set_color(YELLOW),
            a11_in_C.animate.set_color(YELLOW),
            b11_in_B.animate.set_color(GREEN),
            b11_in_C.animate.set_color(GREEN),
            run_time=1.5
        )
        
        self.wait(3)

#if __name__ == "__main__":
#    from manim import config
#    config.pixel_width = 1920
#    config.pixel_height = 1080
#    scene = MatrixMultiplication()
 #   scene.render()
