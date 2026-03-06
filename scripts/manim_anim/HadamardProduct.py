from manim import *

class HadamardProduct(Scene):
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
            [r"b_{11}", r"b_{12}", r"\cdots", r"b_{1n}"],
            [r"b_{21}", r"b_{22}", r"\cdots", r"b_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"b_{m1}", r"b_{m2}", r"\cdots", r"b_{mn}"]
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
        eq_lhs_C = MathTex(r"\mathbf{C} = \mathbf{A} \odot \mathbf{B} =")
        
        # Fixed the elements to represent proper matrix multiplication
        elements_C = [
            [r"a_{11} \cdot b_{11}", r"a_{12} \cdot b_{12}", r"\cdots", r"a_{1n} \cdot b_{1n}"],
            [r"a_{21} \cdot b_{21}", r"a_{22} \cdot b_{22}", r"\cdots", r"a_{2n} \cdot b_{2n}"],
            [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
            [r"a_{m1} \cdot b_{m1}", r"a_{m2} \cdot b_{m2}", r"\cdots", r"a_{mn} \cdot b_{mn}"]
        ]
        
        # Adjusted h_buff so elements don't overlap, scale down to fit screen
        matrixC = Matrix(elements_C, v_buff=0.8, h_buff=2.0)

        cols = matrixC.get_columns() 
        cols[2].stretch_to_fit_width(0.3)
        for i in range(1, len(cols)):
            cols[i].next_to(cols[i-1], RIGHT, buff=0.5)
        bracketCright = matrixC.get_brackets()[1]
        bracketCright.next_to(cols[-1], RIGHT, buff=0.5)

        equation_group_C = VGroup(eq_lhs_C, matrixC).arrange(RIGHT)
        equation_group_C.shift(DOWN * 2.2)
        
        # Draw C
        self.play(Write(eq_lhs_C))
        self.play(Create(matrixC.get_brackets()))
        
        eqnCiiLHS = MathTex(r"c_{ii} = a_{ii} \cdot b_{ii}")
        eqnCiiLHS.shift(DOWN*0.5)
        # eqnCiiRHS = MathTex(r"= a_{11} \cdot b_{11} + a_{12} \cdot b_{21} + \cdots + a_{1n} \cdot b_{n1} ")
        # eqnCiiRHS.next_to(eqnCiiLHS, RIGHT, buff=0.2)
        # eqnCiiRHS.scale(0.7)
        # eqnCiiRHS.shift(LEFT*4.5)
        self.play(Create(eqnCiiLHS), run_time=1)
        self.play(eqnCiiLHS.animate.shift(LEFT*4.5), run_time=1)
        # self.play(Create(eqnCiiRHS), run_time=2)
        # C11
        # for index3 in [0, 1, 3, 4, 5, 7, 12, 13, 15]:
        for index3 in range(16):
            if index3 in [2, 6, 8, 9, 10, 11, 14]:
                dotsC = matrixC.get_entries()[index3]
                self.play(Create(dotsC), run_time=1)
                continue

            entriesA_c11 = matrixA.get_entries()[index3].copy()
            entriesB_c11 = matrixB.get_entries()[index3].copy()
            entriesC_c11_a = matrixC.get_entries()[index3][0][:3]
            entriesC_c11_b = matrixC.get_entries()[index3][0][4:7]

            self.play(entriesA_c11.animate.set_color(YELLOW), run_time=1)
            self.play(entriesB_c11.animate.set_color(YELLOW), run_time=1)

            

            self.play(entriesA_c11.animate.shift(entriesC_c11_a.get_center() - entriesA_c11.get_center()),
                      entriesB_c11.animate.shift(entriesC_c11_b.get_center() - entriesB_c11.get_center()), 
                      Create(matrixC.get_entries()[index3][0][3]), run_time=2)
            self.play(entriesA_c11.animate.set_color(WHITE), entriesB_c11.animate.set_color(WHITE), run_time=1)
        # entriesC = matrixC.get_entries()
        # for i in range(4):
        #     rowC = entriesC[i*4 : (i+1)*4]
        #     self.play(Write(rowC), run_time=1.0)
        
        self.wait(3)
        
        

#if __name__ == "__main__":
#    from manim import config
#    config.pixel_width = 1920
#    config.pixel_height = 1080
#    scene = MatrixMultiplication()
 #   scene.render()
