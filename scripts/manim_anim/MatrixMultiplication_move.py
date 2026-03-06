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
        
        eqnCiiLHS = MathTex(r"c_{11} = \sum\limits_{i=1}^{n} a_{1n} \cdot b_{n1}")
        eqnCiiLHS.shift(DOWN*0.7).scale(0.7)
        eqnCiiRHS = MathTex(r"= a_{11} \cdot b_{11} + a_{12} \cdot b_{21} + \cdots + a_{1n} \cdot b_{n1} ")
        eqnCiiRHS.next_to(eqnCiiLHS, RIGHT, buff=0.2)
        eqnCiiRHS.scale(0.7)
        eqnCiiRHS.shift(LEFT*4.5)
        self.play(Create(eqnCiiLHS), run_time=2)
        self.play(eqnCiiLHS.animate.shift(LEFT*3.5), run_time=1)
        self.play(Create(eqnCiiRHS), run_time=2)
        # C11
        # for index3 in [0, 1, 3, 4, 5, 7, 12, 13, 15]:
        for index3 in range(16):
            if index3 in [2, 6, 8, 9, 10, 11, 14]:
                dotsC = matrixC.get_entries()[index3]
                self.play(Create(dotsC), run_time=1)
                continue
            Aindices = np.array([0, 7, 18])
            Bindices = Aindices + 3
            entriesA_c11 = [j.copy() for j in matrixA.get_entries()[(index3 // 4)*4:(index3 // 4)*4+4]]
            entriesB_c11 = [j.copy() for j in matrixB.get_entries()[(index3 % 4)::4]]
            print(entriesA_c11, entriesB_c11)
            self.play(*[j.animate.set_color(YELLOW) for j in entriesA_c11], run_time=1)
            self.play(*[j.animate.set_color(YELLOW) for j in entriesB_c11], run_time=1)

            for index1, index2 in zip([0, 1, 3], range(3)):
                entriesA_c11_a11 = entriesA_c11[index1]
                entriesB_c11_b11 = entriesB_c11[index1]
                entriesC_c11_a11 = matrixC.get_entries()[index3][0][Aindices[index2]:Aindices[index2]+2]
                entriesC_c11_b11 = matrixC.get_entries()[index3][0][Bindices[index2]:Bindices[index2]+2]

                self.play(entriesA_c11_a11.animate.shift(entriesC_c11_a11.get_center() - entriesA_c11_a11.get_center()).scale(0.5),
                      entriesB_c11_b11.animate.shift(entriesC_c11_b11.get_center() - entriesB_c11_b11.get_center()).scale(0.5), run_time=2)
                if index2 == 0:
                    self.play(Create(matrixC.get_entries()[index3][0][Bindices[index2]+3]))
                elif index2 == 1:
                    self.play(Create(matrixC.get_entries()[index3][0][13:18]))

            self.play(*[j.animate.set_color(WHITE) for j in entriesA_c11], *[j.animate.set_color(WHITE) for j in entriesB_c11], run_time=1)
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
