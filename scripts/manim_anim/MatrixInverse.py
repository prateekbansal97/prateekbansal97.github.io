from manim import *

class MatrixInverseConcept(Scene):
    def construct(self):
        # 1. Title and General Definition
        title = MathTex(r"\text{Inverse of a Matrix: } \mathbf{A}^{-1}").to_edge(UP)
        subtitle = MathTex(
            r"\text{If } \mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A} = \mathbf{I}_n \text{, then } \mathbf{B} \text{ is the inverse } (\mathbf{A}^{-1})"
        ).next_to(title, DOWN).scale(0.8)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

        # 2. Introduce the 2x2 Matrix A
        eq_A = MathTex(r"\mathbf{A} = ")
        elements_A = [["a", "b"], ["c", "d"]]
        mat_A = Matrix(elements_A)
        group_A = VGroup(eq_A, mat_A).arrange(RIGHT).shift(LEFT * 3.5 + DOWN * 0.5)

        self.play(Write(eq_A), Create(mat_A.get_brackets()))
        self.play(Write(mat_A.get_entries()), run_time=1.5)
        self.wait(1)

        # 3. Introduce the 2x2 Inverse Formula Structure
        eq_inv = MathTex(r"\mathbf{A}^{-1} = \frac{1}{ad - bc}")
        elements_inv = [["d", "-b"], ["-c", "a"]]
        mat_inv = Matrix(elements_inv)
        
        # Group and position the inverse equation
        group_inv = VGroup(eq_inv, mat_inv).arrange(RIGHT).shift(RIGHT * 2.5 + DOWN * 0.5)

        self.play(Write(eq_inv), Create(mat_inv.get_brackets()))
        
        # 4. Color code elements to emphasize the transformation
        # Main diagonal swaps (yellow), off-diagonal negates (red)
        mat_A.get_entries()[0].set_color(YELLOW) # a
        mat_A.get_entries()[3].set_color(YELLOW) # d
        mat_A.get_entries()[1].set_color(RED)    # b
        mat_A.get_entries()[2].set_color(RED)    # c
        
        mat_inv.get_entries()[0].set_color(YELLOW) # d
        mat_inv.get_entries()[3].set_color(YELLOW) # a
        mat_inv.get_entries()[1].set_color(RED)    # -b
        mat_inv.get_entries()[2].set_color(RED)    # -c

        # Create copies of A's entries to animate their movement to the inverse matrix
        a_copy = mat_A.get_entries()[0].copy()
        b_copy = mat_A.get_entries()[1].copy()
        c_copy = mat_A.get_entries()[2].copy()
        d_copy = mat_A.get_entries()[3].copy()

        # Animate the swapping and negating visually
        self.play(
            Transform(d_copy, mat_inv.get_entries()[0]), # d moves up
            Transform(a_copy, mat_inv.get_entries()[3]), # a moves down
            Transform(b_copy, mat_inv.get_entries()[1]), # b negates
            Transform(c_copy, mat_inv.get_entries()[2]), # c negates
            run_time=2.5,
            path_arc=PI/4 # Adds a nice sweeping curve to the movement
        )
        self.wait(2)
        
        # 5. Clear the screen for the properties
        self.play(
            FadeOut(group_A), FadeOut(group_inv), FadeOut(subtitle), 
            FadeOut(a_copy), FadeOut(b_copy), FadeOut(c_copy), FadeOut(d_copy)
        )
        
        # 6. Display Key Properties
        prop_title = Text("Key Properties of Inverses", font_size=36, color=BLUE).next_to(title, DOWN, buff=0.5)
        self.play(Write(prop_title))

        prop1 = MathTex(r"1.\ \mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I}_n")
        prop2 = MathTex(r"2.\ (\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}")
        prop3 = MathTex(r"3.\ (\mathbf{A} + \mathbf{B})^{-1} \neq \mathbf{A}^{-1} + \mathbf{B}^{-1}")
        prop4 = MathTex(r"4.\ (\mathbf{A} \odot \mathbf{B})^{-1} \neq \mathbf{A}^{-1} \odot \mathbf{B}^{-1}")
        
        # Arrange the properties in a neat list
        props = VGroup(prop1, prop2, prop3, prop4).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 1)

        for prop in props:
            self.play(FadeIn(prop, shift=RIGHT * 0.5))
            self.wait(1)
        
        self.wait(3)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = MatrixInverseConcept()
    scene.render()
