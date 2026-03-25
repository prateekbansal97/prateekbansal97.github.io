from manim import *

class GeometricTranspose(LinearTransformationScene):
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            **kwargs
        )

    def construct(self):
        # 1. Define Matrix A (Horizontal Shear) and its Transpose (Vertical Shear)
        matrix_a = [[1, 2], [0, 1]]
        transpose_a = [[1, 0], [2, 1]]

        # 2. Setup the Titles
        title_main = MathTex(r"\text{Geometric Meaning of Transpose: } \mathbf{A}^\top").to_edge(UP)
        title_main.add_background_rectangle()
        self.add_foreground_mobject(title_main)

        # Definition from your text
        def_text = MathTex(r"\text{Recall: } b_{ij} = a_{ji}")
        def_text.add_background_rectangle()
        def_text.next_to(title_main, DOWN)
        self.add_foreground_mobject(def_text)
        
        self.wait(1)

        # 3. Label for Original Matrix A
        label_A = MathTex(r"\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix} \text{ (Horizontal Shear)}")
        label_A.add_background_rectangle()
        label_A.to_corner(UL).shift(DOWN * 1.5)

        # 4. Animate the Horizontal Shear
        self.add_foreground_mobject(label_A)
        self.play(FadeIn(label_A))
        self.apply_matrix(matrix_a)
        self.wait(2)

        # 5. Reset the grid to original state for comparison
        # We temporarily clear the applied transformation
        self.play(FadeOut(label_A))
        reset_text = Text("Resetting space...", font_size=32).add_background_rectangle().to_corner(UL).shift(DOWN * 1.5)
        self.add_foreground_mobject(reset_text)
        self.play(FadeIn(reset_text))
        
        # Applying the inverse of A resets our visual grid
        inverse_a = [[1, -2], [0, 1]] 
        self.apply_matrix(inverse_a)
        self.play(FadeOut(reset_text))
        self.wait(1)

        # 6. Label for Transposed Matrix A^T
        label_AT = MathTex(r"\mathbf{A}^\top = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix} \text{ (Vertical Shear)}")
        label_AT.add_background_rectangle()
        label_AT.to_corner(UL).shift(DOWN * 1.5)

        # 7. Animate the Vertical Shear
        self.add_foreground_mobject(label_AT)
        self.play(FadeIn(label_AT))
        self.apply_matrix(transpose_a)
        self.wait(2)

        # 8. Concluding Property
        conclusion = MathTex(r"\text{Property: } (\mathbf{A}^\top)^\top = \mathbf{A}")
        conclusion.add_background_rectangle()
        conclusion.move_to(DOWN * 2.5)
        
        self.add_foreground_mobject(conclusion)
        self.play(Write(conclusion))
        self.wait(3)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = GeometricTranspose()
    scene.render()
