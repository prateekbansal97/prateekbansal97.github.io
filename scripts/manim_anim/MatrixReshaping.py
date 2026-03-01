from manim import *

class MatrixReshapeAnimation(Scene):
    def construct(self):
        # Set up colors and style for squares
        # Colors selected to be close to the input image
        beige_color = "#EEDD88" # A light beige
        purple_color = "#AABBAA" # A light blue-purple
        square_style = {
            "side_length": 0.6,
            "stroke_color": BLACK,
            "stroke_width": 2,
            "fill_opacity": 1
        }

        # Create individual square mobjects for the matrix
        beige_squares = VGroup(*[
            Square(fill_color=beige_color, **square_style)
            for _ in range(4)
        ]).arrange(DOWN, buff=0)
        
        purple_squares = VGroup(*[
            Square(fill_color=purple_color, **square_style)
            for _ in range(4)
        ]).arrange(DOWN, buff=0)

        # Assemble the 4x2 matrix A
        matrix_a = VGroup(beige_squares, purple_squares).arrange(RIGHT, buff=0)
        matrix_a.to_edge(LEFT, buff=3.5).shift(UP * 0.5)

        # Create labels for matrix A
        a_label = MathTex(r"\mathbf{A} \in \mathbb{R}^{4 \times 2}").next_to(matrix_a, LEFT, buff=0.3).shift(UP * 1.5 + RIGHT * 2)

        # Create individual square mobjects for the vector
        vector_beige_squares = VGroup(*[
            Square(fill_color=beige_color, **square_style)
            for _ in range(4)
        ]).arrange(DOWN, buff=0)
        
        vector_purple_squares = VGroup(*[
            Square(fill_color=purple_color, **square_style)
            for _ in range(4)
        ]).arrange(DOWN, buff=0)

        # Assemble the 8x1 column vector a (Target structure)
        vector_a = VGroup(vector_beige_squares, vector_purple_squares).arrange(DOWN, buff=0)
        vector_a.to_edge(RIGHT, buff=3.5).shift(UP * 0.2)

        # Create labels for vector a
        v_label = MathTex(r"\mathbf{a} \in \mathbb{R}^{8}").next_to(vector_a, UP, buff=0.4)

        # Create the reshape text and arrow
        arrow = Arrow(start=LEFT * 1.5 + UP, end=RIGHT * 1.5 + UP, buff=0).shift(DOWN * 0.5)
        reshape_text = MathTex(r"\text{reshape}", font_size=48).next_to(arrow, UP, buff=0.1)
        reshape_group = VGroup(arrow, reshape_text)

        # Define the animation sequence
        self.play(Write(a_label), run_time=1)
        self.play(FadeIn(matrix_a), run_time=1)
        self.wait(1)

        self.play(FadeIn(reshape_group), run_time=1)
        self.wait(1)

        # Transformation step: Transform individual matrix columns into vector sections
        # The key is to transform the original squares into the *final* vector squares
        self.play(
            ReplacementTransform(beige_squares, vector_beige_squares),
            ReplacementTransform(purple_squares, vector_purple_squares),
            run_time=2.5,
            rate_func=linear
        )
        
        # Bring in the vector label after the transform
        self.play(Write(v_label), run_time=1)
        self.wait(3)

if __name__ == "__main__":
    # To run this script and see the animation, install ManimCE and run:
    # manim -pql myscript.py MatrixReshapeAnimation
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = MatrixReshapeAnimation()
    scene.render()
