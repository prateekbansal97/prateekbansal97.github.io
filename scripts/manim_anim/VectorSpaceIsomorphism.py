from manim import *

class VectorSpaceIsomorphism(Scene):
    def construct(self):
        # 1. Setup the Title and Theorem context
        title = MathTex(r"\text{Theorem: } \mathbb{R}^{2 \times 2} \cong \mathbb{R}^4").to_edge(UP)
        subtitle = Text("Spaces of the same dimension can be transformed without loss", font_size=28).next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))

        # 2. Setup the Matrix (R^(2x2))
        # We color-code the entries to visually track that no information is lost
        matrix = Matrix([["x_1", "x_2"], ["x_3", "x_4"]]).shift(LEFT * 3)
        matrix.get_entries()[0].set_color(RED)
        matrix.get_entries()[1].set_color(GREEN)
        matrix.get_entries()[2].set_color(BLUE)
        matrix.get_entries()[3].set_color(YELLOW)
        
        label_matrix = MathTex(r"M \in \mathbb{R}^{2 \times 2}").next_to(matrix, DOWN)
        dim_matrix = MathTex(r"\dim = 4").next_to(label_matrix, DOWN).set_color(GRAY)

        self.play(Write(matrix), Write(label_matrix), FadeIn(dim_matrix))
        self.wait(1)

        # 3. Setup the Vector (R^4) structure on the right
        vector = Matrix([["x_1"], ["x_2"], ["x_3"], ["x_4"]]).shift(RIGHT * 3)
        vector.get_entries()[0].set_color(RED)
        vector.get_entries()[1].set_color(GREEN)
        vector.get_entries()[2].set_color(BLUE)
        vector.get_entries()[3].set_color(YELLOW)
        
        label_vector = MathTex(r"\vec{v} \in \mathbb{R}^4").next_to(vector, DOWN)
        dim_vector = MathTex(r"\dim = 4").next_to(label_vector, DOWN).set_color(GRAY)

        # 4. Show the Bijective Mapping
        # A double-headed arrow visually represents that this street goes both ways (bijective)
        transform_arrow = DoubleArrow(matrix.get_right(), vector.get_left(), buff=0.5, color=WHITE)
        phi_label = MathTex(r"\Phi").next_to(transform_arrow, UP)
        
        self.play(GrowArrow(transform_arrow), Write(phi_label))

        # 5. Animate the "Flattening" / Isomorphism
        # We transform copies of the matrix entries directly into the vector entries
        self.play(
            FadeIn(vector.get_brackets()),
            TransformFromCopy(matrix.get_entries()[0], vector.get_entries()[0]),
            TransformFromCopy(matrix.get_entries()[1], vector.get_entries()[1]),
            TransformFromCopy(matrix.get_entries()[2], vector.get_entries()[2]),
            TransformFromCopy(matrix.get_entries()[3], vector.get_entries()[3]),
            Write(label_vector),
            FadeIn(dim_vector),
            run_time=2
        )
        
        # 6. Emphasize that the dimensions dictate this relationship
        self.play(
            Indicate(dim_matrix, color=WHITE),
            Indicate(dim_vector, color=WHITE)
        )
        
        self.wait(2)
