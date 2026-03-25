from manim import *

class VectorSpaceOperations(Scene):
    def construct(self):
        # --- 1. Introduction ---
        title = Text("Vector Space Operations", font_size=40).to_edge(UP)
        self.play(Write(title))

        # --- 2. Setup the Vector Space V ---
        # We represent V as a 2D plane
        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4}
        )
        plane.next_to(title, DOWN, buff=1)
        plane_label = MathTex(r"\mathcal{V}").set_color(BLUE).scale(1.5).to_corner(DL)
        
        self.play(Create(plane), Write(plane_label))
        self.wait(1)

        # --- 3. The Inner Operation: Vector Addition ---
        inner_title = MathTex(r"\text{Inner Operation: } + : \mathcal{V} \times \mathcal{V} \to \mathcal{V}").next_to(title, DOWN).set_color(YELLOW)
        self.play(FadeIn(inner_title))

        # Create two vectors IN the space V
        v1 = Vector([1, 2], color=YELLOW)
        v1.shift(DOWN)
        v1_label = MathTex(r"\vec{v}").next_to(v1.get_end(), DOWN).set_color(YELLOW)
        
        v2 = Vector([2, -1], color=GREEN)
        v2.shift(DOWN)
        v2_label = MathTex(r"\vec{w}").next_to(v2.get_end(), DOWN).set_color(GREEN)

        self.play(GrowArrow(v1), Write(v1_label))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # Animate addition (Tip to Tail)
        v2_shifted = Vector([2, -1], color=GREEN).shift(v1.get_end())
        self.play(
            Transform(v2, v2_shifted),
            v2_label.animate.next_to(v2_shifted.get_end(), RIGHT)
        )

        # The result is ALSO in V
        v_sum = Vector([3, 1], color=BLUE_B)
        v_sum.shift(DOWN)
        v_sum_label = MathTex(r"\vec{v} + \vec{w}").next_to(v_sum.get_end(), RIGHT).set_color(BLUE_B)
        
        self.play(GrowArrow(v_sum), FadeOut(v1_label), FadeOut(v2_label), Write(v_sum_label))
        self.wait(2)

        # Clean up for the next scene
        self.play(
            FadeOut(v1, v2, v_sum, v_sum_label, inner_title)
        )

        # --- 4. The Outer Operation: Scalar Multiplication ---
        outer_title = MathTex(r"\text{Outer Operation: } \cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}").next_to(title, DOWN).set_color(RED)
        self.play(FadeIn(outer_title))

        # Bring back a single vector in V
        v = Vector([1.5, 1], color=YELLOW)
        v.shift(DOWN)
        v_label = MathTex(r"\vec{v} \in \mathcal{V}").next_to(v.get_end(), RIGHT).set_color(YELLOW)
        self.play(GrowArrow(v), Write(v_label))

        # Represent the field R as a completely separate number line "outside" the space V
        real_line = NumberLine(
            x_range=[-3, 3, 1], length=6, color=RED, include_numbers=True
        ).to_corner(DR).shift(LEFT * 1 + DOWN * 0.1)
        
        real_label = MathTex(r"\mathbb{R}").next_to(real_line, LEFT).set_color(RED)
        
        self.play(Create(real_line), Write(real_label))
        self.wait(1)

        # Highlight a scalar from the outside set
        scalar_dot = Dot(real_line.n2p(2), color=RED)
        scalar_label = MathTex(r"\lambda = 2").next_to(scalar_dot, UP).set_color(RED)
        
        self.play(FadeIn(scalar_dot), Write(scalar_label))
        self.play(Indicate(scalar_dot, scale_factor=1.5))
        self.wait(1)

        # Animate the scalar physically coming down to multiply the vector
        self.play(
            scalar_label.animate.next_to(v_label, UP),
            FadeOut(scalar_dot)
        )

        # The vector scales, but remains trapped inside V
        v_scaled = Vector([3, 2], color=ORANGE)
        v_scaled.shift(DOWN)
        v_scaled_label = MathTex(r"2\vec{v} \in \mathcal{V}").next_to(v_scaled.get_end(), RIGHT).set_color(ORANGE)

        self.play(
            Transform(v, v_scaled),
            Transform(v_label, v_scaled_label),
            FadeOut(scalar_label),
            run_time=1.5
        )
        self.wait(3)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = VectorSpaceOperations()
    scene.render()
