from manim import *

class AutomorphismConcept(Scene):
    def construct(self):
        # 1. Setup the Title and Definition
        title = MathTex(r"\text{Automorphism: } \Phi: V \to V").to_edge(UP)
        subtitle = Text("Linear and Bijective (Maps to itself & Invertible)", font_size=32).next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))

        # 2. Create a single, large Vector Space V
        space_V = Ellipse(width=7, height=5, color=BLUE).shift(DOWN * 0.5)
        label_V = MathTex("V").next_to(space_V, DOWN, buff=0.3)
        
        self.play(Create(space_V), Write(label_V))

        # 3. Create initial vectors (dots) inside V
        v1 = Dot(space_V.get_center() + LEFT * 2 + UP * 1).set_color(YELLOW)
        v2 = Dot(space_V.get_center() + LEFT * 1.5 + DOWN * 1.5).set_color(YELLOW)
        label_v1 = MathTex("v_1").next_to(v1, LEFT)
        label_v2 = MathTex("v_2").next_to(v2, LEFT)

        self.play(FadeIn(v1, v2, label_v1, label_v2))

        # 4. Create target vectors (dots) ALSO inside V
        phi_v1 = Dot(space_V.get_center() + RIGHT * 2 + UP * 0.5).set_color(ORANGE)
        phi_v2 = Dot(space_V.get_center() + RIGHT * 1 + DOWN * 1).set_color(ORANGE)
        label_phi_v1 = MathTex(r"\Phi(v_1)").next_to(phi_v1, RIGHT)
        label_phi_v2 = MathTex(r"\Phi(v_2)").next_to(phi_v2, RIGHT)

        # 5. Animate the Internal Mapping (Phi) using Curved Arrows
        arrow1 = CurvedArrow(v1.get_center(), phi_v1.get_center(), angle=-TAU/6, color=WHITE)
        arrow2 = CurvedArrow(v2.get_center(), phi_v2.get_center(), angle=TAU/8, color=WHITE)
        
        phi_label1 = MathTex(r"\Phi").next_to(arrow1, UP, buff=0.1)
        phi_label2 = MathTex(r"\Phi").next_to(arrow2, DOWN, buff=0.1)
        
        self.play(
            Create(arrow1), FadeIn(phi_v1, label_phi_v1), Write(phi_label1)
        )
        self.play(
            Create(arrow2), FadeIn(phi_v2, label_phi_v2), Write(phi_label2)
        )
        
        self.wait(1)

        # 6. Emphasize Bijectivity by showing the inverse mapping
        # Reversing the start and end points of the curved arrow creates a perfect return loop
        inv_arrow1 = CurvedArrow(phi_v1.get_center(), v1.get_center(), angle=-TAU/6, color=RED)
        inv_arrow2 = CurvedArrow(phi_v2.get_center(), v2.get_center(), angle=TAU/8, color=RED)
        
        inv_label1 = MathTex(r"\Phi^{-1}").next_to(inv_arrow1, DOWN, buff=0.1).set_color(RED)
        inv_label2 = MathTex(r"\Phi^{-1}").next_to(inv_arrow2, UP, buff=0.1).set_color(RED)

        self.play(
            Create(inv_arrow1), Write(inv_label1),
            Create(inv_arrow2), Write(inv_label2)
        )
        
        self.wait(2)
