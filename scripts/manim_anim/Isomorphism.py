from manim import *

class IsomorphismConcept(Scene):
    def construct(self):
        # 1. Setup the Title and Definition
        title = MathTex(r"\text{Isomorphism: } \Phi: V \to W").to_edge(UP)
        subtitle = Text("Linear and Bijective (1-to-1 & Onto)", font_size=32).next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))

        # 2. Create the Vector Spaces V and W as abstract sets (Ellipses)
        space_V = Ellipse(width=3, height=4, color=BLUE).shift(LEFT * 3.5)
        label_V = MathTex("V").next_to(space_V, DOWN)
        
        space_W = Ellipse(width=3, height=4, color=GREEN).shift(RIGHT * 3.5)
        label_W = MathTex("W").next_to(space_W, DOWN)
        
        self.play(Create(space_V), Write(label_V), Create(space_W), Write(label_W))

        # 3. Create vectors (dots) in V
        v1 = Dot(space_V.get_center() + UP * 1).set_color(YELLOW)
        v2 = Dot(space_V.get_center() + DOWN * 1).set_color(YELLOW)
        label_v1 = MathTex("v_1").next_to(v1, LEFT)
        label_v2 = MathTex("v_2").next_to(v2, LEFT)

        self.play(FadeIn(v1, v2, label_v1, label_v2))

        # 4. Create vectors (dots) in W to show the mapping
        w1 = Dot(space_W.get_center() + UP * 1).set_color(YELLOW)
        w2 = Dot(space_W.get_center() + DOWN * 1).set_color(YELLOW)
        label_w1 = MathTex(r"\Phi(v_1)").next_to(w1, RIGHT)
        label_w2 = MathTex(r"\Phi(v_2)").next_to(w2, RIGHT)

        # 5. Animate the Bijective Mapping (Phi)
        arrow1 = Arrow(v1.get_right(), w1.get_left(), buff=0.2, color=WHITE)
        arrow2 = Arrow(v2.get_right(), w2.get_left(), buff=0.2, color=WHITE)
        phi_label = MathTex(r"\Phi").next_to(arrow1, UP)
        
        self.play(GrowArrow(arrow1), FadeIn(w1, label_w1), Write(phi_label))
        self.play(GrowArrow(arrow2), FadeIn(w2, label_w2))
        
        self.wait(1)

        # 6. Emphasize Bijectivity by showing the inverse mapping (Phi inverse)
        inv_arrow = Arrow(w2.get_left(), v2.get_right(), buff=0.2, color=RED).shift(DOWN * 0.5)
        inv_label = MathTex(r"\Phi^{-1}").next_to(inv_arrow, DOWN).set_color(RED)
        
        self.play(GrowArrow(inv_arrow), Write(inv_label))
        
        self.wait(2)
