from manim import *

class HomomorphismExample(Scene):
    def construct(self):
        # 1. Setup the Title and Specific Mapping
        title = MathTex(r"\text{Homomorphism: } \Phi: \mathbb{R}^2 \to \mathbb{C}").to_edge(UP)
        subtitle = MathTex(
            r"\Phi \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = x_1 + i x_2"
        ).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))

        # 2. Create the two distinct spaces
        space_R2 = Ellipse(width=3.5, height=5, color=BLUE).shift(LEFT * 3.5 + DOWN * 0.5)
        label_R2 = MathTex(r"\mathbb{R}^2").next_to(space_R2, DOWN)
        
        space_C = Ellipse(width=3.5, height=5, color=GREEN).shift(RIGHT * 3.5 + DOWN * 0.5)
        label_C = MathTex(r"\mathbb{C}").next_to(space_C, DOWN)
        
        self.play(Create(space_R2), Write(label_R2), Create(space_C), Write(label_C))

        # 3. Create initial elements x and y in R^2
        dot_x = Dot(space_R2.get_center() + UP * 1.5 + LEFT * 0.5).set_color(YELLOW)
        label_x = MathTex("x").next_to(dot_x, LEFT)
        
        dot_y = Dot(space_R2.get_center() + DOWN * 0.5 + LEFT * 0.5).set_color(YELLOW)
        label_y = MathTex("y").next_to(dot_y, LEFT)

        self.play(FadeIn(dot_x, label_x, dot_y, label_y))

        # 4. Map them individually to C
        dot_phix = Dot(space_C.get_center() + UP * 1.5 + LEFT * 0.5).set_color(ORANGE)
        label_phix = MathTex(r"\Phi(x)").next_to(dot_phix, RIGHT)
        
        dot_phiy = Dot(space_C.get_center() + DOWN * 0.5 + LEFT * 0.5).set_color(ORANGE)
        label_phiy = MathTex(r"\Phi(y)").next_to(dot_phiy, RIGHT)

        arrow_x = Arrow(dot_x.get_right(), dot_phix.get_left(), buff=0.1, color=WHITE)
        arrow_y = Arrow(dot_y.get_right(), dot_phiy.get_left(), buff=0.1, color=WHITE)

        self.play(GrowArrow(arrow_x), FadeIn(dot_phix, label_phix))
        self.play(GrowArrow(arrow_y), FadeIn(dot_phiy, label_phiy))

        # 5. Show addition happening inside R^2
        dot_x_plus_y = Dot(space_R2.get_center() + UP * 0.5 + RIGHT * 0.5).set_color(YELLOW)
        label_x_plus_y = MathTex("x+y").next_to(dot_x_plus_y, LEFT)
        
        self.play(TransformFromCopy(VGroup(dot_x, dot_y), dot_x_plus_y))
        self.play(Write(label_x_plus_y))

        # 6. Show addition happening inside C
        dot_phix_plus_phiy = Dot(space_C.get_center() + UP * 0.5 + RIGHT * 0.5).set_color(ORANGE)
        label_phix_plus_phiy = MathTex(r"\Phi(x) + \Phi(y)").next_to(dot_phix_plus_phiy, RIGHT)
        
        self.play(TransformFromCopy(VGroup(dot_phix, dot_phiy), dot_phix_plus_phiy))
        self.play(Write(label_phix_plus_phiy))

        # 7. The Core of Homomorphism: Map the sum over, and show it perfectly aligns
        arrow_sum = Arrow(dot_x_plus_y.get_right(), dot_phix_plus_phiy.get_left(), buff=0.1, color=YELLOW)
        label_phi_xy = MathTex(r"\Phi(x+y)").next_to(arrow_sum, UP*0.1)
        
        self.play(GrowArrow(arrow_sum), Write(label_phi_xy))
        
        # Pulse the final equation to emphasize structure is preserved
        self.play(Indicate(label_phix_plus_phiy, color=YELLOW, scale_factor=1.2))
        
        self.wait(2)
