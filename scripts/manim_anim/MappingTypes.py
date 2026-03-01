from manim import *

class MappingExamples(Scene):
    def construct(self):
        # 1. Main Title
        #title = Text("Injective, Surjective & Bijective Mappings", font_size=40).to_edge(UP)
        title = MathTex(r"\text{Injective, Surjective \& Bijective Mappings}", font_size=40).to_edge(UP)
        self.play(Write(title))

        # 2. Define Ellipses (Sets)
        sur_L = Ellipse(width=1.5, height=5, color=YELLOW).shift(LEFT*6.0 + DOWN*0.5)
        sur_R = Ellipse(width=1.5, height=5, color=YELLOW).shift(LEFT*4.0 + DOWN*0.5)
        inj_L = Ellipse(width=1.5, height=5, color=YELLOW).shift(RIGHT*-1.5 + DOWN*0.5)
        inj_R = Ellipse(width=1.5, height=5, color=YELLOW).shift(RIGHT*0.5 + DOWN*0.5)
        bij_L = Ellipse(width=1.5, height=5, color=YELLOW).shift(RIGHT*3.5 + DOWN*0.5)
        bij_R = Ellipse(width=1.5, height=5, color=YELLOW).shift(RIGHT*5.5 + DOWN*0.5)

        # 3. Define Labels
        sur_label = MathTex(r"\text{Surjective}", font_size=32).next_to(sur_L, DOWN, buff=0.1).shift(RIGHT*1.5)
        inj_label = MathTex(r"\text{Injective}", font_size=32).next_to(inj_L, DOWN, buff=0.1).shift(RIGHT*1.5)
        bij_label = MathTex(r"\text{Bijective}", font_size=32).next_to(bij_L, DOWN, buff=0.1).shift(RIGHT*1.5)
        
        sur_label.shift(LEFT*0.5)
        inj_label.shift(LEFT*0.5)
        bij_label.shift(LEFT*0.5)

        # 4. Define Elements and Arrows for Surjective Mapping
        sur_L_elements = VGroup(
            Text("t", color=BLUE, font_size=36).shift(sur_L.get_center() + UP*1.5),
            Text("u", color=BLUE, font_size=36).shift(sur_L.get_center() + UP*0.5),
            Text("v", color=BLUE, font_size=36).shift(sur_L.get_center() + DOWN*0.5),
            Text("w", color=BLUE, font_size=36).shift(sur_L.get_center() + DOWN*1.5)
        )
        sur_R_elements = VGroup(
            Text("5", color=RED, font_size=36).shift(sur_R.get_center() + UP*1),
            Text("2", color=RED, font_size=36).shift(sur_R.get_center() + 0),
            Text("1", color=RED, font_size=36).shift(sur_R.get_center() + DOWN*1)
        )
        sur_arrows = VGroup(
            Arrow(sur_L_elements[0].get_right(), sur_R_elements[0].get_left(), buff=0.2, color=WHITE),
            Arrow(sur_L_elements[1].get_right(), sur_R_elements[1].get_left(), buff=0.2, color=WHITE),
            Arrow(sur_L_elements[2].get_right(), sur_R_elements[2].get_left(), buff=0.2, color=WHITE),
            Arrow(sur_L_elements[3].get_right(), sur_R_elements[2].get_left(), buff=0.2, color=WHITE)
        )

        # 5. Define Elements and Arrows for Injective Mapping
        inj_L_elements = VGroup(
            Text("x", color=BLUE, font_size=36).shift(inj_L.get_center() + UP*1),
            Text("y", color=BLUE, font_size=36).shift(inj_L.get_center() + 0),
            Text("z", color=BLUE, font_size=36).shift(inj_L.get_center() + DOWN*1)
        )
        inj_R_elements = VGroup(
            Text("1", color=RED, font_size=36).shift(inj_R.get_center() + UP*1.5),
            Text("2", color=RED, font_size=36).shift(inj_R.get_center() + UP*0.5),
            Text("3", color=RED, font_size=36).shift(inj_R.get_center() + DOWN*0.5),
            Text("4", color=RED, font_size=36).shift(inj_R.get_center() + DOWN*1.5)
        )
        inj_arrows = VGroup(
            Arrow(inj_L_elements[0].get_right(), inj_R_elements[0].get_left(), buff=0.2, color=WHITE),
            Arrow(inj_L_elements[1].get_right(), inj_R_elements[1].get_left(), buff=0.2, color=WHITE),
            Arrow(inj_L_elements[2].get_right(), inj_R_elements[2].get_left(), buff=0.2, color=WHITE)
        )

        # 6. Define Elements and Arrows for Bijective Mapping
        bij_L_elements = VGroup(
            Text("x", color=BLUE, font_size=36).shift(bij_L.get_center() + UP*1),
            Text("y", color=BLUE, font_size=36).shift(bij_L.get_center() + 0),
            Text("z", color=BLUE, font_size=36).shift(bij_L.get_center() + DOWN*1)
        )
        bij_R_elements = VGroup(
            Text("1", color=RED, font_size=36).shift(bij_R.get_center() + UP*1),
            Text("2", color=RED, font_size=36).shift(bij_R.get_center() + 0),
            Text("3", color=RED, font_size=36).shift(bij_R.get_center() + DOWN*1)
        )
        bij_arrows = VGroup(
            Arrow(bij_L_elements[0].get_right(), bij_R_elements[0].get_left(), buff=0.2, color=WHITE),
            Arrow(bij_L_elements[1].get_right(), bij_R_elements[1].get_left(), buff=0.2, color=WHITE),
            Arrow(bij_L_elements[2].get_right(), bij_R_elements[2].get_left(), buff=0.2, color=WHITE)
        )

        # 7. Animation Sequence
        self.play(FadeIn(sur_L, sur_R, inj_L, inj_R, bij_L, bij_R))
        self.play(Write(sur_L_elements), Write(sur_R_elements))
        self.play(GrowArrow(sur_arrows[0]), GrowArrow(sur_arrows[1]), GrowArrow(sur_arrows[2]), GrowArrow(sur_arrows[3]))
        self.play(Write(sur_label))

        self.play(Write(inj_L_elements), Write(inj_R_elements))
        self.play(GrowArrow(inj_arrows[0]), GrowArrow(inj_arrows[1]), GrowArrow(inj_arrows[2]))
        self.play(Write(inj_label))

        self.play(Write(bij_L_elements), Write(bij_R_elements))
        self.play(GrowArrow(bij_arrows[0]), GrowArrow(bij_arrows[1]), GrowArrow(bij_arrows[2]))
        self.play(Write(bij_label))
        
        self.wait(2)
