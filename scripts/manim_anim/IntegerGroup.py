from manim import *

class IntegerGroup(Scene):
    def construct(self):
        # 1. Title Setup
        title = MathTex(r"\text{The Infinite Group } (\mathbb{Z}, +)").to_edge(UP)
        definition = MathTex(r"\text{Operation } +: \mathbb{Z} \times \mathbb{Z} \to \mathbb{Z} \text{ (Translation on a discrete line)}").next_to(title, DOWN).scale(0.8)
        self.play(FadeIn(title, shift=DOWN), FadeIn(definition, shift=DOWN))
        
        # 2. Geometric Object Setup (Discrete Number Line for Z)
        nl = NumberLine(
            x_range=[-8, 8, 1],
            length=12,
            color=BLUE,
            include_numbers=True,
            label_direction=DOWN
        ).shift(DOWN * 1.5)
        
        # Highlight that Z is discrete (only the integer points exist in our group)
        integer_dots = VGroup(*[Dot(nl.n2p(i), color=BLUE_C) for i in range(-8, 9)])
        
        self.play(Create(nl))
        self.play(FadeIn(integer_dots, lag_ratio=0.1))
        self.wait(1)

        # Helper function to display the axioms
        def show_axiom(text_string, math_string):
            axiom_text = Text(text_string, font_size=30, color=GOLD).to_corner(UL).shift(DOWN * 1.5)
            axiom_math = MathTex(math_string).next_to(axiom_text, DOWN, aligned_edge=LEFT)
            bg = BackgroundRectangle(VGroup(axiom_text, axiom_math), color=BLACK, fill_opacity=0.8, buff=0.2)
            group = VGroup(bg, axiom_text, axiom_math)
            self.play(FadeIn(group, shift=RIGHT))
            return group

        # --- AXIOM 1: CLOSURE ---
        # For all a, b in G, a + b is in G
        ax1 = show_axiom("1. Closure", r"\forall a, b \in \mathbb{Z}, a + b \in \mathbb{Z}")
        
        # Visualizing a = 2, b = 3
        action_text = MathTex(r"2 + 3 = 5 \text{ (Lands on an integer)}").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Draw vector for 2
        vec_a = Arrow(nl.n2p(0), nl.n2p(2), buff=0, color=YELLOW).shift(UP * 0.5)
        label_a = MathTex("2").next_to(vec_a, UP)
        self.play(GrowArrow(vec_a), Write(label_a))
        
        # Draw vector for 3 starting from the tip of 2
        vec_b = Arrow(nl.n2p(2), nl.n2p(5), buff=0, color=ORANGE).shift(UP * 0.5)
        label_b = MathTex("3").next_to(vec_b, UP)
        self.play(GrowArrow(vec_b), Write(label_b))
        
        # Result lands perfectly on an integer dot
        self.play(Indicate(integer_dots[13], color=GREEN, scale_factor=2)) # Index 13 is the number 5
        self.wait(2)
        
        self.play(
            FadeOut(ax1), FadeOut(action_text), 
            FadeOut(vec_a), FadeOut(label_a), FadeOut(vec_b), FadeOut(label_b)
        )

        # --- AXIOM 3: IDENTITY ---
        # There exists e in G such that a + e = a
        ax3 = show_axiom("3. Identity", r"\exists e \in \mathbb{Z}: a + e = e + a = a")
        
        action_text = MathTex(r"4 + 0 = 4 \text{ (The Identity is } 0\text{)}").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Draw vector for 4
        vec_a = Arrow(nl.n2p(0), nl.n2p(4), buff=0, color=YELLOW).shift(UP * 0.5)
        label_a = MathTex("4").next_to(vec_a, UP)
        self.play(GrowArrow(vec_a), Write(label_a))
        
        # Apply 0 (no translation)
        self.play(Indicate(vec_a, color=RED))
        
        self.wait(2)
        self.play(FadeOut(ax3), FadeOut(action_text), FadeOut(vec_a), FadeOut(label_a))

        # --- AXIOM 4: INVERSE ---
        # For each a in G, there exists a^-1 in G
        ax4 = show_axiom("4. Inverse", r"\forall a \in \mathbb{Z}, \exists (-a): a + (-a) = 0")
        
        action_text = MathTex(r"3 + (-3) = 0 ").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Draw vector for 3
        vec_a = Arrow(nl.n2p(0), nl.n2p(3), buff=0, color=YELLOW).shift(UP * 0.5)
        label_a = MathTex("3").next_to(vec_a, UP)
        self.play(GrowArrow(vec_a), Write(label_a))
        
        # Draw inverse vector -3 returning to 0
        vec_inv = Arrow(nl.n2p(3), nl.n2p(0), buff=0, color=PURPLE).shift(UP * 1)
        label_inv = MathTex("-3").next_to(vec_inv, UP)
        self.play(GrowArrow(vec_inv), Write(label_inv))
        
        # Highlight 0
        self.play(Indicate(integer_dots[8], color=RED, scale_factor=2)) # Index 8 is the number 0
        self.wait(2)
        
        self.play(
            FadeOut(ax4), FadeOut(action_text), 
            FadeOut(vec_a), FadeOut(label_a), FadeOut(vec_inv), FadeOut(label_inv)
        )

        # --- AXIOM 2: ASSOCIATIVITY ---
        # (a + b) + c = a + (b + c)
        ax2 = show_axiom("2. Associativity", r"(a + b) + c = a + (b + c)")
        
        action_text = MathTex(r"(1 + 2) + 3 = 1 + (2 + 3)").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))

        # We will visualize jumping the segments
        # Path 1: (1+2) then +3
        vec_1 = Arrow(nl.n2p(0), nl.n2p(1), buff=0, color=YELLOW).shift(UP * 0.5)
        vec_2 = Arrow(nl.n2p(1), nl.n2p(3), buff=0, color=YELLOW).shift(UP * 0.5)
        brac_12 = Brace(VGroup(vec_1, vec_2), direction=UP)
        label_12 = brac_12.get_text("1+2")
        
        vec_3 = Arrow(nl.n2p(3), nl.n2p(6), buff=0, color=ORANGE).shift(UP * 0.5)
        
        self.play(GrowArrow(vec_1), GrowArrow(vec_2))
        self.play(GrowFromCenter(brac_12), Write(label_12))
        self.play(GrowArrow(vec_3))
        self.wait(1)

        # Path 2: 1+ then (2+3)
        self.play(FadeOut(brac_12), FadeOut(label_12))
        
        vec_2_alt = Arrow(nl.n2p(1), nl.n2p(3), buff=0, color=ORANGE).shift(UP * 0.5)
        self.play(Transform(vec_2, vec_2_alt)) # Change color to group with 3
        
        brac_23 = Brace(VGroup(vec_2, vec_3), direction=UP)
        label_23 = brac_23.get_text("2+3")
        self.play(GrowFromCenter(brac_23), Write(label_23))
        
        # Both paths end at 6
        self.play(Indicate(integer_dots[14], color=GREEN, scale_factor=2)) # Index 14 is the number 6
        self.wait(3)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = IntegerGroup()
    scene.render()
