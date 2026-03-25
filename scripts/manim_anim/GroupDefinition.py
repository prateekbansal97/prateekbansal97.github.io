from manim import *

class AdvancedGroupAxioms(Scene):
    def construct(self):
        # 1. Sophisticated Title Setup
        title = MathTex(r"\text{The Group } G := (\mathcal{G}, \otimes) \text{ as Transformations}").to_edge(UP)
        definition = MathTex(r"\otimes: \mathcal{G} \times \mathcal{G} \to \mathcal{G} \text{ (Composition of Actions)}").next_to(title, DOWN).scale(0.8)
        self.play(FadeIn(title, shift=DOWN), FadeIn(definition, shift=DOWN))
        
        # 2. Geometric Object Setup (Equilateral Triangle for D3 Group)
        # We add numbered vertices so the viewer can track the orientation
        triangle = RegularPolygon(n=3, radius=2, color=BLUE_D, fill_opacity=0.4)
        v_labels = VGroup(
            MathTex("1", color=YELLOW).move_to(triangle.get_vertices()[0] + UP * 0.3),
            MathTex("2", color=YELLOW).move_to(triangle.get_vertices()[1] + LEFT * 0.4 + DOWN * 0.2),
            MathTex("3", color=YELLOW).move_to(triangle.get_vertices()[2] + RIGHT * 0.4 + DOWN * 0.2)
        )
        obj = VGroup(triangle, v_labels).shift(DOWN * 0.5)
        
        self.play(DrawBorderThenFill(triangle), Write(v_labels))
        self.wait(1)

        # Helper function to track text on the left
        def show_axiom(text_string, math_string):
            axiom_text = Text(text_string, font_size=32, color=GOLD).to_corner(UL).shift(DOWN * 1.5)
            axiom_math = MathTex(math_string).next_to(axiom_text, DOWN, aligned_edge=LEFT)
            bg = BackgroundRectangle(VGroup(axiom_text, axiom_math), color=BLACK, fill_opacity=0.8, buff=0.2)
            group = VGroup(bg, axiom_text, axiom_math)
            self.play(FadeIn(group, shift=RIGHT))
            return group

        # --- AXIOM 1: CLOSURE ---
        # For all a, b in G, a (x) b is in G
        ax1 = show_axiom("1. Closure", r"\forall a, b \in \mathcal{G}, a \otimes b \in \mathcal{G}")
        
        action_text = MathTex(r"r \otimes s \text{ (Reflect, then Rotate)}").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Action s: Reflection across y-axis
        self.play(obj.animate.flip(axis=UP), run_time=1.5)
        self.wait(0.5)
        
        # Action r: Rotation by 120 degrees (2*PI/3)
        self.play(Rotate(obj, angle=2*PI/3, about_point=obj.get_center()), run_time=1.5)
        
        # Result: The composition is just a different reflection (s') already in the group
        result_text = MathTex(r"= s' \text{ (A single reflection)}").next_to(action_text, DOWN)
        self.play(Write(result_text))
        self.play(Indicate(obj, color=ORANGE))
        self.wait(2)
        
        self.play(FadeOut(ax1), FadeOut(action_text), FadeOut(result_text))

        # --- AXIOM 3: IDENTITY ---
        # There exists e in G such that a (x) e = a
        # (Doing Identity out of order for better visual flow)
        ax3 = show_axiom("3. Identity", r"\exists e \in \mathcal{G}: a \otimes e = e \otimes a = a")
        
        action_text = MathTex(r"\text{Apply } e \text{ (The 'Do-Nothing' Action)}").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Action e: Pulse to show an action occurred, but geometry is invariant
        self.play(obj.animate.scale(1.1), run_time=0.5)
        self.play(obj.animate.scale(1/1.1), run_time=0.5)
        
        self.wait(2)
        self.play(FadeOut(ax3), FadeOut(action_text))

        # --- AXIOM 4: INVERSE ---
        # For each a in G, there exists a^-1 in G
        ax4 = show_axiom("4. Inverse", r"\forall a \in \mathcal{G}, \exists a^{-1}: a \otimes a^{-1} = e")
        
        action_text = MathTex(r"\text{Apply } r \text{, then apply } r^{-1}").to_corner(UR).shift(DOWN * 2)
        self.play(Write(action_text))
        
        # Action r: Rotate 120 degrees counter-clockwise
        self.play(Rotate(obj, angle=2*PI/3, about_point=obj.get_center()), run_time=1.5)
        self.wait(0.5)
        
        # Action r^-1: Rotate 120 degrees clockwise (returns to start)
        self.play(Rotate(obj, angle=-2*PI/3, about_point=obj.get_center()), run_time=1.5)
        
        result_text = MathTex(r"= e \text{ (Returned to original state)}").next_to(action_text, DOWN)
        self.play(Write(result_text))
        self.play(Indicate(obj, color=GREEN))
        self.wait(2)
        
        self.play(FadeOut(ax4), FadeOut(action_text), FadeOut(result_text))

        # --- AXIOM 2: ASSOCIATIVITY ---
        # (a x b) x c = a x (b x c)
        ax2 = show_axiom("2. Associativity", r"(a \otimes b) \otimes c = a \otimes (b \otimes c)")
        
        desc = Text("Transformation paths may differ,\nbut the final state is identical.", font_size=24).next_to(ax2, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(desc))
        
        # Highlight parentheses shifting
        self.play(Indicate(ax2[2], color=YELLOW)) # Highlighting the math text
        self.wait(3)

        # Final wrap up
        self.play(
            FadeOut(ax2), FadeOut(desc), FadeOut(obj),
            FadeOut(title), FadeOut(definition)
        )
        
        final_text = Text("A Group is the algebra of symmetry.", font_size=48, color=BLUE)
        self.play(Write(final_text))
        self.wait(2)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = AdvancedGroupAxioms()
    scene.render()
