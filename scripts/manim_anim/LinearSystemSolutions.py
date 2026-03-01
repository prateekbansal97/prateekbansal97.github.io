from manim import *

class LinearSystemSolutions(ThreeDScene):
    def construct(self):
        # 1. Title of the scene
        title = MathTex(r"\text{Solutions to Systems of Linear Equations}", font_size=40).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # 2. Text for Case 1: Unique Solution
        text_case1 = VGroup(
            MathTex(r"\text{Case 1: The system has a}", color=BLUE),
            MathTex(r"\text{unique solution}", color=BLUE)
        ).arrange(DOWN).to_corner(UL).shift(DOWN * 1)
        
        # 3. Text for Case 2: No Solution
        text_case2 = VGroup(
            MathTex(r"\text{Case 2: The system has}", color=RED),
            MathTex(r"\text{no solution}", color=RED)
        ).arrange(DOWN).to_corner(UL).shift(DOWN * 1)

        # 4. Text for Case 3: Infinitely Many Solutions
        text_case3 = VGroup(
            MathTex(r"\text{Case 3: The system has}", color=GREEN),
            MathTex(r"\text{infinitely many solutions}", color=GREEN)
        ).arrange(DOWN).to_corner(UL).shift(DOWN * 1)

        # 5. 3D Axes setup
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=6,
            axis_config={"include_tip": True, "tip_shape": StealthTip}
        )
        self.play(Create(axes))
        
        # Animate the camera movement from 2D to 3D view
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)

        # --- CASE 1: Unique Solution ---
        self.add_fixed_in_frame_mobjects(text_case1)
        self.play(FadeIn(text_case1))
        
        # Planes intersecting at a single point 
        plane1 = Surface(lambda u, v: axes.c2p(u, v, 2*u + v), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=BLUE_E)
        plane2 = Surface(lambda u, v: axes.c2p(u, v, -u + v/2), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=BLUE_C)
        plane3 = Surface(lambda u, v: axes.c2p(u, v, u/2 - 2*v), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=BLUE_A)
        intersection = Sphere(radius=0.1)
        intersection.set_color(YELLOW)	 
        
        self.play(Create(plane1))
        self.play(Create(plane2))
        self.play(Create(plane3))
        self.play(Create(intersection))
        self.wait(2)

        # --- CASE 2: No Solution ---
        self.play(
            FadeOut(plane1, plane2, plane3, intersection),
            FadeOut(text_case1)
        )
        self.add_fixed_in_frame_mobjects(text_case2)
        self.play(FadeIn(text_case2))
        
        # Parallel planes (never intersecting all together)
        plane1_para = Surface(lambda u, v: axes.c2p(u, v, u), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=RED_E)
        plane2_para = Surface(lambda u, v: axes.c2p(u, v, u + 1), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=RED_C)
        plane3_para = Surface(lambda u, v: axes.c2p(u, v, u - 1), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=RED_A)
        
        self.play(Create(plane1_para))
        self.play(Create(plane2_para))
        self.play(Create(plane3_para))
        self.wait(2)

        # --- CASE 3: Infinitely Many Solutions ---
        self.play(
            FadeOut(plane1_para, plane2_para, plane3_para),
            FadeOut(text_case2)
        )
        self.add_fixed_in_frame_mobjects(text_case3)
        self.play(FadeIn(text_case3))
        
        # All planes intersecting along a single line (the z-axis)
        plane1_line = Surface(lambda u, v: axes.c2p(u, 0, v), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=GREEN_E) # xz-plane
        plane2_line = Surface(lambda u, v: axes.c2p(0, u, v), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=GREEN_C) # yz-plane
        plane3_line = Surface(lambda u, v: axes.c2p(u, u, v), u_range=[-1.5, 1.5], v_range=[-1.5, 1.5], fill_opacity=0.6, color=GREEN_A) # diagonal plane
        
        # Highlight the intersection line (z-axis)
        line = Line(start=axes.c2p(0, 0, -2), end=axes.c2p(0, 0, 2), color=YELLOW, stroke_width=6)
        
        self.play(Create(plane1_line))
        self.play(Create(plane2_line))
        self.play(Create(plane3_line))
        self.play(Create(line))
        self.wait(2)

        # --- CLEANUP & CONCLUSION ---
        self.play(
            FadeOut(plane1_line, plane2_line, plane3_line, line),
            FadeOut(text_case3), 
            FadeOut(axes)
        )
        
        # Move camera back to 2D view for final text
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=1.5)
        
        final_text = VGroup(
            MathTex(r"\text{These three cases are comprehensive}"),
            MathTex(r"\text{for any system of linear equations.}")
        ).arrange(DOWN).scale(1.2)
        
        self.add_fixed_in_frame_mobjects(final_text)
        self.play(Transform(title, final_text))
        self.wait(3)
