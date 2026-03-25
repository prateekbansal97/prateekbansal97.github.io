from manim import *
import numpy as np

class LinearTransformations(Scene):
    def construct(self):
        # 1. Setup the Coordinate Axes
        axes = Axes(
            x_range=[-6, 6, 1], 
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=7,
            axis_config={"color": GREY, "include_ticks": False},
        )
        self.add(axes)

        # 2. Create the Data Grid with a Color Gradient
        dots = VGroup()
        dot_coords = [] # Store original logical coordinates for precise math
        
        # Create a 21x21 grid of points from -1.5 to 1.5
        for x in np.linspace(-1.5, 1.5, 21):
            for y in np.linspace(-1.5, 1.5, 21):
                # Calculate color gradient based on y-value (Blue at bottom, Yellow at top)
                alpha = (y + 1.5) / 3.0  # Normalizes y to a [0, 1] range
                color = interpolate_color(BLUE, YELLOW, alpha)
                
                dot = Dot(axes.c2p(x, y), radius=0.05, color=color)
                dots.add(dot)
                dot_coords.append(np.array([x, y]))
        self.play(FadeIn(dots), run_time=1.5)
        self.wait(1)

        # 3. Setup Matrices and Labels
        matrix_A1 = np.array([
            [np.cos(PI/4), -np.sin(PI/4)],
            [np.sin(PI/4),  np.cos(PI/4)]
        ])
        label_A1 = MathTex(
            r"\boldsymbol{A}_1 = \begin{bmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{bmatrix}"
        ).to_edge(UP).add_background_rectangle(opacity=0.8).shift(UP*0.3)
        title_A1 = Text("Rotation by 45°", font_size=28).next_to(label_A1, DOWN).add_background_rectangle(opacity=0.8)

        matrix_A2 = np.array([
            [2, 0],
            [0, 1]
        ])
        label_A2 = MathTex(
            r"\boldsymbol{A}_2 = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}"
        ).to_edge(UP).add_background_rectangle(opacity=0.8).shift(UP*0.3)
        title_A2 = Text("Stretch along the horizontal axis", font_size=28).next_to(label_A2, DOWN).add_background_rectangle(opacity=0.8)

        matrix_A3 = np.array([
            [1.5, -0.5],
            [0.5, -0.5]
        ])
        label_A3 = MathTex(
            r"\boldsymbol{A}_3 = \frac{1}{2} \begin{bmatrix} 3 & -1 \\ 1 & -1 \end{bmatrix}"
        ).to_edge(UP).add_background_rectangle(opacity=0.8).shift(UP*0.3)
        title_A3 = Text("General linear mapping", font_size=28).next_to(label_A3, DOWN).add_background_rectangle(opacity=0.8)

        # Helper function to animate transformation to a specific matrix
        def apply_transformation(target_matrix):
            anims = []
            for i, dot in enumerate(dots):
                # Matrix multiplication: A * v
                original_vec = dot_coords[i]
                new_vec = target_matrix @ original_vec
                # Move dot to the new calculated position
                anims.append(dot.animate.move_to(axes.c2p(new_vec[0], new_vec[1])))
            return anims

        # Helper function to revert grid back to original position
        def revert_to_original():
            anims = []
            for i, dot in enumerate(dots):
                orig_vec = dot_coords[i]
                anims.append(dot.animate.move_to(axes.c2p(orig_vec[0], orig_vec[1])))
            return anims

        # --- 4. Animate Matrix A1 (Rotation) ---
        self.play(Write(label_A1), FadeIn(title_A1))
        self.play(*apply_transformation(matrix_A1), run_time=2)
        self.wait(1.5)
        
        # Revert
        self.play(*revert_to_original(), FadeOut(label_A1, title_A1), run_time=1.5)
        self.wait(0.5)

        # --- 5. Animate Matrix A2 (Horizontal Stretch) ---
        self.play(Write(label_A2), FadeIn(title_A2))
        self.play(*apply_transformation(matrix_A2), run_time=2)
        self.wait(1.5)
        
        # Revert
        self.play(*revert_to_original(), FadeOut(label_A2, title_A2), run_time=1.5)
        self.wait(0.5)

        # --- 6. Animate Matrix A3 (General Mapping) ---
        self.play(Write(label_A3), FadeIn(title_A3))
        self.play(*apply_transformation(matrix_A3), run_time=2)
        self.wait(2)

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1920
    config.pixel_height = 1080
    scene = LinearTransformations()
    scene.render()
