from manim import *

class NetworkScene(Scene):
    def construct(self):
        # Architecture: 5-6-6-4
        layers = [5, 6, 6, 4]
        dots = []
        for i, n in enumerate(layers):
            dots.append(VGroup(*[Dot(LEFT*3 + RIGHT*i*2 + UP*(j-(n-1)/2), radius=0.15) for j in range(n)]))
        for group in dots:
            self.add(group)
        # Connections
        for l in range(len(layers)-1):
            for i, d0 in enumerate(dots[l]):
                for j, d1 in enumerate(dots[l+1]):
                    line = Line(d0.get_center(), d1.get_center(), color=BLUE)
                    self.add(line)
        # Forward pass pulse
        for l in range(len(layers)-1):
            for i, d0 in enumerate(dots[l]):
                for j, d1 in enumerate(dots[l+1]):
                    pulse = Line(d0.get_center(), d1.get_center(), color=YELLOW)
                    self.play(Create(pulse), run_time=0.1)
                    self.remove(pulse)
        # Backpropagation color shift
        for l in reversed(range(len(layers)-1)):
            for i, d0 in enumerate(dots[l]):
                for j, d1 in enumerate(dots[l+1]):
                    line = Line(d0.get_center(), d1.get_center(), color=RED)
                    self.play(Create(line), run_time=0.1)
                    self.remove(line)
