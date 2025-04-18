# Self Driving Car


# Importing the libraries
import numpy as np
from random import random, randint

import matplotlib.pyplot as plt
import time


# Importing the Kivy packages

from kivy.app import App

from kivy.uix.widget import Widget

from kivy.uix.button import Button

from kivy.graphics import Color, Ellipse, Line

from kivy.config import Config

from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty

from kivy.vector import Vector

from kivy.clock import Clock

from kivy.core.image import Image as CoreImage

from PIL import Image as PILImage


from ai import TD3
import torch

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

Config.set('graphics', 'resizable', False)

Config.set('graphics', 'width', '1786')

Config.set('graphics', 'height', '777')


last_x = 0
last_y = 0
n_points = 0

length = 0

# Initialize TD3 with continuous action space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TD3(state_dim=7, action_dim=1, max_action=5.0)  # Single continuous action for rotation

last_reward = 0

scores = []

im = CoreImage("./images/road1.png")


# Initializing the map

first_update = True

def init():

    global sand

    global goal_x

    global goal_y

    global first_update

    global longueur

    global largeur

    # Initialize sand array with window dimensions

    longueur = int(Config.get('graphics', 'width'))

    largeur = int(Config.get('graphics', 'height'))

    sand = np.zeros((longueur, largeur))

    img = PILImage.open("./images/road.png").convert('L')

    # Resize image to match window dimensions

    img = img.resize((longueur, largeur))

    sand = np.asarray(img)/255

    sand = sand.T  # Transpose to match width/height dimensions

    goal_x = 1000

    goal_y = 520

    first_update = False

    global swap

    swap = 0



# Initializing the last distance
last_distance = 0


# Creating the car class


class Car(Widget):
    

    angle = NumericProperty(0)

    rotation = NumericProperty(0)

    velocity_x = NumericProperty(0)

    velocity_y = NumericProperty(0)

    velocity = ReferenceListProperty(velocity_x, velocity_y)

    sensor1_x = NumericProperty(0)

    sensor1_y = NumericProperty(0)

    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    sensor2_x = NumericProperty(0)

    sensor2_y = NumericProperty(0)

    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

    sensor3_x = NumericProperty(0)

    sensor3_y = NumericProperty(0)

    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    signal1 = NumericProperty(0)

    signal2 = NumericProperty(0)

    signal3 = NumericProperty(0)


    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos

        # Clamp car position immediately after movement
        self.x = min(max(self.x, 10), longueur - 10)
        self.y = min(max(self.y, 10), largeur - 10)
        
        # Convert continuous action to rotation
        self.rotation = float(rotation)  # Direct continuous control

        self.angle = self.angle + self.rotation

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos

        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos

        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        

        # Ensure sensor positions are within bounds

        self.sensor1_x = min(max(int(self.sensor1_x), 10), longueur - 10)

        self.sensor1_y = min(max(int(self.sensor1_y), 10), largeur - 10)

        self.sensor2_x = min(max(int(self.sensor2_x), 10), longueur - 10)

        self.sensor2_y = min(max(int(self.sensor2_y), 10), largeur - 10)

        self.sensor3_x = min(max(int(self.sensor3_x), 10), longueur - 10)

        self.sensor3_y = min(max(int(self.sensor3_y), 10), largeur - 10)
        

        # Calculate signals using bounded sensor positions

        if self.sensor1_x < 10 or self.sensor1_x > longueur-10 or self.sensor1_y < 10 or self.sensor1_y > largeur-10:
            self.signal1 = 1.

        else:

            self.signal1 = int(np.sum(sand[int(self.sensor1_y)-10:int(self.sensor1_y)+10, int(self.sensor1_x)-10:int(self.sensor1_x)+10]))/400.
            

        if self.sensor2_x < 10 or self.sensor2_x > longueur-10 or self.sensor2_y < 10 or self.sensor2_y > largeur-10:

            self.signal2 = 1.

        else:

            self.signal2 = int(np.sum(sand[int(self.sensor2_y)-10:int(self.sensor2_y)+10, int(self.sensor2_x)-10:int(self.sensor2_x)+10]))/400.
            

        if self.sensor3_x < 10 or self.sensor3_x > longueur-10 or self.sensor3_y < 10 or self.sensor3_y > largeur-10:

            self.signal3 = 1.

        else:

            self.signal3 = int(np.sum(sand[int(self.sensor3_y)-10:int(self.sensor3_y)+10, int(self.sensor3_x)-10:int(self.sensor3_x)+10]))/400.
        

        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.

        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:

            self.signal2 = 10.

        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:

            self.signal3 = 10.
        


class Ball1(Widget):
    pass

class Ball2(Widget):
    pass

class Ball3(Widget):
    pass


# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.pos = (822,452)
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global model
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        longueur = int(Config.get('graphics', 'width'))
        largeur = int(Config.get('graphics', 'height'))

        if first_update:
            init()

        # Ensure car position stays within bounds
        car_x = min(max(int(self.car.x), 10), longueur - 10)
        car_y = min(max(int(self.car.y), 10), largeur - 10)
        
        # Update car position to stay within bounds
        self.car.x = car_x
        self.car.y = car_y

        # Calculate state inputs with enhanced position awareness
        xx = goal_x - car_x
        yy = goal_y - car_y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        velocity_magnitude = np.sqrt(self.car.velocity[0]**2 + self.car.velocity[1]**2)
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation, car_x/longueur, car_y/largeur]
        
        # Get continuous action from TD3
        action = model.update(last_reward, last_signal)
        scores.append(model.score())
        
        # Apply continuous rotation directly
        self.car.move(action[0])  # Use first dimension of action as rotation
        
        distance = np.sqrt((car_x - goal_x)**2 + (car_y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Enhanced reward shaping for continuous control and anti-stuck mechanism
        if 0 <= car_x < longueur and 0 <= car_y < largeur and sand[car_x, car_y] > 0:
            self.car.velocity = Vector(1.0, 0).rotate(self.car.angle)  # Increased base speed in sand
            print(1, goal_x, goal_y, distance, car_x, car_y, im.read_pixel(car_x, car_y))
            last_reward = -1.5  # Reduced penalty to encourage exploration
        else:
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)  # Increased base speed
            # Enhanced continuous reward based on distance improvement and velocity
            distance_improvement = last_distance - distance
            velocity_reward = min(velocity_magnitude / 3.0, 1.0)  # Reward for maintaining speed
            last_reward = distance_improvement * 0.8 + velocity_reward * 0.4  # Balanced reward
            
            print(0, goal_x, goal_y, distance, car_x, car_y, im.read_pixel(car_x, car_y))
            
            # Additional reward for staying on road
            last_reward += 0.1
            
            # Penalty for excessive rotation
            rotation_penalty = -0.1 * abs(action[0]) / model.max_action
            last_reward += rotation_penalty

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -2.0
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -2.0
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -2.0
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -2.0

        if distance < 25:
            if swap == 1:
                goal_x = 1000
                goal_y = 520
                swap = 0
            else:
                goal_x = 1040
                goal_y = 158
                swap = 1
            last_reward = 10.0  # Bonus for reaching goal
            
        last_distance = distance


# Adding the painting tools


class MyPaintWidget(Widget):


    def on_touch_down(self, touch):

        global length, n_points, last_x, last_y

        with self.canvas:

            Color(0.8,0.7,0)
            d = 10.

            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)

            last_x = int(touch.x)

            last_y = int(touch.y)
            n_points = 0

            length = 0

            sand[int(touch.x),int(touch.y)] = 1

            img = PILImage.fromarray(sand.astype("uint8")*255)

            img.save("./images/sand.jpg")


    def on_touch_move(self, touch):

        global length, n_points, last_x, last_y

        if touch.button == 'left':

            touch.ud['line'].points += [touch.x, touch.y]

            x = int(touch.x)

            y = int(touch.y)

            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))

            n_points += 1.

            density = n_points/(length)

            touch.ud['line'].width = int(20 * density + 1)

            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            

            last_x = x
            last_y = y


# Adding the API Buttons (clear, save and load)


class CarApp(App):


    def build(self):

        parent = Game()

        parent.serve_car()

        Clock.schedule_interval(parent.update, 1.0/60.0)

        self.painter = MyPaintWidget()

        clearbtn = Button(text = 'clear')

        savebtn = Button(text = 'save', pos = (parent.width, 0))

        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))

        clearbtn.bind(on_release = self.clear_canvas)

        savebtn.bind(on_release = self.save)

        loadbtn.bind(on_release = self.load)

        parent.add_widget(self.painter)

        # parent.add_widget(clearbtn)

        # parent.add_widget(savebtn)

        # parent.add_widget(loadbtn)
        return parent


    def clear_canvas(self, obj):

        global sand

        self.painter.canvas.clear()

        sand = np.zeros((longueur,largeur))


    def save(self, obj):

        print("saving brain...")
        model.save()
        plt.plot(scores)

        plt.show()


    def load(self, obj):

        print("loading last saved brain...")
        model.load()


# Running the whole thing

if __name__ == '__main__':

    CarApp().run()

