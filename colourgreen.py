import cv2
import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Color Catcher")

# Basket properties
basket_width, basket_height = 100, 20
basket_x = WIDTH // 2 - basket_width // 2
basket_y = HEIGHT - 40

# Falling ball properties
ball_radius = 15
ball_x = np.random.randint(ball_radius, WIDTH - ball_radius)
ball_y = 0
ball_speed = 5

# Score
score = 0
font = pygame.font.SysFont(None, 40)

# OpenCV camera setup
cap = cv2.VideoCapture(0)

# HSV range for detecting light green object
lower_color = np.array([35, 80, 80])   # lower HSV boundary for green
upper_color = np.array([85, 255, 255]) # upper HSV boundary for green

clock = pygame.time.Clock()

while True:
    # --- Pygame event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    # --- OpenCV part ---
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of the green object
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            basket_x = int(x) - basket_width // 2  # Move basket with green object
            if basket_x < 0:
                basket_x = 0
            elif basket_x > WIDTH - basket_width:
                basket_x = WIDTH - basket_width

    # --- Game logic ---
    ball_y += ball_speed
    if ball_y > HEIGHT:
        ball_y = 0
        ball_x = np.random.randint(ball_radius, WIDTH - ball_radius)

    # Collision detection
    if basket_y - ball_radius < ball_y < basket_y + basket_height:
        if basket_x < ball_x < basket_x + basket_width:
            score += 1
            ball_y = 0
            ball_x = np.random.randint(ball_radius, WIDTH - ball_radius)

    # --- Drawing ---
    win.fill((30, 30, 30))
    pygame.draw.rect(win, (0, 200, 0), (basket_x, basket_y, basket_width, basket_height))
    
    # Light green ball 🎾
    pygame.draw.circle(win, (144, 238, 144), (ball_x, ball_y), ball_radius)
    
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(score_text, (10, 10))

    pygame.display.update()
    clock.tick(30)

    # Show OpenCV window for debugging
    cv2.imshow("Mask", mask)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
