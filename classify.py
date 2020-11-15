import numpy as np
import pygame
import sys
import os

from pygame.locals import *
from neu_net import Neural_Network

width = 600
height = 400

PIXELS = 784
TRAIN_SIZE = 800
TOTAL = 1000

dataDir = "data"

doodle = []
doodleLabel = []

def load_data(dataDir):
    for numpyFile in os.listdir(dataDir):
        doodle.append(numpyFile)
        doodleLabel.append(numpyFile.split('.')[0].upper())

load_data(dataDir)

def prepare_data(doodle_data, label):
    data = np.load(os.path.join(dataDir, doodle_data))
    data_train = data[0:TRAIN_SIZE, :]
    data_test = data[TRAIN_SIZE:TOTAL, :]
    labels_train = [label for _ in range(len(data_train))]
    labels_test = [label for _ in range(TOTAL - TRAIN_SIZE)]
    for_train = {"training": data_train, "label": labels_train}
    for_test = {"testing": data_test, "label": labels_test}
    return for_train, for_test

trainingData = []
trainingLabel = []
testingData = []
testingLabel = []

for i in range(len(doodle)):
    forTrain, forTest = prepare_data(doodle[i], doodleLabel.index(doodleLabel[i]))
    trainingData.extend(forTrain["training"])
    trainingLabel.extend(forTrain["label"])
    testingData.extend(forTest["testing"])
    testingLabel.extend(forTest["label"])

nn = Neural_Network(784, 64, 7, 0.01)

# Randomizing the data
def shuffled_data(a, b):
    data = np.array(a)
    label = np.array(b)
    assert len(data) == len(label)
    p = np.random.permutation(len(data))
    return data[p], label[p]

# Train function
def train_epoch(training, labels):
    train_list, label_list = shuffled_data(training, labels)
    for i in range(len(train_list)):
    # for i in range(1):
        data = train_list[i]
        inputs = data.ravel() / 255
        lbl = label_list[i] 
        targets = [0] * 7
        targets[lbl] = 1
        nn.train(inputs, targets)

def test_all(testing, labels):
    correct = 0
    for i in range(len(testing)):
    # for i in range(1):
        data = testing[i]
        inputs = data.ravel() / 255
        label = labels[i] 
        guess = nn.predict(inputs)
        classification = np.argmax(guess)
        if classification == label:
            correct += 1
    percent = round((100 * correct / len(testing)),2)
    print("% Correct:", percent)



# Guess function
def guess(inputs):
    clss = nn.predict(inputs)
    # print("Class:", clss)
    output = np.argmax(clss)
    # print("Output:",output)
    return doodleLabel[output]

pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Doodle Classifier")

clock = pygame.time.Clock()

OPEN_SANS = "OpenSans-Regular.ttf"
buttonFont = pygame.font.Font(OPEN_SANS, 28)
displayFont = pygame.font.Font(OPEN_SANS, 20)

BOARD_PADDING = 20
board_width = ((2 / 3) * width) - (BOARD_PADDING * 2)
board_height = height
board_rect = pygame.Rect(0, 0, board_width, board_height)

button_TrainRect = pygame.Rect((width * 4 / 6), (2.5 / 10) * height, width / 4, 50)
buttonTrain = buttonFont.render("Train", True, "black")
buttonTrainRect = buttonTrain.get_rect()
buttonTrainRect.center = button_TrainRect.center

button_GuessRect = pygame.Rect((width * 4 / 6), (4 / 10) * height, width / 4, 50)
buttonGuess = buttonFont.render("Guess", True, "black")
buttonGuessRect = buttonGuess.get_rect()
buttonGuessRect.center = button_GuessRect.center

button_ResetRect = pygame.Rect((width * 4 / 6), (5.5 / 10) * height, width / 4, 50)
buttonReset = buttonFont.render("Clear", True, "black")
buttonResetRect = buttonReset.get_rect()
buttonResetRect.center = button_ResetRect.center

def display(string):
    text = string
    text = displayFont.render(text, True, "white")
    textRect = text.get_rect()
    textRect.center = ((4.8 / 6) * width, (2.5 / 3) * height)
    screen.blit(text, textRect)


ROWS, COLS = 56, 56

OFFSET = 20
CELL_SIZE = 5

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

handwriting = [[0] * COLS for _ in range(ROWS)]

screen.fill("black")

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        elif event.type == MOUSEBUTTONDOWN and event.button == 1:
            mouse = pygame.mouse.get_pos()
            if button_TrainRect.collidepoint(mouse):
                print("Training...")
                for i in range(10):
                    print("Epoch:", i+1)
                    train_epoch(trainingData, trainingLabel)
                    test_all(testingData, testingLabel)
                print("Training Done")

            elif buttonGuessRect.collidepoint(mouse):
                pixelData = np.array(handwriting)
                pixelData = 255 * pixelData

                surf = pygame.surfarray.make_surface(pixelData)
                surf = pygame.transform.scale(surf, (28, 28))

                inputs = pygame.surfarray.array2d(surf)
                inputs = np.array(inputs) / 255
                inputs = inputs.reshape(784,)
                
                output = guess(inputs)
                display(output)

            elif mouse and buttonResetRect.collidepoint(mouse):
                handwriting = [[0] * COLS for _ in range(ROWS)]
                screen.fill("black")

    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE, OFFSET + i * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    pygame.draw.rect(screen, "black", board_rect, 10)
    # pygame.draw.rect(screen, 'black', panel_rect)

    pygame.draw.rect(screen, "white", button_TrainRect)
    screen.blit(buttonTrain, buttonTrainRect)

    pygame.draw.rect(screen, "white", button_GuessRect)
    screen.blit(buttonGuess, buttonGuessRect)

    pygame.draw.rect(screen, "white", button_ResetRect)
    screen.blit(buttonReset, buttonResetRect)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()