import random

# Define the size of the map
width = 11
height = 11

# Create an empty list to store the map data
map = []

robotPosition = (1, 1)

exploredMap = []


# Loop through the rows of the map
for y in range(height):
    # Create an empty list to store the data for a single row
    row = []

    # Loop through the columns of the map
    for x in reversed(range(width)):
        # Append an empty string to the row list to represent a tile on the map
        row.append(' ')

    # Append the row to the map
    map.append(row)
    exploredMap.append(row)



def addCoordinate(x, y):
    map[10 - y][x] = 'X'

# Data
addCoordinate(8, 0)
addCoordinate(3, 1)
addCoordinate(8, 1)
addCoordinate(1, 2)
addCoordinate(3, 2)
addCoordinate(5, 2)
addCoordinate(6, 2)
addCoordinate(10, 2)
addCoordinate(7, 4)
addCoordinate(8, 4)
addCoordinate(1, 5)
addCoordinate(6, 5)
addCoordinate(7, 5)
addCoordinate(1, 6)
addCoordinate(7, 6)
addCoordinate(8, 6)
addCoordinate(6, 7)
addCoordinate(7, 7)
addCoordinate(0, 8)
addCoordinate(5, 8)
addCoordinate(6, 8)
addCoordinate(4, 10)

def isFieldEmpty(x, y):
    return map[10 - y][x] == ' '

def addInformation(x, y, information):
    exploredMap[10 - y][x] = information

# Print the map to the console
for row in map:
    print(row)

def checkSurronding(rootX, rootY):
    if isFieldEmpty(x - 1, y):
        addInformation(x - 1, y, ' ')
    else:
        addInformation(x - 1, y, 'X')

    if isFieldEmpty(x + 1, y):
        addInformation(x + 1, y, ' ')
    else:
        addInformation(x + 1, y, 'X')

    if isFieldEmpty(x, y - 1):
        addInformation(x, y - 1, ' ')
    else:
        addInformation(x, y - 1, 'X')

    if isFieldEmpty(x, y + 1):
        addInformation(x, y + 1, ' ')
    else:
        addInformation(x, y + 1, 'X')

def isThereAnyObstaclesNextToRoot(x, y):


    if x - 1 >= 0 and not isFieldEmpty(x - 1, y):
        #exploredMap[10 - y][x - 1] = 'X'
        print("true")
        return True
    if x + 1 <= 10 and not isFieldEmpty(x + 1, y):
        #exploredMap[10 - y][x + 1] = 'X'
        print("true")
        return True
    if y - 1 >= 0 and not isFieldEmpty(x, y - 1):
        #exploredMap[10 - y - 1][x] = 'X'
        print("true")
        return True
    if y + 1 <= 10 and not isFieldEmpty(x, y + 1):
        #exploredMap[10 - y + 1][x] = 'X'
        print("true")
        return True

    return False




print(isThereAnyObstaclesNextToRoot(robotPosition[0], robotPosition[1]))

def walking():
    global robotPosition
    i = 0
    while i < 50:
        directions = ["up", "left", "down", "right"]
        choice = random.choice(directions)
        if choice == "up" and robotPosition[1] + 1 <= 10 and isFieldEmpty(robotPosition[0], robotPosition[1] + 1):
            robotPosition = (robotPosition[0], robotPosition[1] + 1)
            print("up")
            i += 1
        if choice == "left" and robotPosition[0] - 1 >= 0 and isFieldEmpty(robotPosition[0] - 1, robotPosition[1]):
            robotPosition = (robotPosition[0] - 1, robotPosition[1])
            print("left")
            i += 1
        if choice == "down" and robotPosition[1] - 1 >= 0 and isFieldEmpty(robotPosition[0], robotPosition[1] - 1):
            robotPosition = (robotPosition[0], robotPosition[1] - 1)
            print("down")
            i += 1
        if choice == "right" and robotPosition[0] + 1 <= 10 and isFieldEmpty(robotPosition[0] + 1, robotPosition[1]):
            robotPosition = (robotPosition[0] + 1, robotPosition[1])
            print("right")
            i += 1


walking()
print(robotPosition)
