with open(r'H:\pythonProject\yolov1test\tests.txt') as f:
    bbox = f.read().split(' ')
    print(bbox)
    results = [float(x) for x in bbox]
    print(results)
    for i in range(1):
        print(i)