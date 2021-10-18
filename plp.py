"""

class addDatapoint:
    def __init__(self):
        self.X = []
        self.y = []
        self.pack = []

    def add(self, value):

        if(isinstance(value, list)):    # value == arr :: X
            self.pack.append(value)

        elif(isinstance(value, float)):    # value == int :: y
            if(not len(self.pack) == 0):  # prevent to push first
                self.pack.append(value)

                print(value)
                self.X.append(self.pack[0])
                self.y.append(self.pack[1])
               
                if(len(self.X) % 100 == 0):
                    print(self.X)
                    print(self.y)
                
                f = open("plpf.txt", "a")
                f.write(self.pack[0])
                f.write(self.pack[1])
                f.close()

                self.pack = []"""

class addDatapoint:
    def __init__(self):
        self.i = 0
        self.pack = []
        self.X = []
        self.y = []


    def add(self, value):
        if not self.i == 0:

            if(len(self.pack) == 1):
                self.pack.append(value)
                if(value > 1):
                    f = open("plpx/plpX.txt", "a")
                    f.write(str(self.pack[0]) + "\n")
                    f.close()
                    f = open("plpx/plpY.txt", "a")
                    f.write(str(self.pack[1]) + "\n")
                    f.close()
                    print(self.pack[1])
                self.pack = []
            else:
                self.pack.append(value)


        self.i += 1




addDatapoint = addDatapoint()
