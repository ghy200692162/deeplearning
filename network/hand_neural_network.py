import numpy as np 

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deriv_sigmoid(x):
    f = sigmoid(x)
    return f * (1-f)

def mse(y_true,y_pred):
    return ((y_true - y_pred)**2).mean()

class Neuron:
    def __init__(self,weights,bias):
        self.weights=weights
        self.bias=bias

    def foward(self,inputs):
        total = np.dot(self.weights,inputs) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:
    '''
        2 inputs(x1,x2)
        1 hiddern layer with 2 neuron(h1,h2)
        1 output layer with 1 neuron(o1)
    '''

    def __init__(self):
        self.w1=np.random.normal()
        self.w2=np.random.normal()
        self.w3=np.random.normal()
        self.w4=np.random.normal()
        self.w5=np.random.normal()
        self.w6=np.random.normal()


        self.b1=np.random.normal()
        self.b2=np.random.normal()
        self.b3=np.random.normal()


    def feedForward(self,x):
        h1 = sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2 = sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)
        o1 = sigmoid(self.w5*h1+self.w6*h2+self.b3)
        return o1
    def train(self,data,all_y_trues):

        lr = 0.01
        epochs=1000
        for  epoch in range(epochs):
            for x,y_true in zip(data,all_y_trues):
                sum_h1 = self.w1*x[0]+self.w2*x[1]+self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3*x[0]+self.w4*x[1]+self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5*h1+self.w6*h2+self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true-y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1*deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2*deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1) 
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)


                # Update weights and bias
                self.w1-=lr*d_L_d_ypred*d_ypred_d_h1*d_h1_d_w1
                self.w2-=lr*d_L_d_ypred*d_ypred_d_h1*d_h1_d_w2
                self.b1-=lr*d_L_d_ypred*d_ypred_d_h1*d_h1_d_b1

                self.w3-=lr*d_L_d_ypred*d_ypred_d_h2*d_h2_d_w3
                self.w4-=lr*d_L_d_ypred*d_ypred_d_h2*d_h2_d_w4
                self.b2-=lr*d_L_d_ypred*d_ypred_d_h2*d_h2_d_b2

                self.w5-=lr*d_L_d_ypred*d_ypred_d_w5
                self.w6-=lr*d_L_d_ypred*d_ypred_d_w6
                self.b3-=lr*d_L_d_ypred*deriv_sigmoid(sum_o1)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedForward, 1, data)
                loss = mse(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))            

# train network

data = np.array([
      [-2, -1],  # Alice
      [25, 6],   # Bob
      [17, 4],   # Charlie
      [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# inputs=np.array([2,3])
# # test output for singal neural unit
# weights=np.array([0,1])
# bias=4
# n=Neuron(weights,bias)
# print(n.foward(inputs))
# # test network forward cal
# nextwork = OurNeuralNetwork()
# output = nextwork.feedForward(inputs)
# print(output)

# # test mse loss
# y_true=np.array([1,0,0,1])
# y_pred=np.array([0,0,0,0])
# mse_loss = mse(y_true,y_pred)
# print(mse_loss)

