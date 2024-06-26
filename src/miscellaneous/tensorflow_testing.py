import tensorflow as tf

class New():
    def __init__(self):
        self.a = tf.Variable(3.0)
    @tf.function()
    def adding(self,x):
        print("Tracing")
        b = x+self.a
        return b

if __name__=="__main__":
    n = New()
    x = tf.constant(1.0)
    print(n.adding(x))
    n.a.assign(10.0)
    print(n.adding(x))