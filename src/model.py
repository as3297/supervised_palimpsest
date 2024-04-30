from tensorflow.keras import Model

class FCModel(Model):
  def __init__(self):
    super().__init__(nb_features)
    self.nb_features = nb_features
    self.d1 = Dense(nb_features, activation='relu')
    self.d2 = Dense(nb_features, activation='relu')
    self.d3 = Dense(2)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()