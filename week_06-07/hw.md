## домашняя работа про перцептрон

Коротко:
Реализуйте класс MLP, представляющий многослойный перцептрон. 
Релизуйте возможность сохранить веса в файлы и загрузить их обратно. 
Убедитесь в том, чтоб перцептрон обучается.



### часть 1
реализуйте классы 

* ActivationF
* TanhF(ActivationF)
* Unit
* Layer
* MLP


### ActivationF, TanhF
класс, представляющий функцию активации.  
* умеет при заданном входном числе (если вам нужно, то каких-то ещё параметрах) вычислить значение функции активации.  
* умеет при заданном том же входном числе (если вам нужно, то каких-то ещё параметрах) вычислить значение производной функции активации.  
* математика активации -- та, что указана в имени класса (tanh), можно реализовать ещё несколько.

### Unit
класс `Unit` будет представлять нейрон в нейронной сети.  
словесно работает нейрон так: 
1. нейрону на вход подаётся набор чисел, 
2. он считает их взвешенную сумму, 
3. применяет к ней функцию активации.

Например,
```python3
class Unit:
    def __init__(self, input_size, prev_layer: "Layer" = None, activation="tanh"):
        self.input_size = input_size
        self.activation = ActivationF.from_name(activation)
        self.weights = []
        for weight_n in range(input_size):
            self.weights.append(random.uniform(-1, 1))
        self.prev_layer = prev_layer
        # your code here

    # your code here

    def forward(self):
        # your code here

    def backward(self, learning_rate=0.01):
        # your code here

```

#### описание класса:
в конструкторе нейрон должен принимать список объектов класса нейрон и объект класса `параметрыНС`. 
* список нейронов надо запомнить в поле объекта `previous_neurons`.
* из параметров НС надо выделить относящуюся к текущему нейрону функцию активации, создать соответствующий объект и записать его в поле `self.activation_f` создаваемого нейрона
* инициализировать поле `self.current_iter` нулём
* инициализировать поле `self.output` `None`
* инициализировать список весов, входящий в нейрон, списком случайных чисел от 0 до 1, длиной, равной установленному выше списку нейронов

в нейроне должен быть метод `forward`, он должен 
* вычислять значение нейрона, беря в качестве входных чисел значения `.output` каждого из `previous_neurons`, 
* запоминать его в self.output
* возвращать вычисленное значение

В начале работы `forward` должен увеличиваться счётчик `current_iter`.  
Если `current_iter` нейронов с прошлого слоя не совпадает с `current_iter` текущего слоя, то что-то идёт не так, и надо кидать ошибку.

### layer
Должен иметь возможность создать очередной слой перцептрона. Например, 
```python3
class Layer:
    def __init__(self, input_size, size, prev_layer = None, activation="tanh"):
        self.n_units = size
        self.activation = activation
        self.units = []
        for unit_n in range(self.n_units):
            self.units.append(Unit(input_size, prev_layer=prev_layer, activation=activation))

    def forward(self):
        return [unit.forward() for unit in self.units]

    def backward(self):
        return [unit.backward() for unit in self.units]
```
 

### perceptron
MLP должен 
* иметь произвольное кол-во слоёв произвольных размерностей
* обучатьтся
* уметь сохранять/загружать веса

Например, 
```python3
class MLP:
    def __init__(self, n_layers, input_size, output_size, sizes, activation="tanh"):
        self.n_layers = n_layers
        self.activation = activation
        self.layers = []

        sizes = [input_size] + sizes + [output_size]
        
        prev_added_layer = None
        for layer_in_size, layer_out_size in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Layer(layer_in_size, layer_out_size, prev_layer=prev_added_layer))
            prev_added_layer = self.layers[-1]

    def train_single_entry(self, features, target_mhe):
        # region forward pass
        for unit in self.layers[0].units:
            unit.features = features

        for layer in self.layers:
            curr_layer_output = layer.forward()
        # endregion forward pass

        # лосс с софтмаксом удобно считать вместе -- 
        # уж больно хорошая получается производная ошибки
        # по выходу с последнего лося
        def softmax(some_data):
            es_x = [math.e ** x for x in some_data]
            return [e_x / sum(es_x) for e_x in es_x]

        pred = softmax(curr_layer_output)
        loss = - sum(class_target * math.log(class_pred)
                     for class_target, class_pred in zip(target_mhe, pred))
        # region backward pass
        # your code here
        # endregion backward pass

        return loss
```