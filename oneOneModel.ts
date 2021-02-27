import * as tf from '@tensorflow/tfjs';
import { FuncCanvas } from './funcCanvas';

export class OneOneModel {
  model: tf.LayersModel;

  constructor() {
    const input = tf.input({ shape: [1] });
    const l1 = tf.layers.dense({ units: 3, activation: 'tanh' }).apply(input);
    const l2 = tf.layers.dense({ units: 13, activation: 'tanh' }).apply(l1);
    const l3 = tf.layers.dense({ units: 1, activation: 'hardSigmoid' })
      .apply(l2) as tf.SymbolicTensor;
    this.model = tf.model({ inputs: input, outputs: l3 });
    this.model.compile({
      optimizer: tf.train.sgd(0.05),
      loss: 'meanSquaredError'
    });
  }

  getXs(func: FuncCanvas) {
    const n = func.func.length;
    const xs = new Float32Array(n);
    for (let i = 0; i < func.func.length; ++i) {
      xs[i] = i / func.func.length;
    }
    return xs;
  }

  learnFromFunction(func: FuncCanvas, maxIter = 100) {
    const n = func.func.length;
    const xTensor = tf.tensor(this.getXs(func), [n, 1]);
    const yTensor = tf.tensor(func.func, [n, 1]);

    return new Promise((resolve, reject) => {
      this.model.fit(xTensor, yTensor, {
        epochs: 100,
        batchSize: n,
      }).then((history: tf.History) => {
        xTensor.dispose();
        yTensor.dispose();
        const mses = history.history['loss'];
        const mse = mses[mses.length - 1];
        console.log(`MSE: ${mse}`);
        if (typeof (mse) === "number") {
          if (mse > 0.01 && maxIter > 0) {
            this.learnFromFunction(func, maxIter - 1).then((history) => {
              resolve(history);
            });
            return;
          }
        }
        resolve(history);
      });
    });
  }

  applyToFunction(func: FuncCanvas) {
    let n = func.func.length;
    const x = tf.tensor(this.getXs(func), [n, 1]);
    const y = this.model.predict(x) as tf.Tensor;
    y.data().then((data) => {
      for (let i = 0; i < n; ++i) {
        func.func[i] = data[i];
      }
    });
  }






}