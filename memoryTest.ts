import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor, Tensor } from '@tensorflow/tfjs';
import { Log } from './log';

class Incrementayer extends tf.layers.Layer {
  constructor() {
    super({});
  }
  computeOutputShape(inputShape) { return inputShape; }

  // call() is where we do the computation.
  call(input: tf.Tensor | tf.Tensor[], kwargs) {
    if (input instanceof tf.Tensor) {
      // Test does not exercies this code.
      const result = input.add(0.5).add(0.5);
      return result;
    } else {
      // Test does exercise this code.
      const result = [];
      for (const i of input) {
        result.push(tf.add(i, 0.5).add(0.5));
      }
      return result;
    }
  }

  // Every layer needs a unique name.
  getClassName() { return 'Increment'; }
}


export class MemoryTest {
  static simpleAdd() {
    let t = tf.tensor([0], [1]);
    Log.info(tf.memory())
    for (let i = 0; i < 100; ++i) {
      t = tf.add(1, t);
    }
    Log.info(tf.memory())
    Log.info(t.dataSync());
  }

  static detachLoop() {
    let t = tf.tensor([0], [1]);
    Log.info(tf.memory())
    for (let i = 0; i < 100; ++i) {
      let t2 = tf.clone(tf.add(1, t));
      t2.dataSync();
      t.dispose();
      t = t2;
    }
    Log.info(tf.memory())
    Log.info(t.dataSync());
  }

  static detachLoop2() {
    let t = tf.tensor([0], [1]);
    Log.info(tf.memory())
    for (let i = 0; i < 100; ++i) {
      let t2 = tf.clone(tf.add(1, t));
      t.dispose();
      t = t2;
    }
    Log.info(tf.memory())
    Log.info(t.dataSync());
  }

  static assignLoop() {
    let t = tf.tensor([0], [1]);
    const v = tf.variable(t);
    const tadd = tf.add(v, 1);

    Log.info(tf.memory())
    for (let i = 0; i < 100; ++i) {
      const tsum = tadd.clone();
      tsum.dataSync();
      v.assign(tsum);
      v.dataSync();
      tsum.dispose();
    }
    Log.info(tf.memory())
    Log.info(v.dataSync());
  }

  static modelLoop() {
    const i = tf.input({ shape: [1] });
    const o = new Incrementayer().apply(i) as SymbolicTensor;
    const model = tf.model({ inputs: i, outputs: o });
    const one = tf.tensor([1], [1]);
    const v = tf.variable(one);

    Log.info(tf.memory())
    for (let i = 0; i < 100; ++i) {
      const a = model.predict(v) as tf.Tensor;
      const b = tf.reshape(a, [1]);
      v.assign(b);
      a.dispose();
      b.dispose();
    }
    Log.info(tf.memory())

    Log.info(v.dataSync());
  }

}