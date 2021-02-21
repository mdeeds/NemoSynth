import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import { Log } from './log';

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

}