import * as tf from '@tensorflow/tfjs';

export class SpringLayer extends tf.layers.Layer {
  private k: tf.Tensor;
  private frequencyResolution: number;
  constructor(k: tf.Tensor) {
    super({});
    this.k = k;
    [this.frequencyResolution] = k.shape;
  }

  private computeOutputShapeOne(inputShape: tf.Shape): tf.Shape {
    const result = [inputShape[0], inputShape[1], this.frequencyResolution]
    return result;
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
    if (Array.isArray(inputShape)) {
      const result: tf.Shape[] = [];
      for (const s of (inputShape as tf.Shape[])) {
        result.push(this.computeOutputShapeOne(s));
      }
      return result;
    } else {
      return this.computeOutputShapeOne(inputShape);
    }
  }

  private callOne(input: tf.Tensor) {
    const [batchSize, sampleCount] = input.shape;
    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);

    const combinedTensors: tf.Tensor[] = [];

    let previousXs = tf.zeros([batchSize, this.frequencyResolution]);
    let previousVs = tf.zeros([batchSize, this.frequencyResolution]);

    let previousYs = tf.zeros([batchSize, this.frequencyResolution]);
    // TODO: 0.02 here is actually related to how long it takes to
    // distinguish the sound... i.e. the boundary between rhythm and tone.
    const alpha = 0.02;
    for (let t = 0; t < sampleCount; ++t) {
      const currentAs = tf.add(
        tf.neg(tf.mul(previousXs, this.k)),
        tf.mul(tf.sub(timeSlices[t], previousXs), 0.0001));
      const currentVs = tf.add(previousVs, currentAs);
      const currentXs = tf.add(currentVs, previousXs);

      const currentYs = tf.add(
        tf.mul(tf.square(currentXs), alpha),
        tf.mul(previousYs, (1.0 - alpha)));

      combinedTensors.push(
        tf.reshape(tf.mul(currentYs, 100),
          [batchSize, 1, this.frequencyResolution]));

      previousXs = currentXs;
      previousVs = currentVs;
      previousYs = currentYs;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs) {
    if (Array.isArray(input)) {
      const result = [];
      for (const i of input) {
        result.push(this.callOne(i));
      }
      return result;
    } else {
      const result = this.callOne(input);
      return result;
    }
  }

  getClassName() { return 'SpringLayer'; }
}

