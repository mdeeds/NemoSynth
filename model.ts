import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';

export class Model {
  // Input shape: [batchSize, sampleCount]
  // Alpha shape: [frequencyResolution]
  // Output shape: [ batchSize, sampleCount, frequencyResolution]
  static getLowPassOutputTensor(input: Tensor, alpha: Tensor): Tensor {
    console.assert(input.shape.length === 2, "Input shape");
    console.assert(alpha.shape.length === 1, "Alpha shape");
    const [batchSize, sampleCount] = input.shape;
    const [frequencyResolution] = alpha.shape;

    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);

    const alpha2 = tf.reshape(alpha, [1, frequencyResolution]);
    const oneMinusAlpha2 = tf.sub(tf.ones(alpha2.shape), alpha2);

    const combinedTensors: Tensor[] = [];

    let previousOutput = tf.zeros([batchSize, 1]);
    for (let t = 0; t < sampleCount; ++t) {
      const t1Part = tf.matMul(timeSlices[t], alpha2);
      const tMinusOnePart = tf.mul(previousOutput, oneMinusAlpha2);
      const sum = tf.add(t1Part, tMinusOnePart);
      const sum3 = tf.reshape(sum, [batchSize, 1, frequencyResolution]);
      combinedTensors.push(sum3);
      previousOutput = sum;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  // Input shape: [batchSize, sampleCount]
  // Alpha shape: [frequencyResolution]
  // Output shape: [ batchSize, sampleCount, frequencyResolution]
  static getBandPassOutputTensor(input: Tensor, alpha: Tensor): Tensor {
    const [batchSize, sampleCount] = input.shape;
    const [frequencyResolution] = alpha.shape;

    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);
    timeSlices.unshift(tf.zeros([batchSize, 1]));

    const alpha2 = tf.reshape(alpha, [1, frequencyResolution]);
    const oneMinusAlpha2 = tf.sub(tf.ones(alpha2.shape), alpha2);

    const combinedTensors: Tensor[] = [];

    let previousOutput = tf.zeros([batchSize, 1]);
    let previousSum3 = tf.zeros([batchSize, 1, frequencyResolution]);
    for (let t = 0; t < sampleCount; ++t) {
      const t1Part = tf.matMul(timeSlices[t + 1], alpha2);
      const tMinusOnePart = tf.mul(previousOutput, oneMinusAlpha2);
      const sum = tf.add(t1Part, tMinusOnePart);
      const sum3 = tf.reshape(sum, [batchSize, 1, frequencyResolution]);
      const output = tf.sub(sum3, previousSum3);
      combinedTensors.push(output);
      previousOutput = sum;
      previousSum3 = sum3;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  // Input shape: [batchSize, sampleCount]
  // Alpha shape: [frequencyResolution]
  // Output shape: [ batchSize, sampleCount, frequencyResolution]
  static getSpringOutputTensor(input: Tensor, k: Tensor): Tensor {
    const [batchSize, sampleCount] = input.shape;
    const [frequencyResolution] = k.shape;
    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);

    const combinedTensors: Tensor[] = [];

    let previousXs = tf.zeros([batchSize, frequencyResolution]);
    let previousVs = tf.zeros([batchSize, frequencyResolution]);

    // TODO: 0.02 here is actually related to how long it takes to
    // distinguish the sound... i.e. the boundary between rhythm and tone.
    const alpha = 0.02;
    for (let t = 0; t < sampleCount; ++t) {
      const currentAs = tf.add(
        tf.neg(tf.mul(previousXs, k)),
        tf.mul(tf.sub(timeSlices[t], previousXs), 0.0001));
      const currentVs = tf.add(previousVs, currentAs);
      const currentXs = tf.add(currentVs, previousXs);

      combinedTensors.push(
        tf.reshape(currentXs, [batchSize, 1, frequencyResolution]));

      previousXs = currentXs;
      previousVs = currentVs;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  // Input shape: [batchSize, sampleCount, 2]
  // Output shape [batchSize, sampleCount]
  static getSynthOutputTensor(input: Tensor) {
    const [batchSize, sampleCount, featureCount] = input.shape;
    const timeSlices = tf.split(input, sampleCount, 1);
    let previousTheta = tf.zeros([batchSize]);

    for (const slice of timeSlices) {
      const [note, amplitude] = tf.split(input,
        /*number of splits=*/2, /*axis=*/2);

      const omega = tf.mul(tf.pow(Math.pow(2, 1 / 12), tf.sub(note, 69)), 440);

    }

  }

  // Input shape: [batchSize, sampleCount]
  // Alpha shape: [frequencyResolution]
  // Output shape: [ batchSize, sampleCount, frequencyResolution]
  static getDampedSpringOutputTensor(input: Tensor, k: Tensor): Tensor {
    const [batchSize, sampleCount] = input.shape;
    const [frequencyResolution] = k.shape;
    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);

    const combinedTensors: Tensor[] = [];

    let previousXs = tf.zeros([batchSize, frequencyResolution]);
    let previousVs = tf.zeros([batchSize, frequencyResolution]);

    let previousYs = tf.zeros([batchSize, frequencyResolution]);
    // TODO: 0.02 here is actually related to how long it takes to
    // distinguish the sound... i.e. the boundary between rhythm and tone.
    const alpha = 0.02;
    for (let t = 0; t < sampleCount; ++t) {
      const currentAs = tf.add(
        tf.neg(tf.mul(previousXs, k)),
        tf.mul(tf.sub(timeSlices[t], previousXs), 0.0001));
      const currentVs = tf.add(previousVs, currentAs);
      const currentXs = tf.add(currentVs, previousXs);

      const currentYs = tf.add(
        tf.mul(tf.square(currentXs), alpha),
        tf.mul(previousYs, (1.0 - alpha)));

      combinedTensors.push(
        tf.reshape(tf.mul(currentYs, 100), [batchSize, 1, frequencyResolution]));

      previousXs = currentXs;
      previousVs = currentVs;
      previousYs = currentYs;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  // Input shape: [batchSize, sampleCount]
  // Alpha shape: [frequencyResolution]
  // Output shape: [ batchSize, sampleCount, frequencyResolution]
  static getBrokenBandPassOutputTensor(input: Tensor, alpha: Tensor, beta: Tensor): Tensor {
    const [batchSize, sampleCount] = input.shape;
    const [frequencyResolution] = alpha.shape;

    // Each time slice is shaped [batchSize, 1]
    const timeSlices = tf.split(input, sampleCount, 1);
    timeSlices.unshift(tf.zeros([batchSize, 1]));

    const alpha2 = tf.reshape(alpha, [1, frequencyResolution]);
    const beta2 = tf.reshape(beta, [1, frequencyResolution]);
    const oneMinusAlpha2 = tf.sub(tf.ones(alpha2.shape), alpha2);

    const combinedTensors: Tensor[] = [];

    const x1Coef = tf.add(alpha2,
      tf.div(tf.add(tf.ones(beta2.shape), beta2), 2.0));
    const x0Coef = tf.neg(
      tf.div(tf.add(tf.ones(beta2.shape), beta2), 2.0));
    const y0Coef = tf.sub(beta2, alpha2);

    let y0 = tf.zeros([batchSize, 1]);
    for (let t = 0; t < sampleCount; ++t) {
      const x0 = timeSlices[t];
      const x1 = timeSlices[t + 1];

      const x1Part = tf.matMul(x1, x1Coef);
      const x0Part = tf.mul(x0, x0Coef);
      const y0part = tf.mul(y0, y0Coef);

      const y1 = tf.add(tf.add(x1Part, x0Part), y0part);
      combinedTensors.push(tf.reshape(y1, [batchSize, 1, frequencyResolution]));
      y0 = y1;
    }

    const output = tf.concat(combinedTensors, 1);
    return output;
  }

  // y1, y2 shape: [ batchSize, sampleCount, frequencyResolution ]
  // output shape: [ batchSize ]
  static getCostFunction(y1: Tensor, y2: Tensor) {
    console.assert(y1.shape.length === y2.shape.length,
      `${y1.shape} != ${y2.shape}`);
    const [batchSize, sampleCount, frequencyResolution] = y1.shape;
    let diff: Tensor = tf.sub(y1, y2);
    if (y1.shape[1] > 1000) {
      diff = tf.slice(diff, [0, 500, 0], [-1, -1, -1]);
    }
    const square = tf.square(diff);
    const sumOfSquares = tf.sum(square, [1, 2]);
    console.assert(sumOfSquares.shape.length === 1, sumOfSquares.shape);
    console.assert(sumOfSquares.shape[0] === batchSize, sumOfSquares.shape);
    const mse = tf.mul(sumOfSquares, 1.0 / sampleCount / frequencyResolution);
    return mse;
  }

  static betaFromFreq(freqHz: number, sampleRate: number) {
    const radiansPerSample = 2 * Math.PI * freqHz / sampleRate;
    const beta = (1 - Math.sin(radiansPerSample)) / Math.cos(radiansPerSample);
    return beta;
  }

  // minFrequency: Lowest frequncy in Hz
  // stepSize: measured in half-steps.
  // stepCount: number of steps.
  static betaArray(minFrequency: number, stepSize: number, stepCount: number,
    sampleRate: number) {
    const result: number[] = [];
    const stepPower = Math.pow(Math.pow(2, 1 / 12), stepSize);
    for (let i = 0; i < stepCount; ++i) {
      let freq = minFrequency * Math.pow(stepPower, i);
      const alpha = Model.betaFromFreq(freq, sampleRate);
      result.push(alpha);
    }
    return result;
  }

  static earBetaArray(sampleRate: number) {
    // 16 Hz is below human perception
    // 2.0 half steps per band
    // 64 bands
    // This puts the top frequency pretty close to the Nyquist frquency
    // for 44kHz sampling.
    return this.betaArray(10, 2.0, 64, sampleRate);
  }

  static earBetaTensor(sampleRate: number) {
    const betaArray = this.earBetaArray(sampleRate);
    return tf.tensor(betaArray, [betaArray.length], "float32");
  }


  static alphaFromFreq(freqHz: number, sampleRate: number) {
    const radiansPerSample = 2 * Math.PI * freqHz / sampleRate;
    const y = 1 - Math.cos(radiansPerSample);
    const alpha = -y + Math.sqrt(y * y + 2 * y);
    return alpha;
  }

  // minFrequency: Lowest frequncy in Hz
  // stepSize: measured in half-steps.
  // stepCount: number of steps.
  static alphaArray(minFrequency: number, stepSize: number, stepCount: number,
    sampleRate: number) {
    const result: number[] = [];
    const stepPower = Math.pow(Math.pow(2, 1 / 12), stepSize);
    for (let i = 0; i < stepCount; ++i) {
      let freq = minFrequency * Math.pow(stepPower, i);
      const alpha = Model.alphaFromFreq(freq, sampleRate);
      result.push(alpha);
    }
    return result;
  }

  static earAlphaArray(sampleRate: number) {
    // 16 Hz is below human perception
    // 2.0 half steps per band
    // 64 bands
    // This puts the top frequency pretty close to the Nyquist frquency
    // for 44kHz sampling.
    return this.alphaArray(10, 2.0, 64, sampleRate);
  }

  static earAlphaTensor(sampleRate: number) {
    const alphaArray = this.earAlphaArray(sampleRate);
    return tf.tensor(alphaArray, [alphaArray.length], "float32");
  }

  static kFromFreq(freqHz: number, sampleRate: number) {
    const radiansPerSample = 2 * Math.PI * freqHz / sampleRate;
    const k = radiansPerSample * radiansPerSample;
    return k;
  }

  // minFrequency: Lowest frequncy in Hz
  // stepSize: measured in half-steps.
  // stepCount: number of steps.
  static kArray(minFrequency: number, stepSize: number, stepCount: number,
    sampleRate: number) {
    const result: number[] = [];
    const stepPower = Math.pow(Math.pow(2, 1 / 12), stepSize);
    for (let i = 0; i < stepCount; ++i) {
      let freq = minFrequency * Math.pow(stepPower, i);
      const alpha = Model.kFromFreq(freq, sampleRate);
      result.push(alpha);
    }
    return result;
  }

  static earKArray(sampleRate: number) {
    // 16 Hz is below human perception
    // 2.0 half steps per band
    // 64 bands
    // This puts the top frequency pretty close to the Nyquist frquency
    // for 44kHz sampling.
    return this.kArray(10, 2.0, 64, sampleRate);
  }

  static earKTensor(sampleRate: number) {
    const kArray = Model.earKArray(sampleRate);
    return tf.tensor(kArray, [kArray.length], "float32");
  }


  static embeddedCost(input: Tensor,
    batchSize: number, sampleCount: number, sampleRate: number) {
    const alphas = Model.earAlphaTensor(sampleRate);
    const betas = Model.earBetaTensor(sampleRate);
    const output = Model.getBandPassOutputTensor(input, alphas);
  }

}