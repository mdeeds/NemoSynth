import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import { AudioTensor } from './audioTensor';
import { Model } from './model';
import { Render } from './render';

export class SmokeTest {
  static smoke() {
    SmokeTest.smokeCost();
    SmokeTest.smokeAlpha();
    AudioTensor.loadClip('wav/BeepOnly.wav').then((clip) => {
      SmokeTest.smokeAudioLow(clip);
      SmokeTest.smokeAudioSpring(clip);
      SmokeTest.smokeAudioDampedSpring(clip);
      // SmokeTest.smokeGradientLow(clip);
      // SmokeTest.smokeGradientSpring(clip);
      SmokeTest.smokeGradientDampedSpring(clip);
    });
  }

  static smokeCost() {
    const t1 = tf.tensor([[1, 2, 3], [4, 5, 6]]);
    console.log(t1.shape);
    const [batchSize, sampleCount] = t1.shape;
    console.log(`Batch Size: ${batchSize}`);
    console.log(`SampleCount: ${sampleCount}`);

    const alpha = tf.tensor([0.1, 0.2, 0.3, 0.4]);
    const lowPass = Model.getBandPassOutputTensor(t1, alpha)
    console.log(`Low pass shape: ${lowPass.shape}`);

    const t2 = tf.tensor([[1, 2, 3], [4, 5, 6]]);
    const lowPass2 = Model.getBandPassOutputTensor(t1, alpha)

    const mse = Model.getCostFunction(lowPass, lowPass2);
    console.log(`Cost shape: ${mse.shape}`);
  }

  static smokeAlpha() {
    const sampleRate = 48000;
    const freqs = [sampleRate / 2, 256, 128, 64];
    for (const freq of freqs) {
      console.log(
        `freq: ${freq}; alpha: ${Model.alphaFromFreq(freq, sampleRate)}`);
    }

    // const earAlpha = Model.earAlphaArray(sampleRate);
    // console.log(`Ear alpha array: ${JSON.stringify(earAlpha)}`);
    const earTensor = Model.earAlphaTensor(sampleRate);
    const maxVal = tf.max(earTensor).dataSync()[0];
    const minVal = tf.min(earTensor).dataSync()[0];
    console.log(`Ear alpha range: ${minVal} to ${maxVal}`);

  }

  static smokeAudioLow(clip: AudioBuffer) {
    console.log(`Sample count: ${clip.length}`);
    console.log(`Sample rate: ${clip.sampleRate}`);

    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const t = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Clip");
    Render.addCanvases(t);
    console.log(`Clip shape: ${t.shape}`);
    const startMs = window.performance.now();
    const earAlpha = Model.earAlphaTensor(clip.sampleRate);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    const embedded = Model.getLowPassOutputTensor(t, earAlpha);
    console.log(`Elapsed: ${window.performance.now() - startMs}ms`);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    console.log(`Embedded shape: ${embedded.shape}`);

    Render.message("Low Pass");
    Render.addCanvases(embedded);
  }

  static smokeAudioSpring(clip: AudioBuffer) {
    console.log(`Sample count: ${clip.length}`);
    console.log(`Sample rate: ${clip.sampleRate}`);

    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const t = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Clip");
    Render.addCanvases(t);
    console.log(`Clip shape: ${t.shape}`);
    const startMs = window.performance.now();
    const earK = Model.earKTensor(clip.sampleRate);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    const embedded = Model.getSpringOutputTensor(t, earK);
    console.log(`Elapsed: ${window.performance.now() - startMs}ms`);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    console.log(`Embedded shape: ${embedded.shape}`);

    Render.message("Spring model");
    Render.addCanvases(embedded);
  }

  static smokeAudioDampedSpring(clip: AudioBuffer) {
    console.log(`Sample count: ${clip.length}`);
    console.log(`Sample rate: ${clip.sampleRate}`);

    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const t = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Clip");
    Render.addCanvases(t);
    console.log(`Clip shape: ${t.shape}`);
    const startMs = window.performance.now();
    const earK = Model.earKTensor(clip.sampleRate);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    const embedded = Model.getDampedSpringOutputTensor(t, earK);
    console.log(`Elapsed: ${window.performance.now() - startMs}ms`);
    console.log(`Memory: ${JSON.stringify(tf.memory())}`);
    console.log(`Embedded shape: ${embedded.shape}`);

    Render.message("Spring model");
    Render.addCanvases(embedded);
  }

  static smokeGradientLow(clip: AudioBuffer) {
    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const rawTarget = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Gradient test");
    Render.addCanvases(rawTarget);
    const earAlpha = Model.earAlphaTensor(clip.sampleRate);
    const target = Model.getLowPassOutputTensor(rawTarget, earAlpha);
    Render.addCanvases(target);

    let current = tf.randomUniform(rawTarget.shape, -0.01, 0.01);

    function f(rawX: Tensor) {
      const y = Model.getLowPassOutputTensor(rawX, earAlpha);
      return Model.getCostFunction(target, y);
    }
    const gradientApplier = tf.grad(f);
    const previousError = 0;
    for (let iteration = 0; iteration < 10; ++iteration) {
      Render.addCanvases(current);
      const gradient: Tensor = gradientApplier(current);
      console.log(`MinGrad: ${gradient.min().dataSync()}`);
      console.log(`MaxGrad: ${gradient.max().dataSync()}`);
      current = tf.add(current, tf.mul(gradient, 1000));
      const currentData = current.dataSync();
      current = tf.tensor(currentData, current.shape);
    }
  }

  static smokeGradientSpring(clip: AudioBuffer) {
    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const rawTarget = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Gradient test");
    Render.addCanvases(rawTarget);
    const earK = Model.earKTensor(clip.sampleRate);
    const target = Model.getSpringOutputTensor(rawTarget, earK);
    Render.addCanvases(target);

    let current = tf.randomUniform(rawTarget.shape, -0.1, 0.1);

    function f(rawX: Tensor) {
      const y = Model.getSpringOutputTensor(rawX, earK);
      return Model.getCostFunction(target, y);
    }
    const gradientApplier = tf.grad(f);
    let learningRate = 0.01;
    let currentData = null;
    for (let iteration = 0; iteration < 50; ++iteration) {
      Render.addCanvases(current);
      const finalValue = Model.getSpringOutputTensor(current, earK);
      Render.addCanvases(finalValue);
      const gradient: Tensor = gradientApplier(current);
      const currentError = f(current).dataSync()[0];
      console.log(`SSE: ${currentError}`);
      console.log(`MinGrad: ${gradient.min().dataSync()}`);
      console.log(`MaxGrad: ${gradient.max().dataSync()}`);
      console.log(`Learning rate: ${learningRate}`);
      current = tf.add(current, tf.mul(gradient, learningRate));
      currentData = current.dataSync();
      current = tf.tensor(currentData, current.shape);
    }
  }

  static smokeGradientDampedSpring(clip: AudioBuffer) {
    const offset = 0;
    const sampleCount = 2000;
    // Reshape into a batch size of 1.
    const rawTarget = tf.reshape(
      AudioTensor.getTensor(clip, offset, sampleCount), [1, sampleCount]);
    Render.message("Gradient test");
    Render.addCanvases(rawTarget);
    const earK = Model.earKTensor(clip.sampleRate);
    const target = Model.getDampedSpringOutputTensor(rawTarget, earK);
    Render.addCanvases(target);

    let current = tf.randomUniform(rawTarget.shape, -0.1, 0.1);

    function f(rawX: Tensor) {
      const y = Model.getSpringOutputTensor(rawX, earK);
      return Model.getCostFunction(target, y);
    }
    const gradientApplier = tf.grad(f);
    let learningRate = 0.01;
    let currentData = null;
    for (let iteration = 0; iteration < 50; ++iteration) {
      Render.addCanvases(current);
      const finalValue = Model.getSpringOutputTensor(current, earK);
      Render.addCanvases(finalValue);
      const dy = f(current);
      const gradient: Tensor = gradientApplier(current, dy);
      const currentError = dy.dataSync()[0]
      console.log(`SSE: ${currentError}`);
      console.log(`MinGrad: ${gradient.min().dataSync()}`);
      console.log(`MaxGrad: ${gradient.max().dataSync()}`);
      console.log(`Learning rate: ${learningRate}`);
      current = tf.sub(current, tf.mul(gradient, learningRate));
      currentData = current.dataSync();
      current = tf.tensor(currentData, current.shape);
    }
  }
}