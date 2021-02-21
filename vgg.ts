import { imag, norm, Tensor } from "@tensorflow/tfjs";
import * as tf from '@tensorflow/tfjs';
import { Log } from './log';
import { DEFAULT_MAX_VERSION } from "tls";

export class VGG {
  private ioCanvas: HTMLCanvasElement;
  private ioTensor: Tensor;
  private targetNumber: number;
  private model: tf.LayersModel;
  private expectedYTensor: Tensor;
  private g: Function;
  constructor() {
    Log.info(tf.memory());
    this.ioCanvas = document.createElement('canvas');
    this.ioCanvas.width = 224;
    this.ioCanvas.height = 224;

    setTimeout(() => {
      const body = document.getElementsByTagName('body')[0];
      body.innerText = "";
      body.appendChild(this.ioCanvas);
      const goButton = document.createElement('div');
      goButton.innerText = "GO!!!!";
      body.appendChild(goButton);
      goButton.addEventListener('click', () => {
        this.start();
      })

      const iterateButton = document.createElement('div');
      iterateButton.innerText = 'ITERATE';
      body.appendChild(iterateButton);
      iterateButton.addEventListener('click', () => {
        this.iterate(10);
      });
    }, 10);
    const ctx = this.ioCanvas.getContext('2d');
    const grad = ctx.createLinearGradient(112, 112 - 10, 112, 112 + 10);
    grad.addColorStop(0, 'skyblue');
    grad.addColorStop(1, 'darkolivegreen')
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 224, 224);

    const grad2 = ctx.createRadialGradient(112 + 10, 112 + 10, 10, 112, 112, 60);
    grad2.addColorStop(0, 'white');
    grad2.addColorStop(1, 'lightgray');
    ctx.fillStyle = grad2;
    ctx.beginPath();
    ctx.arc(112, 112, 90, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = 'black'
    this.targetNumber = Math.trunc(Math.random() * 1000);
    ctx.textAlign = "center";
    ctx.fillText(this.targetNumber.toFixed(0), 112, 20);
  }

  start() {
    const expectedY = new Float32Array(1000);
    expectedY[this.targetNumber] = 1.0;
    this.expectedYTensor = tf.tensor1d(expectedY);

    Log.info('Loading...');
    tf.loadLayersModel('vgg/model.json')
      .then((model: tf.LayersModel) => {
        this.model = model;
        Log.info('Loaded.');
        // model.summary(null, null,
        //   (message) => { Log.info(message); });
        const f = (x: Tensor) => { return this.model.predict(x) as Tensor<tf.Rank> };
        this.g = tf.grad(f);
        this.ioTensor = this.getTensorFromCanvas();
        setTimeout(() => { this.iterate(10) }, 0);
      })
      .catch((e) => {
        console.error(e);
      });
  }

  iterate(iterationCount: number) {
    if (iterationCount === 0) {
      Log.info("Done.");
      return;
    }
    Log.info("Iterating...");
    Log.info(tf.memory());

    const currentYTensor = this.model.predict(this.ioTensor) as Tensor<tf.Rank>;
    const dyTensor = tf.sub(this.expectedYTensor, currentYTensor);
    const deltaTensor = this.g(this.ioTensor, dyTensor);
    const absTensor = tf.abs(deltaTensor);
    const maxTensor = tf.max(absTensor);
    const scaleTensor = tf.div(0.2, maxTensor);
    const mulTensor = tf.mul(deltaTensor, scaleTensor);
    const rndTensor = tf.randomUniform(this.ioTensor.shape, -0.005, 0.005);
    const add1 = tf.add(this.ioTensor, rndTensor);
    const add2 = tf.add(add1, mulTensor);
    const result = tf.clipByValue(add2, 0, 1);
    Log.info('Getting result value...')
    result.data()
      .then((resultData) => {
        this.renderToCanvas(resultData);
        this.ioTensor.dispose();
        currentYTensor.dispose();
        dyTensor.dispose();
        deltaTensor.dispose();
        absTensor.dispose();
        maxTensor.dispose();
        scaleTensor.dispose();
        mulTensor.dispose();
        rndTensor.dispose();
        add1.dispose();
        add2.dispose();
        result.dispose();
        Log.info('Assigning value.  Ready to start again.');
        this.ioTensor = tf.tensor(resultData, [1, 224, 224, 3]);
        setTimeout(() => { this.iterate(iterationCount - 1); });
      });
  }

  materialize(t: Tensor): Promise<Tensor> {
    Log.info("Materializing results...");
    return new Promise((resolve, reject) => {
      t.data().then((data) => {
        const result = tf.tensor(data, t.shape);
        // t.dispose();
        resolve(result);
      });
    });
  }

  getTensorFromCanvas(): Tensor {
    const x =
      tf.div(
        tf.browser.fromPixels(this.ioCanvas)
          .resizeBilinear([224, 224]).toFloat(),
        255.0);
    return tf.reshape(x, [1, 224, 224, 3]);
  }

  renderToCanvas(ioData: tf.TypedArray) {
    const pixelData = new Uint8ClampedArray(4 * 224 * 224);
    let minV = 1000;
    let maxV = -1000;
    for (let y = 0; y < 224; ++y) {
      for (let x = 0; x < 224; ++x) {
        for (let c = 0; c < 3; ++c) {
          const v = ioData[c + x * 3 + y * 3 * 224] * 255;
          minV = Math.min(v, minV);
          maxV = Math.max(v, maxV);
          pixelData[c + x * 4 + y * 4 * 224] = v;
        }
        pixelData[3 + x * 4 + y * 4 * 224] = 255;
      }
    }
    Log.info(`Pixel value range: ${minV} to ${maxV}`);
    const ctx = this.ioCanvas.getContext('2d');
    ctx.putImageData(new ImageData(pixelData, 224, 224), 0, 0);
  }
}